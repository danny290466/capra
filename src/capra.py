"""CAPRA — Capacity-Aware Parity Risk Allocation (method core).

Problem
-------
Per language `l` a policy is a threshold `tau_l`; on the calibration split it induces
    s_l  review share   (fraction of language-l items routed to humans)
    r_l(s) risk of the kept set when the share `s` is reviewed   (decreasing in `s`)
A human team can review a hard capacity `C` of the (balanced, w_l = 1/2) traffic:

    (BUDGET)   Sum_l w_l * s_l  =  C

Three quantities matter in deployment:
    DISPARITY    D = | r_EN(s_EN) - r_VI(s_VI) |      who is left more exposed
    COMPOSITION  K = w_VI * s_VI / C                   who has to read the queue
    CAPACITY     the budget itself

With ONE threshold per language and a binding budget there is a single degree of freedom: writing
the allocation as `alpha = w_EN s_EN / C` (so `K = 1 - alpha` exactly), the policy is fully
determined by `alpha`. Hence:

    PROPOSITION. Under a binding budget, exact parity (D = 0) and exact composition (K = K*) are
    jointly attainable only if the two languages' risk curves happen to satisfy
    D(1 - K*) = 0; in general at most two of {budget, parity, composition} can be met exactly.

CAPRA returns the whole achievable set: sweep `alpha`, evaluate (K, D, r_EN, r_VI), and report
  * the parity-first point  (min D, composition free),
  * the composition-first point (K = K*, disparity whatever it is),
  * the PRICE OF COMPOSITION  = D at K*  -  min D,
  * the PRICE OF PARITY       = |K - K*| at the parity-first point,
  * and the same under a *cost-weighted* capacity (review TIME, per-language cost c_l), where the
    budget becomes Sum_l w_l c_l s_l = C — the lever that restores feasibility for one of the two.

CPU-only: reads probs.npz artefacts. Usage:
  python capra.py --out-root outputs/killtest --langs en,vi --mode NAT
                  --out notes/capra_RESULT.md --json outputs/killtest/capra.json
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conra_parity as CP          # noqa: E402
import killtest_analyze as K       # noqa: E402

C_DEFAULT = 0.05
ALPHA_GRID = np.linspace(0.0, 1.0, 201)


def review_risk_curve(conf, err):
    """Risk of the kept set as a function of the REVIEW share.

    s = 1 - coverage, so the curve is read at increasing s: reviewing more means keeping fewer,
    higher-confidence items, thus lower risk. Returns arrays sorted by increasing s.
    """
    order = np.argsort(-conf, kind="mergesort")
    c = conf[order]
    e = err[order]
    n = len(e)
    k = np.arange(1, n + 1, dtype=float)
    cov = k / n
    risk = np.cumsum(e) / k
    s = 1.0 - cov
    # enforce monotone non-increasing risk in s (envelope), so interpolation is well behaved
    risk_env = np.minimum.accumulate(risk[::-1])[::-1]
    order2 = np.argsort(s, kind="mergesort")
    return s[order2], risk_env[order2]


def risk_at_share(curve, s):
    """r_l(s) with clipping to the observed range (s below the minimum share is unreachable)."""
    xs, ys = curve
    if s <= xs[0]:
        return float(ys[0])
    if s >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(s, xs, ys))


def feasible_alpha(budget, costs, weights, first, second):
    """The exact domain of the allocation parameter: the alpha for which neither review share exceeds 1.

    s_first = alpha * C / (w_first c_first) <= 1  =>  alpha <= w_first c_first / C
    s_second = (1 - alpha) * C / (w_second c_second) <= 1  =>  alpha >= 1 - w_second c_second / C
    Enumerating this interval is what Appendix A of the manuscript states and what Algorithm 1 does;
    sampling [0, 1] and discarding the clipped points samples the same set unevenly and makes the
    reported prices depend on the grid.
    """
    wf, ws = weights
    hi = min(1.0, wf * costs[first] / budget)
    lo = max(0.0, 1.0 - ws * costs[second] / budget)
    return float(min(lo, hi)), float(max(lo, hi))


def sweep(curves, budget, langs, costs, weights=(0.5, 0.5), alpha_grid=None):
    """Evaluate (alpha, K, D, r_l, load) along the allocation sweep of a (possibly cost-weighted) budget.

    `langs = (first, second)`: `alpha` is the share of the budget spent on the FIRST language, so
    `K_B = 1 - alpha` up to the per-language review costs. `weights` = traffic share of each language
    (default 1/2 each; the mix ablation varies it). The sweep runs over the FEASIBLE interval of
    alpha, so every point returned spends the budget exactly and no point is clipped.
    """
    first, second = langs
    wf, ws = weights
    lo, hi = feasible_alpha(budget, costs, weights, first, second)
    if alpha_grid is None:
        alpha_grid = np.linspace(lo, hi, 201)
    out = []
    for a in alpha_grid:
        s = {first: a * budget / (wf * costs[first]),
             second: (1.0 - a) * budget / (ws * costs[second])}
        for l in langs:
            s[l] = min(max(s[l], 0.0), 1.0)
        load = wf * costs[first] * s[first] + ws * costs[second] * s[second]
        r = {l: risk_at_share(curves[l], s[l]) for l in langs}
        q = {first: wf * costs[first] * s[first], second: ws * costs[second] * s[second]}
        k_budget = q[second] / max(load, 1e-12)
        # K is the ITEM share of the queue (who is read), the quantity the composition target K* is
        # about; the cost-weighted share is a different quantity and gets its own name (review R3-11)
        items = {first: wf * s[first], second: ws * s[second]}
        k_item = items[second] / max(items[first] + items[second], 1e-12)
        out.append({"alpha": float(a), "s": dict(s), "risk": r,
                    "D_abs": abs(r[first] - r[second]), "D_signed": r[first] - r[second],
                    "K": float(k_item), "K_budget": float(k_budget), "load": float(load),
                    "r_bar": float(wf * r[first] + ws * r[second]),
                    "clipped": bool(s[first] >= 1.0 or s[second] >= 1.0)})
    return out


def risk_ceiling_point(front, r_max):
    """Parity under a risk ceiling: min D subject to r_bar <= r_max (review finding E1).

    Minimising |r_a - r_b| on its own is satisfied by any allocation that makes both groups equally
    bad, so the parity-first point can sit at a high absolute risk. This returns the allocation with
    the smallest disparity among those whose weighted average residual risk stays under `r_max`,
    or None when the ceiling excludes the whole admissible set.
    """
    ok = [p for p in front if not p["clipped"] and p["r_bar"] <= r_max + 1e-12]
    if not ok:
        return None
    return min(ok, key=lambda p: p["D_abs"])



def pooled_threshold(cal, langs, budget, unit_costs=(1.0, 1.0), weights=(0.5, 0.5)):
    """Threshold of the pooled rule at review load `budget`, in THIS variant's budget unit.

    The pooled rule is ONE threshold for both languages, so it is the pair (tau, tau), and "a
    pooled review share C of the mixed traffic" is the C-quantile of the traffic-weighted mixture
    of the two confidence distributions: each item carries weight `w_l * c_l / n_l`, where `w_l` is
    the traffic share and `c_l` the per-item review cost of the variant's budget unit (1 for both
    when the budget counts items, (1, 1.5) when it counts review time). The returned tau therefore
    satisfies `sum_l w_l c_l s_l = budget` exactly, which is the same constraint the sweep obeys.

    Pooling the two RAW arrays instead — what `conra_parity.rule_thresholds("GLOBAL", ...)` does —
    weights each language by its ITEM COUNT: on the EN/VI calibration split that is 25 000 English
    items against 13 360 Vietnamese ones, so the pooled rule ran at a realised equal-weight load of
    0.0543 for a nominal C = 0.05 (0.1088 at 0.10, 0.2179 at 0.20). Comparing its average risk with
    the family's at load C then compared two different constraints, which is what made the
    risk-ceiling program look infeasible on the whole family (external review round 10, item 1: the
    floor at 6.55 pp against a ceiling of 6.43 pp). `baseline_rules.traj_design_global` uses this
    same weighted quantile for the GLOBAL row of the matched-load audit, and the two agree: at
    C = 0.05 both give tau = 0.7003, design D = 4.50 pp, K = 0.631.
    """
    n = {l: len(cal[l]["conf"]) for l in langs}
    w = np.concatenate([np.full(n[l], weights[i] * unit_costs[i] / n[l]) for i, l in enumerate(langs)])
    concat = np.concatenate([cal[l]["conf"] for l in langs])
    order = np.argsort(concat, kind="mergesort")
    cw = np.cumsum(w[order]) / w.sum()
    level = budget / float(sum(weights[i] * unit_costs[i] for i in range(len(langs))))
    return float(concat[order][np.searchsorted(cw, level, side="left").clip(0, len(cw) - 1)])


def load_of(tau, cal, langs, weights=(0.5, 0.5)):
    """Realised review load of one threshold pair, in the paper's budget units (equal weights)."""
    return float(sum(weights[i] * float((cal[l]["conf"] < tau[l]).mean())
                     for i, l in enumerate(langs)))


def endpoints(front, k_star):
    feasible = [p for p in front if not p["clipped"]] or front
    parity = min(feasible, key=lambda p: p["D_abs"])
    comp = min(feasible, key=lambda p: abs(p["K"] - k_star))
    return parity, comp


def measure_on_test(tau, test, langs, costs=None, weights=(0.5, 0.5)):
    """Apply thresholds fitted on calibration to the test split — same convention as the CONRA tables.

    `load` is the ITEM share of traffic reviewed. When the budget is denominated in review time
    (cost-aware variant) the item share is lower than C by construction, so the budget-unit load is
    reported separately as `load_budget` (review findings R3-12, R2-05).
    """
    out = {}
    for l in langs:
        c = CP.cov_of_tau(test[l]["conf"], tau[l])
        out[l] = {"coverage": c, "risk": CP.risk_of_tau(test[l]["conf"], test[l]["err"], tau[l])}
    load = 1.0 - 0.5 * (out[langs[0]]["coverage"] + out[langs[1]]["coverage"])
    q = {l: 0.5 * (1.0 - out[l]["coverage"]) for l in langs}
    out["load"] = load
    out["D_abs"] = abs(out[langs[0]]["risk"] - out[langs[1]]["risk"])
    out["D_signed"] = out[langs[0]]["risk"] - out[langs[1]]["risk"]
    out["r_bar"] = 0.5 * (out[langs[0]]["risk"] + out[langs[1]]["risk"])
    out["K"] = q[langs[1]] / (q[langs[0]] + q[langs[1]]) if (q[langs[0]] + q[langs[1]]) > 0 else float("nan")
    if costs is not None:
        wa, wb = weights
        cw = {langs[0]: wa * costs[langs[0]], langs[1]: wb * costs[langs[1]]}
        qb = {l: cw[l] * (1.0 - out[l]["coverage"]) for l in langs}
        out["load_budget"] = qb[langs[0]] + qb[langs[1]]
        out["K_budget"] = (qb[langs[1]] / (qb[langs[0]] + qb[langs[1]])
                           if (qb[langs[0]] + qb[langs[1]]) > 0 else float("nan"))
    return out


def tau_from_share(cal_conf, s):
    """Threshold that reviews share `s` of a language (keeps the top 1-s fraction by confidence)."""
    return float(np.quantile(cal_conf, min(max(s, 0.0), 1.0), method="lower"))


def summarise(front, k_star):
    parity, comp = endpoints(front, k_star)
    return {
        "parity_first": {k: parity[k] for k in ("alpha", "s", "risk", "D_abs", "K", "load")},
        "composition_first": {k: comp[k] for k in ("alpha", "s", "risk", "D_abs", "K", "load")},
        "price_of_composition": comp["D_abs"] - parity["D_abs"],
        "price_of_parity": abs(parity["K"] - k_star),
        "feasible_both": bool(comp["D_abs"] < 1e-3 and abs(comp["K"] - k_star) < 1e-3),
        "min_D_achievable": parity["D_abs"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--langs", default="en,vi")
    ap.add_argument("--mode", default="NAT")
    ap.add_argument("--budgets", default="0.05,0.10,0.20")
    ap.add_argument("--composition-target", type=float, default=0.5,
                    help="desired share of the review queue coming from the second language")
    ap.add_argument("--cost-second", "--cost-vi", dest="cost_second", type=float, default=1.5,
                    help="per-item review cost of the SECOND language (cost-aware variant)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    a, b = [s.strip() for s in args.langs.split(",")]
    CP.LANGS = [a, b]
    seed_dirs = sorted(d for d in glob.glob(os.path.join(args.out_root, "seed*"))
                       if os.path.exists(os.path.join(d, "probs.npz")))
    if not seed_dirs and os.path.exists(os.path.join(args.out_root, "probs.npz")):
        seed_dirs = [args.out_root]
    assert seed_dirs

    d = CP.prep(seed_dirs, args.mode) if (a, b) == ("en", "vi") else None
    if d is None:                                   # generic pair: reuse the pair loader
        import pair_parity as PP
        d = PP.build_pair(seed_dirs, a, b)
    curves = {a: review_risk_curve(d["cal"][a]["conf"], d["cal"][a]["err"]),
              b: review_risk_curve(d["cal"][b]["conf"], d["cal"][b]["err"])}

    res = {"generated": datetime.now().isoformat(timespec="seconds"), "detector": args.out_root,
           "pair": [a, b], "mode": args.mode, "composition_target": args.composition_target,
           "costs": {a: 1.0, b: args.cost_second}, "budgets": {}, "fronts": {}}
    for C in [float(x) for x in args.budgets.split(",")]:
        for tag, cvi in (("cost_blind", 1.0), ("cost_aware", args.cost_second)):
            costs = {a: 1.0, b: cvi}
            front = sweep(curves, C, (a, b), costs)
            res["fronts"].setdefault(f"{C:.2f}", {})[tag] = [
                {k: p[k] for k in ("alpha", "D_abs", "K", "load", "clipped")} for p in front]
            m = summarise(front, args.composition_target)   # design-split prices
            # measure both endpoints on the TEST split, using thresholds implied by the allocation
            for key, pt in (("parity_first", m["parity_first"]), ("composition_first", m["composition_first"])):
                tau = {l: tau_from_share(d["cal"][l]["conf"], pt["s"][l]) for l in (a, b)}
                pt["test"] = measure_on_test(tau, d["test"], [a, b], costs=costs)
            # price computed on ONE split: after the endpoints are SELECTED on calibration, both are
            # measured on test, so the price must be a test-minus-test difference (review finding R2-02)
            m["price_of_composition_test"] = (m["composition_first"]["test"]["D_abs"]
                                              - m["parity_first"]["test"]["D_abs"])
            m["price_of_parity_test"] = abs(m["parity_first"]["test"]["K"] - args.composition_target)
            # the printed price is the TEST-minus-TEST difference; the design-split difference is kept
            # only for transparency and must never be the number in a table (review findings R2-02, R3-03)
            m["price_of_composition_design"] = m["price_of_composition"]
            m["price_of_composition"] = m["price_of_composition_test"]
            m["feasible_both_test"] = bool(m["composition_first"]["test"]["D_abs"] < 0.01)
            # ---- risk-ceiling variant (review finding E1). Minimising |r_a - r_b| alone is
            # satisfied by making both groups equally bad; here parity may not buy itself with a
            # higher weighted average residual risk than the pooled (GLOBAL) threshold leaves at the
            # same budget, i.e. min_alpha D(alpha) s.t. r_bar(alpha) <= r_bar_GLOBAL(C).
            # The pooled threshold must be the one the matched-load audit uses: one threshold at the
            # C-quantile of the TRAFFIC-weighted confidence mix, and it must be matched in the SAME
            # unit as this variant's budget (items when costs are equal, review time when they are
            # not). Pooling the raw arrays instead weights each language by its item count, so the
            # pooled rule sat at a realised load of 0.0543 for a nominal C = 0.05 and the program
            # looked infeasible (external review round 10, item 1).
            t_g = pooled_threshold(d["cal"], (a, b), C, unit_costs=(costs[a], costs[b]))
            taus_g = {l: t_g for l in (a, b)}
            taus_g_unweighted = CP.rule_thresholds("GLOBAL", C, CP.curves_of(d["cal"]))
            r_max = 0.5 * (CP.risk_of_tau(d["cal"][a]["conf"], d["cal"][a]["err"], taus_g[a])
                           + CP.risk_of_tau(d["cal"][b]["conf"], d["cal"][b]["err"], taus_g[b]))
            r_max_unweighted = 0.5 * (
                CP.risk_of_tau(d["cal"][a]["conf"], d["cal"][a]["err"], taus_g_unweighted[a])
                + CP.risk_of_tau(d["cal"][b]["conf"], d["cal"][b]["err"], taus_g_unweighted[b]))
            # Section 3.4 states that the pooled threshold IS a point of the family, so the closed
            # admissible set used by the ceiling program is the sweep plus that point: the grid alone
            # is a discretisation (alpha step 0.005 = ~12 English items) and a jagged r_bar makes the
            # min over a grid exceed the value at a point that is admissible by construction. The
            # ceiling program is therefore feasible whenever the pooled point spends this variant's
            # budget, and what it costs is disparity, not existence.
            s_g = {l: float((d["cal"][l]["conf"] < taus_g[l]).mean()) for l in (a, b)}
            r_g = {l: CP.risk_of_tau(d["cal"][l]["conf"], d["cal"][l]["err"], taus_g[l]) for l in (a, b)}
            items_g = {a: 0.5 * s_g[a], b: 0.5 * s_g[b]}
            pooled_pt = {"alpha": 0.5 * costs[a] * s_g[a] / C, "s": s_g, "risk": r_g,
                         "D_abs": abs(r_g[a] - r_g[b]), "D_signed": r_g[a] - r_g[b],
                         "K": items_g[b] / max(items_g[a] + items_g[b], 1e-12),
                         "load": 0.5 * costs[a] * s_g[a] + 0.5 * costs[b] * s_g[b],
                         "r_bar": 0.5 * (r_g[a] + r_g[b]), "clipped": False,
                         "is_pooled": True}
            lo_c, hi_c = feasible_alpha(C, costs, (0.5, 0.5), a, b)
            front_ceiling = sweep(curves, C, (a, b), costs,
                                  alpha_grid=np.linspace(lo_c, hi_c, 2001)) + [pooled_pt]
            ceil_pt = risk_ceiling_point(front_ceiling, r_max)
            m["risk_ceiling"] = {"ceiling_rule": "pooled threshold at the same realised load (traffic-weighted, matched in this variant's budget unit)",
                                 "tau": t_g,
                                 "load_design": load_of(taus_g, d["cal"], (a, b), (0.5, 0.5)),
                                 "load_design_budget_unit": pooled_pt["load"],
                                 "r_max_design": r_max,
                                 # kept for the audit trail: the pre-round-10 unweighted pooled
                                 # threshold, the one that sat 0.43 pp above C
                                 "r_max_design_unweighted": r_max_unweighted,
                                 "load_design_unweighted": load_of(taus_g_unweighted, d["cal"], (a, b), (0.5, 0.5)),
                                 "pooled_point": {k: pooled_pt[k] for k in ("alpha", "D_abs", "K", "load", "r_bar")},
                                 "ceiling_grid_points": len(front_ceiling),
                                 "parity_first_r_bar_design": m["parity_first"]["risk"][a] * 0.5
                                 + m["parity_first"]["risk"][b] * 0.5,
                                 "parity_first_r_bar_test": m["parity_first"]["test"].get("r_bar"),
                                 "parity_first_under_ceiling": bool(
                                     (m["parity_first"]["risk"][a] * 0.5 + m["parity_first"]["risk"][b] * 0.5)
                                     <= r_max + 1e-12),
                                 "attainable": ceil_pt is not None}
            if ceil_pt is not None:
                tau_c = {l: tau_from_share(d["cal"][l]["conf"], ceil_pt["s"][l]) for l in (a, b)}
                m["risk_ceiling"].update({
                    "alpha": ceil_pt["alpha"], "r_bar_design": ceil_pt["r_bar"],
                    "D_design": ceil_pt["D_abs"], "K_design": ceil_pt["K"],
                    "test": measure_on_test(tau_c, d["test"], [a, b], costs=costs)})
                m["risk_ceiling"]["price_vs_parity_first_test"] = (
                    m["risk_ceiling"]["test"]["D_abs"] - m["parity_first"]["test"]["D_abs"])
            # ---- risk floor of the family: the allocation with the LOWEST weighted average residual
            # risk over the same closed admissible set as the ceiling program (the alpha sweep plus
            # the pooled point, which Section 3.4 puts on the curve). If even this point sits above
            # the pooled rule's average risk the ceiling program has no solution, and because the
            # pooled point is admissible at the same load that can only happen when the ceiling was
            # imposed at a different constraint.
            free = [p for p in front_ceiling if not p["clipped"]] or front_ceiling
            floor = min(free, key=lambda p: p["r_bar"])
            tau_f = {l: tau_from_share(d["cal"][l]["conf"], floor["s"][l]) for l in (a, b)}
            m["risk_floor"] = {"min_r_bar_design": floor["r_bar"], "D_design": floor["D_abs"],
                               "K_design": floor["K"], "alpha": floor["alpha"],
                               "is_pooled_point": bool(floor.get("is_pooled")),
                               "test": measure_on_test(tau_f, d["test"], [a, b], costs=costs)}
            res["budgets"].setdefault(f"{C:.2f}", {})[tag] = m
        print(f"[C={C:.2f}] done", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("json ->", args.json, flush=True)

    if args.out:
        A = [f"# CAPRA — capacity-aware parity vs queue composition ({a} vs {b}, {args.mode})\n",
             f"Generated {res['generated']} · detector `{args.out_root}` · composition target "
             f"K* = {args.composition_target:.2f} · queue cost of {b} = {args.cost_second}× per item "
             f"(cost-aware variant). Thresholds/curves fitted on the calibration split.\n",
             "\n`K` = share of the review queue that comes from the second language; `D` = |r_" + a +
             " − r_" + b + "| in pp. `parity-first` minimises D; `composition-first` hits K*; the two are "
             "generally different points (proposition in the module docstring).\n",
             "\n| budget C | variant | parity-first D design (pp) | D test (pp) | its K test | composition-first D design (pp) | D test (pp) | its K test | price of composition (pp) | both feasible? |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for C, v in res["budgets"].items():
            for tag, m in v.items():
                pf, cf = m["parity_first"], m["composition_first"]
                A.append(f"| {C} | {tag} | {pf['D_abs']*100:.2f} | **{pf['test']['D_abs']*100:.2f}** | "
                         f"{pf['test']['K']:.3f} | {cf['D_abs']*100:.2f} | **{cf['test']['D_abs']*100:.2f}** | "
                         f"{cf['test']['K']:.3f} | {m['price_of_composition']*100:.2f} | "
                         f"{'YES' if m['feasible_both'] else 'no'} |")
        A += ["", "## Parity under a risk ceiling  (min D subject to r_bar <= r_bar(pooled at C))",
              "", "`r_bar` is the weighted average residual risk of the kept set at equal traffic "
              "weights. The ceiling is the average risk the pooled threshold leaves at the same "
              "REALISED load, so parity may not buy itself with a worse average outcome than the "
              "default a team operates today. The pooled threshold is one threshold at the "
              "C-quantile of the traffic-weighted confidence mix (`pooled_threshold`); `load` is the "
              "realised equal-weight review load it reviews, which the weighted quantile fixes at C "
              "by construction. `risk floor` is the point of the family with the lowest average "
              "risk; when even that sits above the ceiling the constrained program is infeasible on "
              "the whole family, and when it does not the program has a solution and `ceiling D test` "
              "gives the smallest disparity the ceiling admits.", "",
              "| budget C | variant | ceiling r_bar (design, pp) | ceiling load | "
              "parity-first r_bar (design, pp) | risk floor r_bar (design, pp) | floor D test (pp) | "
              "ceiling D test (pp) | penalty vs parity-first (pp) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for C, v in res["budgets"].items():
            for tag, m in v.items():
                rc = m.get("risk_ceiling") or {}
                fl = (m.get("risk_floor") or {}).get("test") or {}
                base = f"| {C} | {tag} | {rc.get('r_max_design', 0)*100:.2f} | " \
                       f"{rc.get('load_design', 0):.5f} | " \
                       f"{rc.get('parity_first_r_bar_design', 0)*100:.2f} | " \
                       f"{(m.get('risk_floor') or {}).get('min_r_bar_design', 0)*100:.2f} | "
                if not rc.get("attainable"):
                    A.append(base + f"**{fl.get('D_abs', float('nan'))*100:.2f}** | — | "
                                    f"infeasible |")
                    continue
                A.append(base + f"{fl.get('D_abs', float('nan'))*100:.2f} | "
                                f"**{rc['test']['D_abs']*100:.2f}** | "
                                f"{rc['price_vs_parity_first_test']*100:+.2f} |")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(A) + "\n")
        print("report ->", args.out, flush=True)


if __name__ == "__main__":
    main()
