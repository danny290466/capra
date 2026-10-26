"""Prior-art baselines for the parity study — is the CAPRA frontier actually better?

Reviewers of a method paper will ask for the strongest *published* family, not just the two rules we
generalise. The relevant published family is **group-conditional risk control**: instead of controlling
the pooled risk (standard CRC) or equalising coverage (equalized conformal), one calibrates a threshold
per group so that each group's own risk meets a target alpha — optionally with a finite-sample slack
(the Bonferroni / Hoeffding style correction that gives a worst-group guarantee).

Everything here is computed from the SAME dev-calibrated curves the rest of the project uses, so the
comparison is like-for-like:

  GLOBAL            one pooled threshold, review share C of the mixed traffic
  EQUAL-COV         per-language thresholds at the same review share C  (equalized conformal)
  CONRA             parity-constrained equal coverage (our earlier rule)
  ABS-RISK(a)       per-language share = min{s : r_l(s) <= a}          <-- group-conditional CRC
  ABS-RISK(a, d)    same with a finite-sample slack: a_eff = a - sqrt(log(1/d)/(2 n_l))
  BONF-CRC(a)       group-conditional CRC with a split across the two groups (a/2 each)
  MINMAX-CAP(C)     capacity-aware worst-group risk minimisation: equalise r_l(s_l) under the budget
  CAPRA(a)          our one-parameter family (capra.sweep), read at the same load

Comparison protocol (identical for every family):
  1. fit on dev -> an allocation (s_first, s_second);
  2. keep only allocations whose *load* is within the budget band around C (bar the risk-target
     families, which cannot control the load — they are interpolated to load = C and reported with
     their true achieved load when that is impossible);
  3. apply the thresholds to TEST and report (D = |r_a - r_b|, K = share of the queue from `b`);
  4. mark a baseline DOMINATED when a CAPRA point at the same load has D' <= D and |K' - K*| <= |K - K*|
     with at least one strict inequality.

Usage:
  python baseline_rules.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
      --budgets 0.05 --out notes/baseline_killtest.md --json outputs/killtest/baseline.json
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capra as CA          # noqa: E402


def _bad(x):
    try:
        return x is None or x != x
    except Exception:                                              # noqa: BLE001
        return x is None


def num(x, nd=3):
    return "—" if _bad(x) else f"{x:.{nd}f}"


def pp(x, nd=2):
    return "—" if _bad(x) else f"{100.0 * x:+.{nd}f}"
import conra_parity as CP   # noqa: E402

ALPHA_GRID = np.round(np.linspace(0.002, 0.5, 250), 5)
RISK_GRID = np.round(np.linspace(0.001, 0.5, 250), 5)
S_GRID = np.linspace(0.0, 1.0, 501)


# ---------------------------------------------------------------- curve helpers
def share_for_risk(curve, alpha):
    """Smallest review share whose kept-set risk is <= alpha (None when unreachable)."""
    xs, ys = curve
    if ys[0] <= alpha:
        return 0.0
    if ys[-1] > alpha:
        return None
    s = float(np.interp(-alpha, -ys, xs))     # ys decreasing in xs
    return max(0.0, min(1.0, s))


def share_for_equal_risk(curve_a, curve_b):
    """Shares that equalise the two risks as far as the budget-free allocation can (min-max)."""
    best = None
    for s in S_GRID:
        d = abs(CA.risk_at_share(curve_a, s) - CA.risk_at_share(curve_b, s))
        if best is None or d < best[0]:
            best = (d, s)
    return best[1], best[1]


def describe(curves, shares, langs, costs, weights=(0.5, 0.5)):
    """Design-time (dev) load / disparity / composition of an allocation."""
    a, b = langs
    wa, wb = weights
    s = {a: float(np.clip(shares[a], 0, 1)), b: float(np.clip(shares[b], 0, 1))}
    load = wa * costs[a] * s[a] + wb * costs[b] * s[b]
    ra, rb = CA.risk_at_share(curves[a], s[a]), CA.risk_at_share(curves[b], s[b])
    qb = wb * costs[b] * s[b]
    k = qb / load if load > 1e-12 else float("nan")
    return {"s": s, "load": float(load), "risk": {a: float(ra), b: float(rb)},
            "D_abs": abs(ra - rb), "K": float(k)}


def traj_design_risk_targets(curves, langs, costs, weights, slack=0.0, n_dev=None,
                             split_alpha=False, delta=0.05):
    """Family ABS-RISK / BONF-CRC: per-language risk target -> shares (load is NOT controlled)."""
    pts = []
    for alpha in RISK_GRID:
        tgt = alpha / 2.0 if split_alpha else alpha
        if slack and n_dev:
            tgt = max(1e-6, tgt - float(np.sqrt(np.log(1.0 / delta) / (2.0 * min(n_dev.values())))))
        shares = {}
        ok = True
        for l in langs:
            s = share_for_risk(curves[l], tgt)
            if s is None:
                ok = False
                break
            shares[l] = s
        if not ok:
            continue
        pts.append({"alpha": float(alpha), **describe(curves, shares, langs, costs, weights)})
    return pts


def traj_design_fixed_share(curves, langs, costs, weights):
    """EQUAL-COV (equalized conformal): one common review share per language, swept."""
    pts = []
    for C in ALPHA_GRID:
        shares = {l: float(C) for l in langs}
        pts.append({"alpha": float(C), **describe(curves, shares, langs, costs, weights)})
    return pts


def traj_design_global(d, curves, langs, costs, weights):
    """GLOBAL: ONE pooled threshold for both languages; per-language review shares emerge.

    A pooled review share C of the mixed traffic selects the threshold tau at the C-quantile of the
    confidence distribution of the mix (weights w_l); each language's own share is then
    s_l = P_l(conf < tau). This is the "just use one threshold" baseline every practitioner starts from.
    """
    a, b = langs
    wa, wb = weights
    ca, cb = d["cal"][a]["conf"], d["cal"][b]["conf"]
    pts = []
    for C in ALPHA_GRID:
        # weighted quantile of the pooled confidence distribution (review = low confidence)
        concat = np.concatenate([ca, cb])
        w = np.concatenate([np.full(len(ca), wa / len(ca)), np.full(len(cb), wb / len(cb))])
        order = np.argsort(concat, kind="mergesort")
        cw = np.cumsum(w[order]) / w.sum()
        tau = float(concat[order][np.searchsorted(cw, C, side="left").clip(0, len(cw) - 1)])
        shares = {a: float((ca < tau).mean()), b: float((cb < tau).mean())}
        pts.append({"alpha": float(C), "tau": tau, **describe(curves, shares, langs, costs, weights)})
    return pts



def traj_minmax_cap(curves, langs, costs, budget, weights=(0.5, 0.5)):
    """Worst-group allocation under the budget: spend all of C minimising max(risk_a, risk_b).

    Equal per-language shares is EQUAL-COV, a different rule; the documented mechanism - equalising the
    two risks - lived in a helper that was never called (review finding R3-15).
    """
    a, b = langs
    wa, wb = weights
    pts = []
    for sa in np.linspace(0.0, 1.0, 501):
        sb = (budget - wa * costs[a] * sa) / (wb * costs[b])
        if sb < 0.0 or sb > 1.0:
            continue
        d = describe(curves, {a: float(sa), b: float(sb)}, langs, costs, weights)
        d["alpha"] = float(sa)
        d["max_risk"] = max(d["risk"][a], d["risk"][b])
        pts.append(d)
    return pts


def interp_at_load(traj, target):
    """Linear interpolation of a family trajectory at a given load (D, K, shares). Returns None if out of range."""
    pts = sorted(traj, key=lambda p: p["load"])
    if not pts or target < pts[0]["load"] - 1e-9 or target > pts[-1]["load"] + 1e-9:
        return None
    for p0, p1 in zip(pts, pts[1:]):
        if p0["load"] - 1e-9 <= target <= p1["load"] + 1e-9:
            if abs(p1["load"] - p0["load"]) < 1e-12:
                w = 0.0
            else:
                w = (target - p0["load"]) / (p1["load"] - p0["load"])
            langs = list(p0["s"])
            return {"alpha": float(p0["alpha"] + w * (p1["alpha"] - p0["alpha"])),
                    "load": float(target),
                    "s": {l: float(p0["s"][l] + w * (p1["s"][l] - p0["s"][l])) for l in langs},
                    "D_abs": float(p0["D_abs"] + w * (p1["D_abs"] - p0["D_abs"])),
                    "K": float(p0["K"] + w * (p1["K"] - p0["K"]))}
    return None


# ---------------------------------------------------------------- per-detector analysis
def load_pair(out_root, a, b, mode, match_prev=None):
    seed_dirs = sorted(d for d in glob.glob(os.path.join(out_root, "seed*"))
                       if os.path.exists(os.path.join(d, "probs.npz")))
    if not seed_dirs and os.path.exists(os.path.join(out_root, "probs.npz")):
        seed_dirs = [out_root]
    assert seed_dirs, f"no probs.npz under {out_root}"
    CP.LANGS = [a, b]
    if (a, b) == ("en", "vi"):
        return CP.prep(seed_dirs, mode), len(seed_dirs)
    import pair_parity as PP
    return PP.build_pair(seed_dirs, a, b, match_prev), len(seed_dirs)


def analyse_pair(out_root, a, b, mode, budget, k_star, cost_second, match_prev=None):
    d, n_seeds = load_pair(out_root, a, b, mode, match_prev)
    curves = {l: CA.review_risk_curve(d["cal"][l]["conf"], d["cal"][l]["err"]) for l in (a, b)}
    n_dev = {l: int(len(d["cal"][l]["conf"])) for l in (a, b)}
    langs, K = (a, b), k_star
    res = {"detector": out_root, "pair": [a, b], "mode": mode, "seeds": n_seeds, "budget": budget,
           "n_dev": n_dev, "families": {}}

    def test_of(shares, costs=None):
        tau = {l: CA.tau_from_share(d["cal"][l]["conf"], shares[l]) for l in langs}
        t = CA.measure_on_test(tau, d["test"], langs, costs=costs)
        return {"D_test": float(t["D_abs"]), "K_test": float(t["K"]), "load_test": float(t["load"]),
                # the fallback is a HARD failure for the cost-aware variant: the item share and the
                # time-equivalent load are different quantities and printing one as the other is how the
                # round-3 review caught six wrong cells (RV3C-02)
                "load_budget_test": float(t["load_budget"] if "load_budget" in t
                                          else (_ for _ in ()).throw(AssertionError(
                                              "load_budget missing: call test_of(..., costs=...)")))}

    def random_rule(costs, draws=200, seed=20260923):
        """Null policy: review a uniformly RANDOM share of each language - SIMULATED.

        The kept set keeps the base error rate of its language, so this is the disparity a team gets
        with no targeting at all; without it the frontier has no null to beat (review finding R2-26).
        It is a randomisation, so it is drawn: 200 seeded draws give a mean and an interval, and the
        queue share is measured from the draw sizes instead of assumed (review finding R3-14).

        The share is solved ON THE BUDGET'S OWN UNIT (round-2 review, item 1). In the cost-blind variant
        the budget is a share of items, so the null reviews share C of each language's items. In the
        cost-aware variant the budget is review TIME, and re-using the item share there spends
        (w_a c_a + w_b c_b) C > C of it - the one row of the comparison that left the constraint it was
        compared under. Solving sum_l w_l c_l s_l = C keeps the null inside the budget in both units.
        """
        rng = np.random.default_rng(seed)
        ea_, eb_ = d["test"][a]["err"], d["test"][b]["err"]
        na, nb = len(ea_), len(eb_)
        wa, wb = 0.5, 0.5
        s = budget / (wa * costs[a] + wb * costs[b])            # equal item share per language
        n_keep = {a: max(1, int(round(s * na))), b: max(1, int(round(s * nb)))}
        Ds = np.empty(draws)
        for i in range(draws):
            ia = rng.choice(na, size=n_keep[a], replace=False)
            ib = rng.choice(nb, size=n_keep[b], replace=False)
            Ds[i] = abs(ea_[ia].mean() - eb_[ib].mean())
        r_cal = {l: float(d["cal"][l]["err"].mean()) for l in langs}
        return {"design": {"D_abs": abs(r_cal[a] - r_cal[b]), "K": 0.5,
                           "load": wa * costs[a] * s + wb * costs[b] * s,
                           "s": {l: s for l in langs}},
                "test": {"D_test": float(Ds.mean()), "K_test": float(n_keep[b] / (n_keep[a] + n_keep[b])),
                         "load_test": float((n_keep[a] + n_keep[b]) / (na + nb)),
                         "load_budget_test": wa * costs[a] * (n_keep[a] / na)
                         + wb * costs[b] * (n_keep[b] / nb),
                         "D_test_ci": [float(np.percentile(Ds, 2.5)), float(np.percentile(Ds, 97.5))],
                         "draws": int(draws)}}

    def oracle_rule(costs, target_load, load_key="load"):
        """Upper bound: the best disparity achievable ON TEST at the SAME REALISED LOAD (an oracle).

        Two things had to be fixed for this row to mean anything. First it must be measured with the
        thresholds the test side implies - the previous version computed them and then called test_of(),
        which refits on calibration, so ORACLE was identical to CAPRA by construction (review finding
        R3-08). Second it must be load-matched on the REALISED item load, not on the nominal budget: the
        test-side sweep spends C in design terms but realises a slightly different share of items, and a
        "bound" measured at a lower load is not a bound (review findings R2-05, N3-10). We therefore scan
        the test-side frontier over a range of nominal budgets and keep the best D among the allocations
        whose realised item load matches the calibration-selected endpoint's to within 0.2 pp.

        `load_key` selects WHICH load is matched, and it is the one the budget is denominated in
        (round-2 review, item 1): the item share in the cost-blind variant and the time-equivalent share
        in the cost-aware one. Matching a time-budgeted row on its item share put the bound above the
        constraint the rest of the comparison satisfies.
        """
        tc = {l: CA.review_risk_curve(d["test"][l]["conf"], d["test"][l]["err"]) for l in langs}
        best = None
        for Cx in np.linspace(0.4 * budget, 3.0 * budget, 80):
            front = CA.sweep(tc, float(Cx), langs, costs)
            feas = [p for p in front if not p["clipped"]] or front
            p = min(feas, key=lambda x: x["D_abs"])
            tau = {l: CA.tau_from_share(d["test"][l]["conf"], p["s"][l]) for l in langs}
            m = CA.measure_on_test(tau, d["test"], langs, costs=costs)
            if abs(float(m[load_key]) - target_load) > 0.002:
                continue
            if best is None or float(m["D_abs"]) < best[0]:
                best = (float(m["D_abs"]), p, m)
        if best is None:                                   # cannot realise that load on the test side
            return {"design": {"D_abs": float("nan"), "K": float("nan"), "load": float("nan"), "s": {}},
                    "test": {"D_test": float("nan"), "K_test": float("nan"), "load_test": float("nan"),
                             "load_budget_test": float("nan"),
                             "note": "no test-side allocation realises load %.4f" % target_load}}
        _d, p, m = best
        t = {"D_test": float(m["D_abs"]), "K_test": float(m["K"]), "load_test": float(m["load"]),
             "load_budget_test": float(m.get("load_budget", m["load"])), "alpha": float(p["alpha"]),
             "load_matched_to": float(target_load)}
        return {"design": {"D_abs": float("nan"), "K": float("nan"), "load": float("nan"), "s": p["s"]},
                "test": t}

    for tag, costs in (("cost_blind", {a: 1.0, b: 1.0}),
                       ("cost_aware", {a: 1.0, b: cost_second})):
        fam = {}
        # --- our family, at exactly the budget
        front = CA.sweep(curves, budget, langs, costs)
        feas = [p for p in front if not p["clipped"]] or front
        capra_pts = [{"alpha": p["alpha"], "load": p["load"], "D_abs": p["D_abs"], "K": p["K"],
                      "s": p["s"]} for p in feas]
        pf = min(feas, key=lambda p: p["D_abs"])
        fam["CAPRA"] = {"kind": "ours", "design": {"D_abs": pf["D_abs"], "K": pf["K"],
                                                   "load": pf["load"], "s": pf["s"]},
                        "front": capra_pts, "test": test_of(pf["s"], costs)}

        fam["RANDOM"] = {"kind": "null", **random_rule(costs)}
        # the oracle is matched to the load CAPRA's endpoint actually realises on test, IN THE UNIT THE
        # BUDGET IS DENOMINATED IN (round-2 review, item 1): the item share when the two languages cost
        # the same, the review-time share when they do not.
        matched_on = "load_test" if costs[a] == costs[b] else "load_budget_test"
        load_key = "load" if matched_on == "load_test" else "load_budget"
        fam["ORACLE(test-optimal)"] = {"kind": "oracle",
                                       **oracle_rule(costs, fam["CAPRA"]["test"][matched_on], load_key)}

        # --- prior-art families
        for name, traj in (
            ("GLOBAL", traj_design_global(d, curves, langs, costs, (0.5, 0.5))),
            ("EQUAL-COV", traj_design_fixed_share(curves, langs, costs, (0.5, 0.5))),
            ("ABS-RISK", traj_design_risk_targets(curves, langs, costs, (0.5, 0.5))),
            ("ABS-RISK-finite", traj_design_risk_targets(curves, langs, costs, (0.5, 0.5),
                                                         slack=1.0, n_dev=n_dev)),
            ("BONF-CRC", traj_design_risk_targets(curves, langs, costs, (0.5, 0.5), split_alpha=True)),
        ):
            pt = interp_at_load(traj, budget)
            if pt is None:
                # cannot reach the budget: report the biggest load it can use, honestly
                far = max(traj, key=lambda p: p["load"]) if traj else None
                fam[name] = {"kind": "prior", "design": None, "test": None,
                             "note": "cannot reach load=%.3f (max %.3f)" % (
                                 budget, far["load"] if far else float("nan")),
                             "max_load": far["load"] if far else None}
                continue
            fam[name] = {"kind": "prior", "design": pt, "test": test_of(pt["s"], costs)}

        mm = traj_minmax_cap(curves, langs, costs, budget)
        if mm:
            pt = min(mm, key=lambda p: p["max_risk"])
            fam["MINMAX-CAP"] = {"kind": "prior", "design": pt, "test": test_of(pt["s"], costs)}

        # --- dominance: assessed on CALIBRATION (that is where a deployment chooses), but a sub-0.1 pp
        #     design gap is calibration noise, not a verdict, and equal test values must get equal
        #     verdicts; every row also carries the realised TEST load (review findings R2-04, R2-05)
        capra_test = fam["CAPRA"]["test"]
        for name, e in fam.items():
            if e.get("design") is None or e.get("test") is None:
                continue
            Dn, Kn = e["test"]["D_test"], abs(e["test"]["K_test"] - K)
            e["D_test"], e["absK_test"] = Dn, Kn
            e["load_test"] = e["test"].get("load_test")
            e["indistinguishable_test"] = bool(
                abs(e["test"]["D_test"] - capra_test["D_test"]) < 0.001
                and abs(e["test"]["K_test"] - capra_test["K_test"]) < 0.01)
            tol = 1e-9
            cands = [p for p in capra_pts
                     if p["D_abs"] <= e["design"]["D_abs"] + tol
                     and abs(p["K"] - K) <= abs(e["design"]["K"] - K) + tol]
            strict = any((p["D_abs"] < e["design"]["D_abs"] - 0.001)      # > 0.1 pp to count
                         or (abs(p["K"] - K) < abs(e["design"]["K"] - K) - 0.001) for p in cands)
            # a design-side verdict alone is not enough: it is decided on the very split the endpoints
            # were SELECTED from, so the test side must also separate the two rules by more than the
            # demonstrated resolution before we call it dominance (review findings R3-16, R2-11)
            e["dominated_by_capra_design"] = bool(cands) and strict and not e["indistinguishable_test"] \
                and (e["test"]["D_test"] - capra_test["D_test"]) > 0.008
        res["families"][tag] = fam

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lang-a", default="en")
    ap.add_argument("--lang-b", required=True, help="comma list of target languages")
    ap.add_argument("--mode", default="NAT")
    ap.add_argument("--budgets", default="0.05")
    ap.add_argument("--composition-target", type=float, default=0.5)
    ap.add_argument("--cost-second", type=float, default=1.5)
    ap.add_argument("--match-prevalence", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    a = args.lang_a
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "detector": args.out_root,
           "lang_a": a, "mode": args.mode, "budgets": [float(x) for x in args.budgets.split(",")],
           "composition_target": args.composition_target, "cost_second": args.cost_second,
           "pairs": {}}
    lines = [f"# Prior-art baselines vs CAPRA — `{args.out_root}` ({a} vs targets, mode {args.mode})\n",
             f"Generated {out['generated']}. Every family is fitted on the calibration split and "
             f"evaluated on test at **load = C** (interpolated when the family cannot control its own "
             f"load). `D` = |risk gap| in pp; `K` = queue share of the second language; "
             f"`K*` = {args.composition_target:.2f}. `load_test` is the ITEM share of traffic reviewed: "
             f"for the cost-blind variant it must match C to within 0.5 pp, while in the cost-aware "
             f"variant the budget is denominated in review TIME, so the item share is lower than C by "
             f"construction (the budget-unit load is printed alongside). Dominance is "
             f"assessed on CALIBRATION (where a deployment chooses) and requires a gap above 0.1 pp; "
             f"`indistinguishable` means the rule's test disparity and queue share match CAPRA's to "
             f"within 0.1 pp and 0.01. RANDOM is the null policy (a uniformly random review of each "
             f"language, drawn on the budget's own unit: share C of the items when costs are equal, and "
             f"the item share that spends C of review time when they are not) and ORACLE is the "
             f"allocation that is optimal on test at the same load, matched in that same unit — an "
             f"upper bound, not a deployable rule.\n"]

    for b in [s.strip() for s in args.lang_b.split(",") if s.strip()]:
        for C in out["budgets"]:
            res = analyse_pair(args.out_root, a, b, args.mode, C, args.composition_target,
                               args.cost_second, args.match_prevalence)
            out["pairs"].setdefault(b, {})[f"{C:.2f}"] = res
            for tag in ("cost_blind", "cost_aware"):
                lines.append(f"\n## {b} · C = {C:.2f} · {tag}\n")
                lines.append("| family | design D (pp) | design K | load (design) | test D (pp) | "
                             "test K | test load | |K−K*| | vs CAPRA on test | note |")
                lines.append("|---|---|---|---|---|---|---|---|---|---|")
                for name, e in res["families"][tag].items():
                    if e.get("design") is None or e.get("test") is None:
                        lines.append(f"| {name} | — | — | — | — | — | — | — | n/a | {e.get('note', '')} |")
                        continue
                    dsn = e["design"]
                    if e.get("indistinguishable_test"):
                        verdict = "indistinguishable"
                    elif e.get("dominated_by_capra_design"):
                        verdict = "dominated (design)"
                    else:
                        verdict = "different"
                    lines.append(
                        f"| {name} | {pp(dsn['D_abs'])} | {num(dsn['K'])} | {num(dsn['load'])} | "
                        f"**{pp(e['test']['D_test'])}** | {num(e['test']['K_test'])} | "
                        f"{num(e.get('load_test'))} | {num(e['absK_test'])} | {verdict} | "
                        f"{'ours' if e['kind'] == 'ours' else e['kind']} |")
            print(f"[{args.out_root}] {a}/{b} C={C:.2f} done", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w", encoding="utf-8"), indent=2)
        print("json ->", args.json, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print("report ->", args.out, flush=True)


if __name__ == "__main__":
    main()
