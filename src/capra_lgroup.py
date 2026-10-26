"""CAPRA for L > 2 groups — the L-group allocation, and its demonstration on the 7-language benchmark.

Why this file exists
--------------------
The paper's formalisation is written for two populations, where a policy is a pair of thresholds and the
binding budget leaves ONE degree of freedom, so the achievable set is a curve that a sweep can trace. With
L populations there are L review shares, the binding budget leaves L-1 degrees of freedom, and a sweep can
no longer trace the set: the allocation has to be solved. This script does three things.

1. SOLVE. For L groups it minimises the worst pairwise residual-risk gap over the (L-1)-simplex of budget
   shares, subject to the same work-conserving capacity as the two-group case. Budget share p_l spent on
   group l implies review share s_l = p_l C / (w_l c_l), so the feasible polytope is
   {p >= 0, sum p = 1, p_l <= w_l c_l / C} — the L-group analogue of the interval of Appendix A.
2. TRACE the trade-off between worst-gap parity and queue composition on the same polytope, with the
   scalarised objective max_gap + lambda * max_l |K_l - w_l|, which is the (L-1)-dimensional surface the
   two-group paper reports as a curve with two endpoints.
3. VALIDATE. The same code with L = 2 on the flagship detector must reproduce the two-group number the
   manuscript already prints (parity-first test disparity 0.94 pp on English-Vietnamese at C = 5 %). An
   aggregator that cannot reproduce a known number is not evidence for anything else.

Conventions (identical to capra.py / pair_parity.py, so the numbers are comparable with Table 7):
  * thresholds and risk curves fitted on the calibration (dev) split, everything reported on test;
  * per-seed arrays POOLED over the training seeds (n per language is the pooled n of the paper);
  * equal traffic weights w_l = 1/L, cost-blind variant (c_l = 1);
  * risk of a group = error rate of its KEPT set, at the review share the allocation induces.

Usage (CPU only, reads the frozen probs.npz artefacts):
  python capra_lgroup.py --out-root outputs/textdetox/xlmr_base \
      --langs en,de,uk,ar,hi,zh,ru --budget 0.05 \
      --json outputs/textdetox/capra_lgroup_xlmr_base.json --out notes/lgroup_xlmr_base.md
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
import conra_parity as CP   # noqa: E402
import pair_parity as PP    # noqa: E402

C_REF = 0.05


# ---------------------------------------------------------------- data
def seed_dirs_of(out_root):
    dirs = sorted(d for d in glob.glob(os.path.join(out_root, "seed*"))
                  if os.path.exists(os.path.join(d, "probs.npz")))
    if not dirs and os.path.exists(os.path.join(out_root, "probs.npz")):
        dirs = [out_root]
    return dirs


def build(seed_dirs, langs):
    """Calibration from dev, evaluation from test, per language, pooled over seeds."""
    return {"cal": {l: PP.load_lang(seed_dirs, l, "dev") for l in langs},
            "test": {l: PP.load_lang(seed_dirs, l, "test") for l in langs}}


def curves_of(d, langs):
    return {l: CA.review_risk_curve(d["cal"][l]["conf"], d["cal"][l]["err"]) for l in langs}


# ---------------------------------------------------------------- allocation
def shares(p, C, w, c):
    return np.asarray(p, dtype=float) * C / (w * c)


def project(p, pmax):
    """Nearest point of {0 <= p <= pmax, sum p = 1}: clip, then move the residual mass to free entries."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, pmax)
    for _ in range(64):
        gap = 1.0 - p.sum()
        if abs(gap) < 1e-12:
            break
        if gap > 0:                                     # add to the entries with room
            room = pmax - p
            free = room > 1e-15
            if not free.any():
                break
            p[free] += gap * room[free] / room[free].sum()
        else:                                           # take from the entries that have mass
            take = p > 1e-15
            if not take.any():
                break
            p[take] += gap * p[take] / p[take].sum()
    return np.clip(p, 0.0, pmax) * (1.0 / max(np.clip(p, 0.0, pmax).sum(), 1e-12))


def evaluate(p, C, w, c, curves, langs):
    s = np.clip(shares(p, C, w, c), 0.0, 1.0)
    r = np.array([CA.risk_at_share(curves[l], s[i]) for i, l in enumerate(langs)])
    items = w * s
    K = items / max(items.sum(), 1e-12)
    return {"p": np.asarray(p, float), "s": s, "risk": r,
            "max_gap": float(r.max() - r.min()), "mean_risk": float((w * r).sum()),
            "comp_dev": float(np.abs(K - w).max()), "comp_dev_l1": float(np.abs(K - w).sum()),
            "K": K, "load": float((w * c * s).sum())}


def risk_vec(curve, s):
    """Vectorised r_l(s): np.interp already clamps outside the observed range, like risk_at_share."""
    xs, ys = curve
    return np.interp(np.asarray(s, dtype=float), xs, ys)


def evaluate_many(P, C, w, c, curves, langs):
    """(max_gap, comp_dev) for a whole matrix of allocations at once — the inner loop of the search."""
    S = np.clip(np.asarray(P, dtype=float) * C / (w * c), 0.0, 1.0)
    R = np.stack([risk_vec(curves[l], S[:, i]) for i, l in enumerate(langs)], axis=1)
    items = S * w
    K = items / np.maximum(items.sum(axis=1, keepdims=True), 1e-12)
    return R, K


def optimise(C, w, c, curves, langs, lam=0.0, delta=None, n_random=40000, seed=7, topk=8):
    """argmin over the feasible simplex of max_gap + lam * max_l |K_l - w_l|.

    `delta` switches the composition term into a CEILING rather than a price: the objective becomes
    max_gap + 100 * max(0, comp_dev - delta), which traces the frontier "how small can the worst gap be
    if the queue may deviate from the traffic shares by at most delta" — the L-group analogue of the
    (K, D) achievable set, which for two groups is a curve a sweep can draw.
    """
    n = len(langs)
    pmax = np.minimum(1.0, w * c / C)

    rng = np.random.default_rng(seed)
    cand = rng.dirichlet(np.ones(n), size=n_random)
    # boundary optima are the rule, not the exception (the two-group parity-first point spends the whole
    # budget on one group), so seed the search with the vertices and the pooled point as well
    cand = np.vstack([cand, np.eye(n), np.full((1, n), 1.0 / n)])
    cand = cand[(cand <= pmax + 1e-12).all(axis=1)]
    if not len(cand):                                   # polytope degenerate at this budget
        return None, float("inf")
    R, K = evaluate_many(cand, C, w, c, curves, langs)
    dev = np.abs(K - w).max(axis=1)
    penalty = 100.0 * np.maximum(0.0, dev - delta) if delta is not None else lam * dev
    vals = (R.max(axis=1) - R.min(axis=1)) + penalty
    order = np.argsort(vals)[:topk]

    def obj(p):
        R1, K1 = evaluate_many(np.asarray(p, float)[None, :], C, w, c, curves, langs)
        d1 = float(np.abs(K1 - w).max())
        pen = 100.0 * max(0.0, d1 - delta) if delta is not None else lam * d1
        return float(R1.max() - R1.min()) + pen

    best_p, best_v = None, float("inf")
    steps = (0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
    for idx in order:
        p = project(cand[idx], pmax)
        v = obj(p)
        for h in steps:                                 # pattern search on the simplex
            for _ in range(40):                         # bounded: a shrinking step cannot improve forever
                improved = False
                for i in range(n):
                    for d in (+1.0, -1.0):
                        q = project(p + d * h * np.eye(n)[i], pmax)
                        vq = obj(q)
                        if vq < v - 1e-13:
                            p, v, improved = q, vq, True
                if not improved:
                    break
        if v < best_v:
            best_p, best_v = p, v
    return best_p, best_v


def tau_of(p, C, w, c, d, langs):
    s = np.clip(shares(p, C, w, c), 0.0, 1.0)
    return {l: CA.tau_from_share(d["cal"][l]["conf"], s[i]) for i, l in enumerate(langs)}


def on_test(tau, d, langs, w):
    """`w` is a {lang: traffic weight} dict here, because the per-group items are keyed by language."""
    r, cov, items = {}, {}, {}
    for l in langs:
        cov[l] = float(CP.cov_of_tau(d["test"][l]["conf"], tau[l]))
        r[l] = float(CP.risk_of_tau(d["test"][l]["conf"], d["test"][l]["err"], tau[l]))
        items[l] = w[l] * (1.0 - cov[l])
    rv = np.array([r[l] for l in langs])
    wa = np.array([w[l] for l in langs])
    tot = max(sum(items.values()), 1e-12)
    K = np.array([items[l] / tot for l in langs])
    return {"risk": r, "coverage": cov, "K": K.tolist(),
            "max_gap": float(rv.max() - rv.min()), "mean_risk": float(sum(w[l] * r[l] for l in langs)),
            "comp_dev": float(np.abs(K - wa).max()), "comp_dev_l1": float(np.abs(K - wa).sum()),
            "load": float(sum(w[l] * (1.0 - cov[l]) for l in langs))}


# ---------------------------------------------------------------- reporting
def rule_block(p, C, w, c, w_d, d, curves, langs, tag, seed_dirs, rule="fixed"):
    """`rule` says what to recompute per training seed: "fixed" re-uses the same allocation, "parity"
    re-solves it (the rule is the solver, so a seed-level spread is only meaningful if it re-solves),
    "oracle" re-solves on that seed's test curves. Re-using the parity-first allocation for the pooled
    and composition-first rows would report one rule's seed spread under three names.
    """
    design = evaluate(p, C, w, c, curves, langs)
    tau = tau_of(p, C, w, c, d, langs)
    test = on_test(tau, d, langs, w_d)
    per_seed = []
    for sd in seed_dirs:
        ds = {"cal": {l: PP.load_lang([sd], l, "dev") for l in langs},
              "test": {l: PP.load_lang([sd], l, "test") for l in langs}}
        cv = curves_of(ds, langs)
        if rule == "fixed":
            ps = p
        elif rule == "oracle":
            tv = {l: CA.review_risk_curve(ds["test"][l]["conf"], ds["test"][l]["err"]) for l in langs}
            ps, _ = optimise(C, w, c, tv, langs, lam=0.0, n_random=8000, seed=11)
        else:
            ps, _ = optimise(C, w, c, cv, langs, lam=0.0, n_random=8000, seed=11)
        if ps is None:
            continue
        t = tau_of(ps, C, w, c, ds, langs)
        per_seed.append(on_test(t, ds, langs, w_d)["max_gap"])
    return {"tag": tag, "p": np.asarray(p, float).tolist(),
            "design": {k: (design[k].tolist() if isinstance(design[k], np.ndarray) else design[k])
                       for k in ("s", "risk", "max_gap", "mean_risk", "comp_dev", "comp_dev_l1",
                                 "K", "load")},
            "test": test, "seed_max_gap_mean": float(np.mean(per_seed)) if per_seed else None,
            "seed_max_gap_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None,
            "seed_max_gap": per_seed, "n_seeds": len(per_seed)}


def analyse(seed_dirs, langs, C, trace=True):
    w = np.full(len(langs), 1.0 / len(langs))
    c = np.ones(len(langs))
    w_d = {l: 1.0 / len(langs) for l in langs}      # the same weights, keyed for per-group lookups
    d = build(seed_dirs, langs)
    curves = curves_of(d, langs)
    pmax = np.minimum(1.0, w * c / C)

    res = {"langs": langs, "budget": C, "n_per_lang": {l: int(len(d["test"][l]["conf"])) for l in langs},
           "seeds": [os.path.basename(s) for s in seed_dirs], "n_seeds": len(seed_dirs),
           "pmax": pmax.tolist(), "polytope_ok": bool(pmax.sum() >= 1.0 - 1e-12)}

    # pooled (single threshold): equal review share = the two-group GLOBAL rule
    p_pooled = project(np.full(len(langs), 1.0 / len(langs)), pmax)
    res["pooled"] = rule_block(p_pooled, C, w, c, w_d, d, curves, langs, "pooled", seed_dirs, rule="fixed")

    # parity-first: minimise the worst pairwise gap
    p_pf, v_pf = optimise(C, w, c, curves, langs, lam=0.0)
    res["parity_first"] = rule_block(p_pf, C, w, c, w_d, d, curves, langs, "parity-first", seed_dirs, rule="parity")

    # composition-first: K_l = w_l exactly. With equal weights and equal costs that forces s_l = C,
    # i.e. the pooled allocation itself -- the L-group statement of "the queue can only mirror traffic
    # if every group is reviewed at the same share".
    p_cf = project(np.full(len(langs), 1.0 / len(langs)), pmax)
    res["composition_first"] = rule_block(p_cf, C, w, c, w_d, d, curves, langs, "composition-first", seed_dirs, rule="fixed")

    # oracle: test-optimal at the same budget (optimistic bound, not a deployable rule). The ALLOCATION is
    # solved on the test curves, but its design-side numbers are READ on the design curves like every other
    # row's: handing `d_test_curves` into rule_block made the "Design (pp)" cell of Table C.1 the oracle's
    # value on the TEST curves, so the table reported a design disparity below the design optimum it was
    # being compared with (round-2 external review, item 2; probe probe_round2_lgroup_oracle.py). The
    # per-seed re-solve below still uses each seed's own test curves, as the rule requires.
    d_test_curves = {l: CA.review_risk_curve(d["test"][l]["conf"], d["test"][l]["err"]) for l in langs}
    p_or, v_or = optimise(C, w, c, d_test_curves, langs, lam=0.0)
    res["oracle"] = rule_block(p_or, C, w, c, w_d, d, curves, langs, "oracle (test-optimal)", seed_dirs, rule="oracle")

    pf, cf = res["parity_first"], res["composition_first"]
    res["prices"] = {
        "price_of_composition_test": cf["test"]["max_gap"] - pf["test"]["max_gap"],
        "price_of_parity_test": pf["test"]["comp_dev_l1"],
        "both_feasible_test": bool(cf["test"]["max_gap"] < 0.005 and cf["test"]["comp_dev"] < 0.005),
        "worst_gap_pooled_test": res["pooled"]["test"]["max_gap"],
        "gap_reduction_pooled_to_parity_first": (res["pooled"]["test"]["max_gap"]
                                                 - pf["test"]["max_gap"]),
    }
    res["feasible_policy_count_dim"] = len(langs) - 1     # dimension of the budget-binding set

    if trace:
        # The two-group achievable set is a curve in (K, D) that a sweep can draw. With L groups the set is
        # (L-1)-dimensional, so we report its projection on the two quantities the paper reasons about: the
        # frontier "smallest worst gap attainable if the queue may deviate from the traffic shares by at
        # most delta", traced by putting a hard ceiling on the composition deviation.
        tr = []
        devs = sorted({0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0})
        for delta in devs:
            p, v = optimise(C, w, c, curves, langs, delta=delta, n_random=20000, seed=13)
            if p is None:
                continue
            m = evaluate(p, C, w, c, curves, langs)
            t = on_test(tau_of(p, C, w, c, d, langs), d, langs, w_d)
            tr.append({"delta": delta, "design_max_gap": m["max_gap"], "design_comp_dev": m["comp_dev"],
                       "test_max_gap": t["max_gap"], "test_comp_dev": t["comp_dev"],
                       "test_comp_dev_l1": t["comp_dev_l1"], "p": np.asarray(p, float).tolist()})
        res["tradeoff_frontier"] = tr

        # the L-group analogue of the structural parity capacity: the smallest budget at which the worst
        # pairwise gap of the parity-first allocation falls to the resolution the paper reports in. The
        # budget must cover the traffic of every group, so this is NOT the max of the pairwise capacities
        # of Table 7 and is expected to sit far above a deployment budget.
        cap = []
        for Cx in (0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            px, _ = optimise(Cx, w, c, curves, langs, lam=0.0, n_random=20000, seed=17)
            if px is None:
                cap.append({"budget": Cx, "design_max_gap": None})
                continue
            mx = evaluate(px, Cx, w, c, curves, langs)
            cap.append({"budget": Cx, "design_max_gap": mx["max_gap"], "p": np.asarray(px, float).tolist()})
        res["capacity_sweep"] = cap
    return res


def markdown(res, title):
    langs = res["langs"]
    n_txt = ", ".join(f"{l} {res['n_per_lang'][l]}" for l in langs)
    L = [f"# {title}\n",
         f"Generated from the frozen artefacts · L = {len(langs)} groups · budget C = {res['budget']:.0%} · "
         f"equal traffic weights w = 1/{len(langs)} · cost-blind · thresholds on dev, everything measured on "
         f"test · n per group = {n_txt} · seeds = {res['n_seeds']} pooled\n",
         "`max gap` = worst pairwise residual-risk gap (pp) between groups; `K dev` = max deviation of the "
         "queue composition from the traffic share; `load` = realised review share of traffic.\n",
         "| rule | design max gap (pp) | **test max gap (pp)** | K dev | load | seed mean ± SD (pp) |",
         "|---|---|---|---|---|---|"]
    for tag in ("pooled", "parity_first", "composition_first", "oracle"):
        b = res[tag]
        sd = "—" if b["seed_max_gap_sd"] is None else f"{b['seed_max_gap_mean']*100:.2f} ± {b['seed_max_gap_sd']*100:.2f}"
        L.append(f"| {b['tag']} | {b['design']['max_gap']*100:.2f} | **{b['test']['max_gap']*100:.2f}** | "
                 f"{b['test']['comp_dev']:.3f} | {b['test']['load']:.3f} | {sd} |")
    L += ["", "## Per-group detail (test)", "",
          "| group | risk pooled (pp) | risk parity-first (pp) | queue share pooled | queue share parity-first | traffic share |",
          "|---|---|---|---|---|---|"]
    for i, l in enumerate(langs):
        L.append(f"| {l} | {res['pooled']['test']['risk'][l]*100:.2f} | "
                 f"{res['parity_first']['test']['risk'][l]*100:.2f} | "
                 f"{res['pooled']['test']['K'][i]:.3f} | {res['parity_first']['test']['K'][i]:.3f} | "
                 f"{1.0/len(langs):.3f} |")
    p = res["prices"]
    L += ["", "## Prices", "",
          f"- price of composition (test max gap at K = w minus at parity-first): "
          f"**{p['price_of_composition_test']*100:+.2f} pp**",
          f"- price of parity (L1 queue deviation of the parity-first point): "
          f"**{p['price_of_parity_test']:.3f}**",
          f"- both objectives feasible: **{'yes' if p['both_feasible_test'] else 'no'}**",
          f"- pooled worst gap {p['worst_gap_pooled_test']*100:.2f} pp -> parity-first "
          f"{res['parity_first']['test']['max_gap']*100:.2f} pp", ""]
    if res.get("tradeoff_frontier"):
        L += ["## Trade-off frontier: the worst gap under a ceiling on the queue deviation", "",
              "`delta` is the largest deviation of the queue composition from the traffic shares the policy "
              "is allowed; the row reports the smallest worst pairwise gap attainable under that ceiling.",
              "", "| delta ceiling | design max gap (pp) | design K dev | test max gap (pp) | test K dev |",
              "|---|---|---|---|---|"]
        for t in res["tradeoff_frontier"]:
            L.append(f"| {t['delta']:g} | {t['design_max_gap']*100:.2f} | {t['design_comp_dev']:.3f} | "
                     f"{t['test_max_gap']*100:.2f} | {t['test_comp_dev']:.3f} |")
    if res.get("capacity_sweep"):
        L += ["", "## Capacity at which the worst gap closes (parity-first allocation per budget)", "",
              "| budget C | design max gap (pp) |", "|---|---|"]
        for c in res["capacity_sweep"]:
            g = "—" if c["design_max_gap"] is None else f"{c['design_max_gap']*100:.2f}"
            L.append(f"| {c['budget']:.0%} | {g} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--langs", required=True, help="comma list, e.g. en,de,uk,ar,hi,zh,ru")
    ap.add_argument("--budget", type=float, default=C_REF)
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    CP.LANGS = list(langs)
    seed_dirs = seed_dirs_of(args.out_root)
    assert seed_dirs, f"no probs.npz under {args.out_root}"
    res = analyse(seed_dirs, langs, args.budget, trace=not args.no_trace)
    res["generated"] = datetime.now().isoformat(timespec="seconds")
    res["detector"] = args.out_root
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("tradeoff_frontier", "capacity_sweep")}, indent=2)[:3000], flush=True)
    pf, po = res["parity_first"]["test"], res["pooled"]["test"]
    print(f"\nworst gap: pooled {po['max_gap']*100:.2f} pp -> parity-first {pf['max_gap']*100:.2f} pp "
          f"| composition deviation {pf['comp_dev_l1']:.3f} (L1)", flush=True)
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("json ->", args.json, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(markdown(res, args.title or f"CAPRA with L = {len(langs)} groups ({args.out_root})"))
        print("report ->", args.out, flush=True)


if __name__ == "__main__":
    main()
