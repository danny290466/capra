"""Statistical significance for the parity comparisons (IPM convention: paired bootstrap over items,
seed-level aggregation for retrained detectors, dagger markers in the tables).

Two levels, because the paper makes two kinds of claim:

1. ITEM LEVEL (paired bootstrap). All rules are evaluated on the SAME bootstrap resample of the test
   split (independently drawn per language), so differences between rules are paired and the
   calibration thresholds stay fixed — we quantify the sampling variability of the measured disparity,
   not the variability of refitting.
       D_r(b) for b = 1..B, then
       CI_r = percentile interval of D_r,   p(r, r') = 2 min(P(D_r - D_r' > 0), P(< 0)).

2. SEED LEVEL (retrained detectors). The whole fit -> calibrate -> measure pipeline is repeated for each
   training seed, giving one disparity per seed; we report mean ± SD and a paired t-test across seeds
   between rules. A rule pair is marked 'not different' when the seed-level paired test is not
   significant at alpha = 0.05.

Output: notes/significance_<detector>.md (tables with daggers) and <root>/significance.json.

Usage:
  python significance.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
      --budgets 0.05 --out notes/significance_killtest_NAT.md --json outputs/killtest/significance_NAT.json
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capra as CA             # noqa: E402
import conra_parity as CP      # noqa: E402
import baseline_rules as BR    # noqa: E402

B = 1000
ALPHA = 0.05


# ---------------------------------------------------------------- data
def load_pair(out_root, a, b, mode, one_seed=None):
    seed_dirs = sorted(d for d in glob.glob(os.path.join(out_root, "seed*"))
                       if os.path.exists(os.path.join(d, "probs.npz")))
    if not seed_dirs and os.path.exists(os.path.join(out_root, "probs.npz")):
        seed_dirs = [out_root]
    assert seed_dirs, f"no probs.npz under {out_root}"
    if one_seed is not None:
        seed_dirs = [seed_dirs[one_seed]]
    CP.LANGS = [a, b]
    if (a, b) == ("en", "vi"):
        return CP.prep(seed_dirs, mode), seed_dirs
    import pair_parity as PP
    return PP.build_pair(seed_dirs, a, b), seed_dirs


# ---------------------------------------------------------------- rules
def rule_thresholds(d, cp_curves, ca_curves, langs, C, k_star):
    """Threshold per language for every rule, fitted on the calibration split.

    `cp_curves` are conra_parity's curve dicts (with the 'raw' arrays the CP rules need);
    `ca_curves` are capra's (review-share, risk) pairs used by the allocation sweep.
    """
    out = {}
    for rule in ("GLOBAL", "EQUAL-COV", "CONRA"):
        out[rule] = CP.rule_thresholds(rule, C, cp_curves)
    front = CA.sweep(ca_curves, C, langs, {langs[0]: 1.0, langs[1]: 1.0})
    feas = [p for p in front if not p["clipped"]] or front
    pf = min(feas, key=lambda p: p["D_abs"])
    out["CAPRA-parity"] = {l: CA.tau_from_share(d["cal"][l]["conf"], pf["s"][l]) for l in langs}
    cf = min(feas, key=lambda p: abs(p["K"] - k_star))
    out["CAPRA-composition"] = {l: CA.tau_from_share(d["cal"][l]["conf"], cf["s"][l]) for l in langs}
    # group-conditional risk control, defined EXACTLY as in baseline_rules.py: sweep the per-group risk
    # target and interpolate the family to the same load as the budget. Using a second definition here
    # made Table 7 and Table 8 disagree about the same baseline (caught in review).
    traj = BR.traj_design_risk_targets(ca_curves, langs, {langs[0]: 1.0, langs[1]: 1.0}, (0.5, 0.5))
    pt = BR.interp_at_load(traj, C)
    if pt is not None:
        out["ABS-RISK"] = {l: CA.tau_from_share(d["cal"][l]["conf"], pt["s"][l]) for l in langs}
    return out



def disparity(conf, err, tau, langs):
    """|risk_a - risk_b| and the per-language risks for one rule on one sample."""
    r = {}
    for l in langs:
        c, e, t = conf[l], err[l], tau[l]
        kept = c >= t
        r[l] = float(e[kept].mean()) if kept.any() else float("nan")
    return abs(r[langs[0]] - r[langs[1]]), r


def as_items(arr, n_seeds):
    """Reshape a per-seed concatenation into (n_seeds, n_items).

    For a trained detector `load_pair` concatenates the SAME test items once per training seed, so the
    naive item bootstrap resampled pseudo-replicates (25 000 rows for 5 000 items) and produced
    intervals that were too narrow — and an effective n five times larger than the hosted moderators'
    (review finding R2-13). Reshaping lets us resample unique items and average over seeds.
    """
    n = len(arr)
    if n_seeds > 1:
        # the old fallback silently reshaped to (1, n), which restored the pseudo-replication this
        # function exists to remove: a mismatch is a bug, not something to paper over (review R3-19)
        assert n % n_seeds == 0, (f"per-seed concatenation of {n} rows is not divisible by "
                                  f"{n_seeds} seeds - the cluster bootstrap cannot be formed")
        return np.asarray(arr).reshape(n_seeds, n // n_seeds)
    return np.asarray(arr).reshape(1, n)


def risk_from_items(conf2, err2, tau, idx, diag=None):
    """Cluster bootstrap: resample ITEM columns, average the kept-set risk over seeds.

    Seeds whose kept set is empty are dropped by nanmean; the count is recorded in `diag` and reported,
    because a rule that fails often would otherwise get an artificially narrow interval (review R3-19).
    """
    c = conf2[:, idx]
    e = err2[:, idx]
    kept = c >= tau
    per_seed = np.where(kept.sum(axis=1) > 0,
                        (e * kept).sum(axis=1) / np.maximum(kept.sum(axis=1), 1),
                        np.nan)
    good = np.isfinite(per_seed)
    if diag is not None:
        diag["dropped_seed_risk"] = diag.get("dropped_seed_risk", 0) + int((~good).sum())
        diag["replicates_with_no_seed"] = diag.get("replicates_with_no_seed", 0) + int(good.sum() == 0)
    return float(np.nanmean(per_seed))


# ---------------------------------------------------------------- bootstrap
def bootstrap(test, tau_rules, langs, rng, b=B, n_seeds=1):
    """Cluster bootstrap over UNIQUE test items (see as_items): one resample of items per replicate,
    the rule's risk averaged over training seeds, so every row of the table has the same effective n."""
    conf2 = {l: as_items(test[l]["conf"], n_seeds) for l in langs}
    err2 = {l: as_items(test[l]["err"], n_seeds) for l in langs}
    n_items = {l: conf2[l].shape[1] for l in langs}
    dist = {rule: np.empty(b) for rule in tau_rules}
    dist_signed = {rule: np.empty(b) for rule in tau_rules}
    diag = {"dropped_seed_risk": 0, "replicates_with_no_seed": 0, "n_items": dict(n_items),
            "n_seeds": int(n_seeds), "replicates": int(b)}
    for i in range(b):
        idx = {l: rng.integers(0, n_items[l], n_items[l]) for l in langs}
        for rule, tau in tau_rules.items():
            r = {l: risk_from_items(conf2[l], err2[l], tau[l], idx[l], diag) for l in langs}
            # the SIGNED gap is what the paper's claim is about; |D| alone cannot show a rule whose gap
            # flips sign across seeds (review finding R3-18)
            dist_signed[rule][i] = r[langs[0]] - r[langs[1]]
            dist[rule][i] = abs(r[langs[0]] - r[langs[1]])
    return dist, dist_signed, diag


def paired_p(d1, d2):
    diff = d1 - d2
    p = 2.0 * min(float((diff > 0).mean()), float((diff < 0).mean()))
    return min(1.0, max(p, 1.0 / len(diff)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lang-a", default="en")
    ap.add_argument("--lang-b", required=True)
    ap.add_argument("--mode", default="NAT")
    ap.add_argument("--budgets", default="0.05")
    ap.add_argument("--composition-target", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--reference", default=None,
                    help="rule to use as the reference ('best') for the two p columns. The default is the "
                         "test-split minimum over the rule list; pass a name when the manuscript's table "
                         "does not print the rule that happens to win, so that the p columns are computed "
                         "against the reference the reader can actually see")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    a = args.lang_a
    res = {"generated": datetime.now().isoformat(timespec="seconds"), "detector": args.out_root,
           "lang_a": a, "mode": args.mode, "boot": args.boot, "alpha": ALPHA, "pairs": {}}
    lines = [f"# Significance — `{args.out_root}` ({a} vs targets, mode {args.mode})\n",
             f"Generated {res['generated']}. Item-level paired bootstrap (B = {args.boot}, thresholds "
             f"fixed at their calibration values) and seed-level paired t-tests over the training seeds. "
             f"A dagger (†) marks a rule that is **not** significantly different from the best rule in "
             f"that column at alpha = {ALPHA}.\n"]

    for b_lang in [s.strip() for s in args.lang_b.split(",") if s.strip()]:
        langs = (a, b_lang)
        d, seed_dirs = load_pair(args.out_root, a, b_lang, args.mode)
        cp_curves = CP.curves_of(d["cal"])
        ca_curves = {l: CA.review_risk_curve(d["cal"][l]["conf"], d["cal"][l]["err"]) for l in langs}
        res["pairs"][b_lang] = {}
        for C in [float(x) for x in args.budgets.split(",")]:
            tau_rules = rule_thresholds(d, cp_curves, ca_curves, langs, C, args.composition_target)
            rng = np.random.default_rng(20260923)
            dist, dist_signed, bdiag = bootstrap(d["test"], tau_rules, langs, rng, args.boot,
                                                 n_seeds=len(seed_dirs))
            # Point estimates are computed ONCE on the original test split, with the SAME estimator for
            # both columns, so that D = |delta| holds exactly and the bootstrap supplies intervals only
            # (review round 5, item 5). `point_signed_boot_mean` keeps the old convention for reference.
            _full = {rule: disparity({l: d["test"][l]["conf"] for l in langs},
                                     {l: d["test"][l]["err"] for l in langs}, tau, langs)
                     for rule, tau in tau_rules.items()}
            point = {rule: v[0] for rule, v in _full.items()}
            point_signed_orig = {rule: v[1][langs[0]] - v[1][langs[1]] for rule, v in _full.items()}
            ci = {r: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for r, v in dist.items()}
            ci_signed = {r: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                         for r, v in dist_signed.items()}
            point_signed_boot_mean = {r: float(np.mean(dist_signed[r])) for r in dist_signed}
            point_signed = point_signed_orig
            best = min(point, key=lambda r: point[r])
            if args.reference:
                if args.reference not in point:
                    sys.exit(f"--reference {args.reference!r} is not in the rule set {sorted(point)}")
                best = args.reference
            # seed-level: refit the whole pipeline per seed
            seed_rows = []
            for i, sd in enumerate(seed_dirs):
                d_i, _ = load_pair(args.out_root, a, b_lang, args.mode, one_seed=i)
                cp_i = CP.curves_of(d_i["cal"])
                ca_i = {l: CA.review_risk_curve(d_i["cal"][l]["conf"], d_i["cal"][l]["err"]) for l in langs}
                tau_i = rule_thresholds(d_i, cp_i, ca_i, langs, C, args.composition_target)
                row = {r: disparity({l: d_i["test"][l]["conf"] for l in langs},
                                    {l: d_i["test"][l]["err"] for l in langs}, t, langs)[0]
                       for r, t in tau_i.items()}
                row["seed"] = os.path.basename(sd)
                seed_rows.append(row)
            seed_test = {}
            if len(seed_rows) > 1:
                for r1 in point:
                    for r2 in point:
                        if r1 >= r2 or r1 not in seed_rows[0] or r2 not in seed_rows[0]:
                            continue
                        x = np.array([s[r1] for s in seed_rows], dtype=float)
                        y = np.array([s[r2] for s in seed_rows], dtype=float)
                        diff = x - y
                        if np.allclose(diff, 0):
                            p = 1.0
                        else:
                            se = diff.std(ddof=1) / np.sqrt(len(diff))
                            tstat = diff.mean() / se if se > 0 else 0.0
                            # Student t with df = n-1: at n = 5 the normal approximation is
                            # anti-conservative (1.96 vs 2.776), review finding R2-12
                            df = len(diff) - 1
                            try:
                                from scipy import stats as _st
                                p = float(2.0 * _st.t.sf(abs(tstat), df))
                            except Exception:                          # noqa: BLE001
                                crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(df, 1.96)
                                p = 1.0 if abs(tstat) < crit else 0.01
                        seed_test[f"{r1}|{r2}"] = float(p)
            res["pairs"][b_lang][f"{C:.2f}"] = {"point": point, "ci": ci, "best": best,
                                                "point_signed": point_signed,
                                                "point_signed_boot_mean": point_signed_boot_mean,
                                                "ci_signed": ci_signed,
                                                "bootstrap_diagnostics": bdiag,
                                                "reference_rule": best,
                                                "p_vs_best_item": {r: (1.0 if r == best else
                                                                       paired_p(dist[r], dist[best]))
                                                                   for r in dist},
                                                "seed_rows": seed_rows, "seed_p": seed_test}

            lines.append(f"\n## target: {b_lang} · C = {C:.2f}\n")
            lines.append("| rule | disparity D (pp) | 95% bootstrap CI | p vs best (item-level) | "
                         f"p vs best (seed-level, n = {max(len(seed_rows), 1)}) | "
                         "seed mean ± SD (pp) |")
            lines.append("|---|---|---|---|---|---|")
            sp = res["pairs"][b_lang][f"{C:.2f}"]["seed_p"]
            for r in sorted(point, key=lambda x: point[x]):
                # the dagger follows the SEED-level test: at ~12k test items the item-level paired
                # bootstrap calls a 0.01 pp difference significant, which is sampling noise, not a result
                key = "|".join(sorted([r, best]))
                p_seed = sp.get(key)
                if r == best:
                    mark = ""
                elif p_seed is not None:                      # trained detector: seed-level test
                    mark = "†" if p_seed > ALPHA else ""
                else:
                    # a single-run detector has no seed-level test; the item-level test at ~12k items
                    # flags sub-0.05 pp differences, so no dagger is assigned (caption says so)
                    mark = ""
                sd_txt = "—"
                if len(seed_rows) > 1 and r in seed_rows[0]:
                    vals = np.array([s[r] for s in seed_rows], dtype=float)
                    sd_txt = f"{vals.mean()*100:.2f} ± {vals.std(ddof=1)*100:.2f} (n={len(vals)})"
                elif len(seed_rows) == 1 and r in seed_rows[0]:
                    sd_txt = f"{seed_rows[0][r]*100:.2f} (single run)"
                lines.append(f"| {r}{mark} | **{point[r]*100:.2f}** | "
                             f"[{ci[r][0]*100:.2f}, {ci[r][1]*100:.2f}] | "
                             f"{res['pairs'][b_lang][f'{C:.2f}']['p_vs_best_item'][r]:.3f} | "
                             f"{'—' if p_seed is None else f'{p_seed:.3f}'} | {sd_txt} |")
            print(f"[{args.out_root}] {a}/{b_lang} C={C:.2f} done (best = {best})", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(res, open(args.json, "w", encoding="utf-8"), indent=2)
        print("json ->", args.json, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print("report ->", args.out, flush=True)


if __name__ == "__main__":
    main()
