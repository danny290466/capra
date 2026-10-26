"""CONRA-Parity — Constrained Risk-parity Abstention under a shared human-review budget.

Task 4 of CONTEXT. CPU-only: reads the kill-test artefacts (outputs/killtest/seed*/probs.npz),
fits thresholds on the calibration split, and compares abstention rules at a fixed review capacity.

Rules compared at each budget C (share of the traffic a human team can review):
  R1 GLOBAL      one threshold for both languages, budget exactly spent             (deployment default)
  R2 EQUAL-COV   per-language thresholds with EQUAL coverage, budget exactly spent  (equalized conformal)
  R3 CONRA       per-language thresholds with EXACT risk parity, budget exactly spent (our rule)
  R4 PARITY-COST risk parity at a fixed target risk, budget NOT capped               (diagnostic: the
                 review capacity parity costs, versus a global rule spending the same load)

Metrics: per-language coverage/risk, risk gap, VI share of the review queue, review load.
Bootstrap CIs (calibration + test resampled) for the risk gap.

Performance note: calibration risk–coverage curves are built ONCE per bootstrap draw; the risk-parity
root-find then costs O(log n) per step instead of a full argsort (the naive version took ~30 min).
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import killtest_analyze as K  # noqa: E402  (reuse load / build_sets)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = [0.05, 0.10, 0.20]
RISK_GRID = [0.02, 0.03, 0.05]
N_BOOT = 1000
RNG_SEED = 11
LANGS = ("en", "vi")


# ------------------------------------------------------------------ curves & thresholds
def build_curve(conf, err):
    """Sorted-by-confidence cumulative risk/coverage curve (calibration side only)."""
    order = np.argsort(-conf, kind="mergesort")
    c = conf[order]
    e = err[order]
    k = np.arange(1, len(e) + 1, dtype=float)
    return {"conf": c, "risk": np.cumsum(e) / k, "cov": k / len(e), "raw": conf}


def tau_at_risk(curve, r):
    ok = np.where(curve["risk"] <= r)[0]
    return None if ok.size == 0 else float(curve["conf"][ok[-1]])


def cov_of_curve_at(curve, tau):
    return float((curve["conf"] >= tau).mean())


def cov_of_tau(conf, tau):
    return float((conf >= tau).mean())


def risk_of_tau(conf, err, tau):
    keep = conf >= tau
    return float(err[keep].mean()) if keep.any() else float("nan")


def load_parity(r, curves):
    """Review load (share of balanced traffic) that risk parity at level r requires."""
    tot = 0.0
    for l in LANGS:
        tau = tau_at_risk(curves[l], r)
        tot += 1.0 - (0.0 if tau is None else cov_of_curve_at(curves[l], tau))
    return tot / 2.0


def solve_parity_risk(C, curves):
    """Largest load is at the smallest risk; bisect on r with load_parity(r) = C."""
    lo, hi = 1e-4, 0.90
    if load_parity(lo, curves) <= C:
        return lo, load_parity(lo, curves)
    if load_parity(hi, curves) >= C:
        return hi, load_parity(hi, curves)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if load_parity(mid, curves) > C:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    return r, load_parity(r, curves)


def rule_thresholds(rule, C, curves):
    tau = {}
    if rule == "GLOBAL":
        pooled = np.concatenate([curves[l]["raw"] for l in LANGS])
        t = float(np.quantile(pooled, C, method="lower"))   # keep the top (1-C) fraction
        tau = {l: t for l in LANGS}                          # one threshold for every language
    elif rule == "EQUAL-COV":
        for l in LANGS:
            tau[l] = float(np.quantile(curves[l]["raw"], C, method="lower"))
    elif rule == "CONRA":
        r, _ = solve_parity_risk(C, curves)
        for l in LANGS:
            t = tau_at_risk(curves[l], r)
            tau[l] = 0.0 if t is None else t
    else:
        raise ValueError(rule)
    return tau


def evaluate(tau, test):
    """Per-language coverage/risk plus the pair-level aggregates.

    Uses the module-level LANGS (which callers set for an arbitrary pair, e.g. en/de in pair_parity)
    so this stays correct beyond EN/VI. `vi_share_of_queue` keeps its name for EN/VI but means
    "share of the queue from the SECOND language" for a generic pair.
    """
    first, second = LANGS
    out = {}
    for l in LANGS:
        c = cov_of_tau(test[l]["conf"], tau[l])
        out[l] = {"coverage": c, "risk": risk_of_tau(test[l]["conf"], test[l]["err"], tau[l]),
                  "review_per_1k": 1000 * (1 - c)}
    out["review_load"] = 1.0 - 0.5 * (out[first]["coverage"] + out[second]["coverage"])
    out["risk_gap"] = out[first]["risk"] - out[second]["risk"]
    out["coverage_gap"] = out[first]["coverage"] - out[second]["coverage"]
    q_first = 0.5 * (1 - out[first]["coverage"])
    q_second = 0.5 * (1 - out[second]["coverage"])
    out["both_review_per_1k"] = (out[first]["review_per_1k"], out[second]["review_per_1k"])
    out["queue_share_second"] = q_second / (q_first + q_second) if (q_first + q_second) > 0 else float("nan")
    out["vi_share_of_queue"] = out["queue_share_second"]      # back-compat alias (EN/VI)
    out["pooled_residual_err"] = (q_first * out[first]["risk"] + q_second * out[second]["risk"])
    return out


def prep(seed_dirs, mode, ts=False, balance=False):
    """Pooled calibration/test arrays for a mode, from the kill-test artefacts.

    balance=False (default) keeps every item of both languages and weights the two languages equally
    (w = 1/2) when averaging the review load — the convention used across every table in the paper.
    balance=True subsamples to equal per-language volume instead (kept for the older kill-test run).
    """
    acc = {"cal": {l: [] for l in LANGS}, "test": {l: [] for l in LANGS}}
    for sd in seed_dirs:
        P, L = K.load(sd)
        out = K.build_sets(P, L, mode, ts=ts)
        acc["cal"]["en"].append(out[0])
        acc["cal"]["vi"].append(out[1])
        acc["test"]["en"].append(out[2])
        acc["test"]["vi"].append(out[3])
    rng = np.random.default_rng(RNG_SEED)
    res = {"cal": {}, "test": {}}
    for part in ("cal", "test"):
        for l in LANGS:
            res[part][l] = {k: np.concatenate([o[k] for o in acc[part][l]])
                            for k in ("conf", "err", "labels", "p1")}
        if balance:
            n = min(len(res[part][l]["conf"]) for l in LANGS)
            for l in LANGS:
                d = res[part][l]
                if len(d["conf"]) > n:
                    idx = np.sort(rng.choice(len(d["conf"]), size=n, replace=False))
                    res[part][l] = {k: v[idx] for k, v in d.items()}
    return res


def curves_of(d):
    return {l: build_curve(d[l]["conf"], d[l]["err"]) for l in LANGS}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.path.join(PROJ, "outputs", "killtest"))
    ap.add_argument("--report", default=os.path.join(PROJ, "notes", "conra_parity_RESULT.md"))
    ap.add_argument("--json-out", default=os.path.join(PROJ, "outputs", "killtest", "conra_parity.json"))
    ap.add_argument("--ts", action="store_true", help="temperature-scaled confidences variant")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    seed_dirs = sorted(d for d in glob.glob(os.path.join(args.out_root, "seed*"))
                       if os.path.exists(os.path.join(d, "probs.npz")))
    assert seed_dirs, "no probs.npz found"
    seeds = [os.path.basename(d) for d in seed_dirs]

    modes = []
    data = {}
    for mode in ("NAT", "MAT", "PAIR"):
        try:
            data[mode] = prep(seed_dirs, mode, ts=args.ts)
            modes.append(mode)
        except KeyError:
            pass
    print("seeds:", seeds, "| modes:", modes, "| ts:", args.ts, flush=True)

    res = {"seeds": seeds, "ts": bool(args.ts), "budgets": BUDGETS, "n_boot": args.n_boot,
           "generated": datetime.now().isoformat(timespec="seconds"), "modes": {}}

    for mode in modes:
        d = data[mode]
        m = {"n": {l: len(d["test"][l]["conf"]) for l in LANGS}, "budgets": {}}
        for C in BUDGETS:
            entry = {"rules": {}}
            for rule in ("GLOBAL", "EQUAL-COV", "CONRA"):
                tau = rule_thresholds(rule, C, curves_of(d["cal"]))
                ev = evaluate(tau, d["test"])
                ev["tau"] = dict(tau)
                ev["parity_risk_level"] = solve_parity_risk(C, curves_of(d["cal"]))[0] \
                    if rule == "CONRA" else None
                rng = np.random.default_rng(RNG_SEED)
                gaps = []
                for _ in range(args.n_boot):
                    cal_b, test_b = {}, {}
                    for l in LANGS:
                        i = rng.integers(0, len(d["cal"][l]["conf"]), len(d["cal"][l]["conf"]))
                        cal_b[l] = {k: d["cal"][l][k][i] for k in d["cal"][l]}
                        j = rng.integers(0, len(d["test"][l]["conf"]), len(d["test"][l]["conf"]))
                        test_b[l] = {k: d["test"][l][k][j] for k in d["test"][l]}
                    t = rule_thresholds(rule, C, curves_of(cal_b))
                    gaps.append(risk_of_tau(test_b["en"]["conf"], test_b["en"]["err"], t["en"]) -
                                risk_of_tau(test_b["vi"]["conf"], test_b["vi"]["err"], t["vi"]))
                gaps = np.array([g for g in gaps if not np.isnan(g)])
                ev["risk_gap_ci95"] = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
                entry["rules"][rule] = ev
                print(f"  [{mode} C={C:.2f}] {rule} done", flush=True)
            entry["parity_cost"] = {}
            for r in RISK_GRID:
                needed = load_parity(r, curves_of(d["cal"]))
                ev_g = evaluate(rule_thresholds("GLOBAL", min(needed, 0.95), curves_of(d["cal"])), d["test"])
                entry["parity_cost"][f"{r:.2f}"] = {
                    "load_needed_for_parity": needed,
                    "risk_gap_under_global_at_same_load": ev_g["risk_gap"],
                    "global_coverage": {l: ev_g[l]["coverage"] for l in LANGS},
                }
            m["budgets"][f"{C:.2f}"] = entry
        res["modes"][mode] = m
        print(f"[{mode}] done", flush=True)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    L = []
    A = L.append
    tag = " (temperature-scaled confidences)" if args.ts else ""
    A(f"# CONRA-Parity — constrained risk-parity abstention under a shared review budget{tag}\n")
    A(f"- Generated {res['generated']} (Asia/Ho_Chi_Minh) | seeds: {', '.join(seeds)} | "
      f"bootstrap draws: {args.n_boot}\n"
      f"- Input: kill-test artefacts `outputs/killtest/seed*/probs.npz` (CPU only, no retraining)\n"
      f"- Traffic mix assumed **balanced** (equal volume per language); the budget `C` is the share "
      f"of that traffic the human team can review, `C ∈ {[f'{c:.0%}' for c in BUDGETS]}`\n"
      f"- Thresholds are always fitted on the dev/calibration split, then applied to the test split\n")
    A("\n**Rules.** R1 `GLOBAL`: one threshold for both languages, budget exactly spent. "
      "R2 `EQUAL-COV`: per-language thresholds with equal coverage (the equalized-conformal "
      "baseline), budget exactly spent. R3 `CONRA`: per-language thresholds with *exact risk "
      "parity*, budget exactly spent — the common risk level r* is solved for. "
      "`risk_gap = r_EN − r_VI`; negative means VI is left riskier by the rule.\n"
      "\n`queue share VI` = fraction of the review queue that is Vietnamese content "
      "(0.5 = the queue reflects the traffic); `review load` = realised share of traffic reviewed.\n")

    for mode in modes:
        m = res["modes"][mode]
        A(f"\n## {mode} (n per language = {m['n']['en']})\n")
        for C in BUDGETS:
            e = m["budgets"][f"{C:.2f}"]
            A(f"\n### budget C = {C*100:.0f}% of traffic reviewed\n")
            A("| Rule | τ_EN | τ_VI | cov EN | cov VI | risk EN | risk VI | risk gap (pp) | 95 % CI | "
              "queue share VI | review load | parity r* | risk premium EN (pp) |")
            A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            g_risk_en = e["rules"]["GLOBAL"]["en"]["risk"]
            for rule in ("GLOBAL", "EQUAL-COV", "CONRA"):
                ev = e["rules"][rule]
                ci = ev["risk_gap_ci95"]
                rstar = ev["parity_risk_level"]
                rstar_s = "—" if rstar is None else f"{rstar:.4f}"
                prem = "—" if rstar is None else f"{(rstar - g_risk_en)*100:+.2f}"
                A(f"| {rule} | {ev['tau']['en']:.3f} | {ev['tau']['vi']:.3f} | "
                  f"{ev['en']['coverage']:.4f} | {ev['vi']['coverage']:.4f} | "
                  f"{ev['en']['risk']:.4f} | {ev['vi']['risk']:.4f} | "
                  f"**{ev['risk_gap']*100:+.2f}** | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | "
                  f"{ev['vi_share_of_queue']:.3f} | {ev['review_load']:.4f} | {rstar_s} | {prem} |")
            A("\n**Cost of parity (uncapped).** Review load that exact risk parity requires at a fixed "
              "target risk, and what a single global threshold leaves as a risk gap when it is given "
              "exactly that much review capacity:\n")
            A("| target risk r | load needed for parity | risk gap if GLOBAL spends that load (pp) |")
            A("|---|---|---|")
            for r, v in e["parity_cost"].items():
                A(f"| {r} | {v['load_needed_for_parity']:.4f} | "
                  f"{v['risk_gap_under_global_at_same_load']*100:+.2f} |")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("report ->", args.report, flush=True)
    print("json   ->", args.json_out, flush=True)


if __name__ == "__main__":
    main()
