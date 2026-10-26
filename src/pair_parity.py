"""Parity analysis for an ARBITRARY language pair (task 5b — textdetox en/de/uk/ar).

The EN/VI analysis lives in killtest_analyze.py + conra_parity.py and is deliberately hard-wired to
that pair (it knows about tr_* translated sets and the NAT/MAT/PAIR modes). This script reuses the
CONRA machinery (curves, risk-target inversion, bisection, rules, evaluation) for any pair of
languages present in a detector directory, so the same frontier metrics can be computed for
en<->de, en<->uk, en<->ar without touching the original pipeline.

Metrics per pair (balanced traffic, w = 1/2):
  quality  : acc / AUC / ECE per language on the eval split
  at C_ref : risk gap of GLOBAL, EQUAL-COV, CONRA + VI-side queue share
  structural min parity capacity + measured zero crossing + review load for parity at r = 5 %

Usage:
  python pair_parity.py --out-root outputs/textdetox/xlmr_base --lang-a en --lang-b de \
                        --out notes/pair_textdetox.md --json outputs/textdetox/pair_en_de.json
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

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_REF = 0.05
FINE = np.linspace(0.005, 0.40, 80)


def load_lang(seed_dirs, lang, part, match_prev=None):
    """Pooled (conf, err, labels, p1) for one language/part across seeds, optional prevalence match."""
    acc = {"conf": [], "err": [], "labels": [], "p1": []}
    for sd in seed_dirs:
        P, L = K.load(sd)
        key = f"{lang}_{part}"
        if key not in P:
            raise KeyError(key)
        p1, c, e = K.conf_err(P[key], L[key])
        acc["conf"].append(c)
        acc["err"].append(e)
        acc["labels"].append(L[key])
        acc["p1"].append(p1)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    if match_prev is not None:
        sel = K.match_subset(np.stack([1 - out["p1"], out["p1"]], axis=1), out["labels"], match_prev)
        out = {k: v[sel] for k, v in out.items()}
    return out


def build_pair(seed_dirs, a, b, match_prev=None):
    """Calibration from dev, evaluation from test, for both languages of the pair."""
    return {
        "cal": {a: load_lang(seed_dirs, a, "dev", match_prev),
                b: load_lang(seed_dirs, b, "dev", match_prev)},
        "test": {a: load_lang(seed_dirs, a, "test", match_prev),
                 b: load_lang(seed_dirs, b, "test", match_prev)},
    }


def min_parity_capacity(curves):
    r_nat = min(float(curves[l]["risk"][-1]) for l in CP.LANGS)
    for C in FINE:
        r, _ = CP.solve_parity_risk(C, curves)
        if r < r_nat:
            return float(C), r_nat, float(r)
    return None, r_nat, None


def gap_zero_crossing(d, curves):
    for C in FINE:
        ev = CP.evaluate(CP.rule_thresholds("CONRA", C, curves), d["test"])
        if ev["risk_gap"] >= 0:
            return float(C)
    return None


def quality(d):
    q = {}
    for l in CP.LANGS:
        q[l] = {"acc": float(1 - d["test"][l]["err"].mean()),
                "auc": K.auc(d["test"][l]["p1"], d["test"][l]["labels"]),
                "ece": K.ece(d["test"][l]["p1"], d["test"][l]["labels"]),
                "n": int(len(d["test"][l]["labels"]))}
    return q


def analyse(seed_dirs, a, b, match_prev=None):
    CP.LANGS = [a, b]                       # patch the pair into the shared machinery
    d = build_pair(seed_dirs, a, b, match_prev)
    curves = CP.curves_of(d["cal"])
    res = {"n": {l: len(d["test"][l]["conf"]) for l in (a, b)}, "quality": quality(d),
           "min_parity_capacity": min_parity_capacity(curves),
           "gap_zero_crossing": gap_zero_crossing(d, curves),
           "parity_load_at_r05": CP.load_parity(0.05, curves), "at_c_ref": {}}
    for rule in ("GLOBAL", "EQUAL-COV", "CONRA"):
        ev = CP.evaluate(CP.rule_thresholds(rule, C_REF, curves), d["test"])
        res["at_c_ref"][rule] = {
            "risk_gap": ev["risk_gap"],
            "cov": {l: ev[l]["coverage"] for l in (a, b)},
            "risk": {l: ev[l]["risk"] for l in (a, b)},
            "queue_share_b": ev["vi_share_of_queue"],   # ev names the second language "vi"
            "parity_rstar": CP.solve_parity_risk(C_REF, curves)[0] if rule == "CONRA" else None,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="detector dir with seed*/probs.npz")
    ap.add_argument("--lang-a", default="en")
    ap.add_argument("--lang-b", required=True, help="comma list to run several pairs in one go")
    ap.add_argument("--match-prevalence", type=float, default=None,
                    help="if set (e.g. 0.5), evaluate on prevalence-matched subsets")
    ap.add_argument("--c-ref", type=float, default=C_REF)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    seed_dirs = sorted(d for d in glob.glob(os.path.join(args.out_root, "seed*"))
                       if os.path.exists(os.path.join(d, "probs.npz")))
    if not seed_dirs and os.path.exists(os.path.join(args.out_root, "probs.npz")):
        seed_dirs = [args.out_root]
    assert seed_dirs, f"no probs.npz under {args.out_root}"

    res = {"generated": datetime.now().isoformat(timespec="seconds"), "detector": args.out_root,
           "seeds": [os.path.basename(d) for d in seed_dirs], "lang_a": args.lang_a,
           "c_ref": args.c_ref, "match_prevalence": args.match_prevalence, "pairs": {}}
    for b in [s.strip() for s in args.lang_b.split(",") if s.strip()]:
        print(f"[pair] {args.lang_a} <-> {b}", flush=True)
        res["pairs"][b] = analyse(seed_dirs, args.lang_a, b, args.match_prevalence)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("json ->", args.json, flush=True)

    if args.out:
        a = args.lang_a
        L = [f"# Pair parity — {a} vs each target ({os.path.basename(os.path.normpath(args.out_root))})\n",
             f"Generated {res['generated']} · balanced traffic · budget C = {args.c_ref:.0%} · "
             f"thresholds fitted on dev, evaluated on test · seeds: {', '.join(res['seeds'])}"
             + (f" · prevalence-matched to {args.match_prevalence}" if args.match_prevalence else "") + "\n",
             "\nStructural `min parity cap.` = first capacity where both languages are strictly thresholded "
             "(parity stops being one-sided); `Δ=0 crossing` = first capacity where the measured test gap hits 0.\n",
             f"\n| target | n/lang | acc {a} | acc tgt | AUC {a} | AUC tgt | ECE {a} | ECE tgt | "
             f"GLOBAL Δ (pp) | EQUAL-COV Δ | CONRA Δ | queue share tgt | min parity cap. | r* @ min cap | "
             f"Δ=0 crossing | parity load @r=5% |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for b, m in res["pairs"].items():
            q, c = m["quality"], m["at_c_ref"]
            cap, r_nat, r_cap = m["min_parity_capacity"]
            cross = m["gap_zero_crossing"]
            L.append(f"| {b} | {m['n'][a]} | {q[a]['acc']:.4f} | {q[b]['acc']:.4f} | {q[a]['auc']:.4f} | "
                     f"{q[b]['auc']:.4f} | {q[a]['ece']:.4f} | {q[b]['ece']:.4f} | "
                     f"**{c['GLOBAL']['risk_gap']*100:+.2f}** | {c['EQUAL-COV']['risk_gap']*100:+.2f} | "
                     f"{c['CONRA']['risk_gap']*100:+.2f} | {c['GLOBAL']['queue_share_b']:.3f} | "
                     f"{'—' if cap is None else f'{cap*100:.1f} %'} | "
                     f"{'—' if r_cap is None else f'{r_cap*100:.1f} %'} | "
                     f"{'—' if cross is None else f'{cross*100:.1f} %'} | {m['parity_load_at_r05']*100:.1f} % |")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print("report ->", args.out, flush=True)


if __name__ == "__main__":
    main()
