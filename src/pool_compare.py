"""Cross-detector comparison (task 5) — does the EN/VI abstention finding survive a wider pool?

For every detector directory (each holding seed*/probs.npz) it reports, per mode (NAT/MAT/PAIR):

  * detector quality per language: accuracy, AUC, ECE on the test split
  * at a 5 % review budget: the risk gap of the three rules and the VI share of the review queue
  * the **minimum review capacity for language parity** (first budget where CONRA's gap crosses 0)
  * the review load that exact risk parity needs at a 5 % target risk (the cost of parity)

Writes a markdown table + json. Usage:
  python pool_compare.py --out notes/pool_compare_RESULT.md [--json outputs/pool/pool_compare.json]
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

PROJ = os.environ.get("SP_PROJ") or (
    os.getcwd() if os.path.isdir(os.path.join(os.getcwd(), "outputs"))
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C_REF = 0.05
FINE = np.linspace(0.005, 0.40, 80)


def quality(d, mode):
    """acc / AUC / ECE per language on the evaluation split of a mode."""
    out = {}
    for l in CP.LANGS:
        conf, err, labels, p1 = d["test"][l]["conf"], d["test"][l]["err"], \
            d["test"][l]["labels"], d["test"][l]["p1"]
        out[l] = {"acc": float(1 - err.mean()), "auc": K.auc(p1, labels), "ece": K.ece(p1, labels),
                  "n": int(len(labels))}
    return out


def _min_parity_capacity_local(curves):   # kept only as documentation; the live one is in pair_parity
    pass


def min_parity_capacity(curves):
    """Structural threshold: first budget at which parity is TWO-SIDED.

    A language can only be held at risk `r` if `r < r_l^nat` (its full-coverage risk): above that the
    language is saturated at coverage 1 and the constraint degenerates to `max_l r_l <= r*`. So parity
    is genuinely enforced once `r*` drops below the *best* full-coverage risk. Tested on `r*` rather
    than on coverage, because discretised scores (an LLM's integer probabilities) make coverage-based
    saturation tests jumpy.
    """
    r_nat = min(float(curves[l]["risk"][-1]) for l in CP.LANGS)
    for C in FINE:
        r, _ = CP.solve_parity_risk(C, curves)
        if r < r_nat:
            return float(C), r_nat, float(r)
    return None, r_nat, None


def gap_zero_crossing(d, curves):
    """Measured threshold: first budget at which CONRA's TEST risk gap reaches >= 0.

    Reaches zero earlier than the structural threshold when the better language's calibration risk
    is below its test risk — i.e. it is partly transfer luck, not enforced parity. Report both.
    """
    for C in FINE:
        ev = CP.evaluate(CP.rule_thresholds("CONRA", C, curves), d["test"])
        if ev["risk_gap"] >= 0:
            return float(C)
    return None


def refusal_counts(root, seed_dirs):
    """LLM moderators can refuse to score content (content policy). Report the unscored rate per set
    from the `refused_<set>.jsonl` artefacts, with the npz item count as denominator."""
    out = {}
    files = glob.glob(os.path.join(root, "refused_*.jsonl"))
    if not files:
        return out
    import numpy as _np
    z = _np.load(os.path.join(seed_dirs[0], "probs.npz"))
    n_by_set = {k.replace("_labels", ""): int(len(z[k])) for k in z.files if k.endswith("_labels")}
    for f in files:
        name = os.path.basename(f)[len("refused_"):-len(".jsonl")]
        n = sum(1 for _ in open(f, encoding="utf-8"))
        if n:
            out[name] = (n, n_by_set.get(name, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors",
                    # llm_qwen38_flash is deliberately absent: it left 397 EN/VI items unscored, and an
                    # unscored item is NaN, not a score, so it cannot sit in a table beside detectors that
                    # scored every item (review findings R3-05, R3-06).
                    default="xlmr_base:outputs/killtest,xlmr_large:outputs/pool/xlmr_large,"
                            "specialist_pair:outputs/pool/specialist_pair,"
                            "llm_deepseek_flash:outputs/pool/llm_deepseek_flash,"
                            "llm_gemini31_flashlite:outputs/pool/llm_gemini31_flashlite",
                    help="detector:dir list. The monolingual specialists (roberta_en, phobert_vi) are "
                         "INPUTS to specialist_pair, not standalone cross-lingual detectors — their "
                         "cross-language numbers are meaningless, so they are not in this table.")
    ap.add_argument("--out", default=os.path.join(PROJ, "notes", "pool_compare_RESULT.md"))
    ap.add_argument("--json", default=os.path.join(PROJ, "outputs", "pool", "pool_compare.json"))
    args = ap.parse_args()

    res = {"generated": datetime.now().isoformat(timespec="seconds"), "c_ref": C_REF, "detectors": {}}
    for spec in args.detectors.split(","):
        name, rel = spec.split(":", 1)
        root = rel if os.path.isabs(rel) else os.path.join(PROJ, rel)
        seed_dirs = sorted(d for d in glob.glob(os.path.join(root, "seed*"))
                           if os.path.exists(os.path.join(d, "probs.npz")))
        if not seed_dirs and os.path.exists(os.path.join(root, "probs.npz")):
            seed_dirs = [root]          # single-run detectors (e.g. the LLM moderator) have no seeds
        if not seed_dirs:
            print(f"[skip] {name}: no seed dirs under {root}", flush=True)
            continue
        entry = {"root": root, "seeds": [os.path.basename(d) for d in seed_dirs],
                 "refused": refusal_counts(root, seed_dirs), "modes": {}}
        for mode in ("NAT", "MAT", "PAIR"):
            try:
                d = CP.prep(seed_dirs, mode)
            except KeyError as e:
                print(f"[skip] {name}/{mode}: missing set {e}", flush=True)
                continue
            curves = CP.curves_of(d["cal"])
            m = {"n": {l: len(d["test"][l]["conf"]) for l in CP.LANGS}, "quality": quality(d, mode)}
            m["at_c_ref"] = {}
            for rule in ("GLOBAL", "EQUAL-COV", "CONRA"):
                ev = CP.evaluate(CP.rule_thresholds(rule, C_REF, curves), d["test"])
                m["at_c_ref"][rule] = {
                    "risk_gap": ev["risk_gap"], "cov": {l: ev[ l ]["coverage"] for l in CP.LANGS},
                    "risk": {l: ev[l]["risk"] for l in CP.LANGS},
                    "queue_vi": ev["vi_share_of_queue"],
                    "parity_rstar": CP.solve_parity_risk(C_REF, curves)[0] if rule == "CONRA" else None,
                }
            cap, r_nat, r_cap = min_parity_capacity(curves)
            m["min_parity_capacity"] = cap
            m["calib_full_cov_risk"] = r_nat
            m["rstar_at_min_cap"] = r_cap
            m["gap_zero_crossing"] = gap_zero_crossing(d, curves)
            m["parity_load_at_r05"] = CP.load_parity(0.05, curves)
            entry["modes"][mode] = m
            print(f"[ok] {name}/{mode}", flush=True)
        res["detectors"][name] = entry

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    L, A = [], None
    A = L.append
    A("# Detector pool — does the EN/VI abstention finding generalise? (task 5)\n")
    A(f"Generated {res['generated']} · reference budget **C = {C_REF:.0%}** · balanced EN/VI traffic · "
      f"thresholds fitted on dev, reported on test · seeds pooled per detector.\n")
    A("\n`min parity capacity` = first review capacity (0.5 %-steps) at which CONRA-Parity's risk gap "
      "reaches 0 *without either language being saturated at full coverage* — i.e. the capacity below "
      "which language parity is structurally unattainable.\n"
      "`parity load @r=5%` = review load exact risk parity needs at a 5 % target residual risk.\n")

    for mode in ("NAT", "MAT", "PAIR"):
        A(f"\n## {mode}\n")
        A("| detector | n/lang | acc EN | acc VI | AUC EN | AUC VI | ECE EN | ECE VI | "
          "GLOBAL Δ (pp) | EQUAL-COV Δ | CONRA Δ | queue VI (GLOBAL) | parity cap. (struct.) | "
          "r* @ min cap | Δ=0 crossing (meas.) | parity load @r=5% | refused (unscored) |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, e in res["detectors"].items():
            m = e["modes"].get(mode)
            if not m:
                continue
            q, a = m["quality"], m["at_c_ref"]
            ref = e.get("refused", {})
            ref_txt = ", ".join(f"{k} {v}/{tot} ({v/tot*100:.1f} %)" for k, (v, tot) in ref.items() if tot) \
                if ref else "—"
            cap, cross, rcap = m["min_parity_capacity"], m.get("gap_zero_crossing"), m.get("rstar_at_min_cap")
            A(f"| {name} | {m['n']['en']} | {q['en']['acc']:.4f} | {q['vi']['acc']:.4f} | "
              f"{q['en']['auc']:.4f} | {q['vi']['auc']:.4f} | {q['en']['ece']:.4f} | {q['vi']['ece']:.4f} | "
              f"**{a['GLOBAL']['risk_gap']*100:+.2f}** | {a['EQUAL-COV']['risk_gap']*100:+.2f} | "
              f"{a['CONRA']['risk_gap']*100:+.2f} | {a['GLOBAL']['queue_vi']:.3f} | "
              f"{'—' if cap is None else f'{cap*100:.1f} %'} | "
              f"{'—' if rcap is None else f'{rcap*100:.1f} %'} | "
              f"{'—' if cross is None else f'{cross*100:.1f} %'} | {m['parity_load_at_r05']*100:.1f} % | "
              f"{ref_txt} |")

    A("\n## Reading\n")
    A("- A finding that only holds for one architecture is an artefact; the table above is the test.\n"
      "- `queue VI` under GLOBAL is the deployable-rule disparity; CONRA's Δ shows how much of it a "
      "risk-parity rule can remove at the same budget, and `min parity cap.` says at what capacity "
      "the removal becomes possible at all.\n"
      "- The LLM moderator matters twice: as an external confidence source, and because its "
      "verbalised probabilities are typically poorly calibrated in the lower-resource language — "
      "the per-language ECE columns are the place to look for that.\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("report ->", args.out, flush=True)
    print("json   ->", args.json, flush=True)


if __name__ == "__main__":
    main()
