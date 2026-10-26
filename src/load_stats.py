"""Bundle every number the manuscript needs into one stats.json (run on the results node).

The manuscript never contains a hand-typed number: content modules read this file, which is assembled
directly from the run artefacts (probs.npz-derived JSONs, the consistency gate, the LLM usage files).

Usage:  python scripts/load_stats.py --proj . --out manuscript/stats.json
"""
import argparse
import glob
import json
import os
from datetime import datetime

# llm_qwen38_flash is NOT in this list: it left 397 EN/VI items unscored (stored as NaN, not as a
# score), so it cannot be printed beside detectors that scored every item - the equal-n convention
# forbids it. Its refusal behaviour is reported separately (review findings R3-05, R3-06).
DETS_ENVI = ["xlmr_base", "xlmr_large", "specialist_pair", "llm_deepseek_flash",
             "llm_gemini31_flashlite"]
# the two monolingual specialists are inputs to the merged detector, not cross-lingual detectors, but
# their training configuration belongs in Table 2 (review finding R2-16)
DETS_TRAIN = ["roberta_en", "phobert_vi"]
# reported (Table 3) but excluded from every comparison: it left items unscored
DETS_REPORT = ["llm_qwen38_flash"]
POOL_ROOTS = {"roberta_en": "outputs/pool/roberta_en", "phobert_vi": "outputs/pool/phobert_vi",
              "xlmr_large": "outputs/pool/xlmr_large", "specialist_pair": "outputs/pool/specialist_pair",
              "llm_deepseek_flash": "outputs/pool/llm_deepseek_flash",
              "llm_gemini31_flashlite": "outputs/pool/llm_gemini31_flashlite",
              "llm_qwen38_flash": "outputs/pool/llm_qwen38_flash"}


def jload(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                              # noqa: BLE001
        return None


def _scored(root):
    """Items the scorer wrote a score for, read from the artefact (the EN/VI pass has no counts block)."""
    import glob as _g
    import numpy as _np
    tot = 0
    for f in sorted(_g.glob(os.path.join(root, "seed*", "probs.npz"))) or [os.path.join(root, "probs.npz")]:
        if not os.path.exists(f):
            continue
        try:
            z = _np.load(f)
        except Exception:                                              # noqa: BLE001
            continue
        for k in [k for k in z.files if k.endswith("_probs")]:
            tot += int(_np.isfinite(z[k][:, 1]).sum())
        for k in [k for k in z.files if k.endswith("_labels")]:
            if not any(x.endswith("_probs") for x in z.files):
                tot += int(len(z[k]))
    return tot


def _unscored(root):
    """Items the scorer did not score, from the mask pool_llm writes (per set sums to the total)."""
    import glob as _g
    import numpy as _np
    tot = 0
    for f in sorted(_g.glob(os.path.join(root, "seed*", "probs.npz"))) or [os.path.join(root, "probs.npz")]:
        if not os.path.exists(f):
            continue
        try:
            z = _np.load(f)
        except Exception:                                              # noqa: BLE001
            continue
        for k in [k for k in z.files if k.endswith("_scored")]:
            tot += int((~z[k].astype(bool)).sum())
        if tot == 0 and len([k for k in z.files if k.endswith("_scored")]) == 0:
            for k in [k for k in z.files if k.endswith("_probs")]:
                tot += int(_np.isnan(z[k][:, 1]).sum())
    return tot


def _dig(obj, *path):
    """Safe read of a nested key, for the digest."""
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def _clean(o):
    """NaN/inf are not valid JSON: they made stats.json unparseable by strict readers (R3-26)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--out", default="manuscript/stats.json")
    args = ap.parse_args()
    P = args.proj
    S = {"generated": datetime.now().isoformat(timespec="seconds"), "proj": P}

    # ---------------------------------------------------------------- EN/VI
    envi = {"pool_compare": jload(f"{P}/outputs/pool/pool_compare.json"),
            "conra": jload(f"{P}/outputs/killtest/conra_parity.json"),
            "frontier": jload(f"{P}/outputs/killtest/conra_frontier.json"),
            "capra": jload(f"{P}/outputs/killtest/capra.json"),
            "baseline": jload(f"{P}/outputs/killtest/baseline_NAT.json"),
            "ablations": jload(f"{P}/outputs/killtest/ablations_NAT.json"),
            "significance": jload(f"{P}/outputs/killtest/significance_NAT.json"),
            "modes": jload(f"{P}/outputs/killtest/modes_ENVI.json"),
            "detectors": {}}
    for d in DETS_ENVI + DETS_TRAIN + DETS_REPORT:
        root = "outputs/killtest" if d == "xlmr_base" else POOL_ROOTS.get(d)
        if not root:
            continue
        n_seeds = len(glob.glob(f"{P}/{root}/seed*/probs.npz")) or (1 if os.path.exists(f"{P}/{root}/probs.npz") else 0)
        # metrics.json is written INSIDE each seed directory, so reading it from the detector root always
        # returned None and Table 2 (training configuration) rebuilt header-only (review finding R3-10)
        metrics = jload(f"{P}/{root}/seed42/metrics.json") or jload(f"{P}/{root}/metrics.json")
        envi["detectors"][d] = {"root": root, "n_seeds": n_seeds,
                                "metrics": metrics,
                                "usage": jload(f"{P}/{root}/usage.json"),
                                # every detector gets its own capra.json: the "Delta CAPRA-parity" column
                                # of Table 4 was CONRA's number for five of six rows because only xlmr_base
                                # had one (review finding R3-22)
                                "capra": jload(f"{P}/{root}/capra.json"),
                                "baseline": jload(f"{P}/{root}/baseline_NAT.json"),
                                "ablations": jload(f"{P}/{root}/ablations_NAT.json"),
                                # single-run moderators: their item-level intervals go into the
                                # significance table (review findings R2-07, N2-08)
                                "significance": jload(f"{P}/{root}/significance_NAT.json"),
                                "unscored_items": _unscored(f"{P}/{root}"),
                                "scored_items": _scored(f"{P}/{root}"),
                                # the merged detector has no metrics of its own: its two components do
                                "metrics_parts": ({c: jload(f"{P}/outputs/pool/{c}/seed42/metrics.json")
                                                   for c in ("roberta_en", "phobert_vi")}
                                                  if d == "specialist_pair" else None)}
    # the EN/VI splits themselves (n + prevalence), so Table 1 can report both benchmarks
    envi["data"] = {}
    for f in sorted(glob.glob(f"{P}/data/killtest/*.jsonl")):
        name = os.path.basename(f).replace(".jsonl", "")
        try:
            rows = [json.loads(l) for l in open(f, encoding="utf-8")]
            envi["data"][name] = {"n": len(rows),
                                  "prevalence": (sum(int(r["label"]) for r in rows) / len(rows))
                                  if rows else None}
        except Exception:                                          # noqa: BLE001
            pass
    S["envi"] = envi

    # ---------------------------------------------------------------- textdetox
    td = {"prep": jload(f"{P}/data/textdetox/summary.json"), "detectors": {}, "llm_usage": {}}
    for f in sorted(glob.glob(f"{P}/outputs/textdetox/*/pair_parity.json")):
        det = os.path.basename(os.path.dirname(f))
        td["detectors"][det] = {"pair_parity": jload(f),
                                "metrics": jload(f"{P}/outputs/textdetox/{det}/seed42/metrics.json"),
                                "baseline": jload(f"{P}/outputs/textdetox/{det}/baseline.json"),
                                "ablations": jload(f"{P}/outputs/textdetox/{det}/ablations.json")}
    for f in sorted(glob.glob(f"{P}/outputs/textdetox/llm_*/usage.json")):
        td["llm_usage"][os.path.basename(os.path.dirname(f))] = jload(f)
    for f in sorted(glob.glob(f"{P}/outputs/textdetox/xlmr_*/capra_en_*_NAT.json")):
        det = os.path.basename(os.path.dirname(f))
        tgt = os.path.basename(f).replace("capra_en_", "").replace("_NAT.json", "")
        td["detectors"].setdefault(det, {}).setdefault("capra", {})[tgt] = jload(f)
    S["textdetox"] = td

    # ---------------------------------------------------------------- gate + inventory
    S["gate"] = jload(f"{P}/outputs/consistency.json")
    # the small versioned digest: the table cells the manuscript was built from, plus a hash of the
    # bundle, so "the numbers the manuscript came from" is a pinned artefact (review finding R3-26)
    digest = {"generated": S["generated"], "proj": P,
              "envi": {"capra_parity_D_test": _dig(envi, "capra", "budgets", "0.05", "cost_blind",
                                                   "parity_first", "test", "D_abs"),
                       "n_detectors": len(envi["detectors"])},
              "textdetox": {"n_detectors": len(td["detectors"])}}
    S["digest"] = digest
    S["inventory"] = {"envi_seed_dirs": len(glob.glob(f"{P}/outputs/killtest/seed*/probs.npz")),
                      "pool_seed_dirs": len(glob.glob(f"{P}/outputs/pool/*/seed*/probs.npz")),
                      "textdetox_seed_dirs": len(glob.glob(f"{P}/outputs/textdetox/*/seed*/probs.npz")),
                      "figures": len(glob.glob(f"{P}/figures/*"))}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_clean(S), f, indent=2)   # NaN is not valid JSON (review R3-26)
    print("wrote", args.out)
    for k, v in S["inventory"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
