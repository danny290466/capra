#!/usr/bin/env python3
"""Round-3 external review, item 2.2 (follow-up) — does the §5.6 operational block simulate the
manuscript's own parity-first policy, or the artefact of a mis-oriented envelope?

`sim_operational.envelope_curve` applies `np.minimum.accumulate` to the ascending-k array, i.e. the
running minimum runs from the HIGH-review end. Because the top-1 kept item is almost always correct,
that envelope is identically ~0 at every review share, so `sweep_parity` (which minimises |r_a - r_b|)
ties on every alpha and returns the first grid point, alpha = 0 — "review no English at all".

The main pipeline uses the other convention (`capra.py:69`: np.minimum.accumulate(risk[::-1])[::-1]),
which is a running minimum from the LOW-review end and reproduces the published Table 8 values.

This probe prints, per seed: the policy thresholds `sim_operational.policies()` returns under both
conventions, the design/realised disparity of each, and the published parity-first shares read out of
the main pipeline's own artefact, so the difference is a number rather than a suspicion.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sim"))   # src/sim/sim_operational.py
sys.path.insert(0, HERE)                        # work from any cwd
import sim_operational as S          # noqa: E402


def env_main_pipeline(conf, err):
    """The convention the rest of the paper uses (capra.py:69)."""
    order = np.argsort(-conf, kind="mergesort")
    e = err[order]
    k = np.arange(1, len(e) + 1, dtype=float)
    risk = np.cumsum(e) / k
    return np.minimum.accumulate(risk[::-1])[::-1]


def short(t):
    return {k: (round(v["en"], 4), round(v["vi"], 4)) for k, v in t.items()}


def main():
    published = None
    try:
        cj = json.load(open("outputs/killtest/capra.json"))
        fronts = cj.get("fronts")
        if isinstance(fronts, dict):
            b = fronts.get("0.05") or fronts.get("0.050") or list(fronts.values())[0]
            published = {"keys": list(b)[:8] if isinstance(b, dict) else type(b).__name__,
                         "sample": {k: (str(b[k])[:160] if not isinstance(b[k], list) else f"list[{len(b[k])}]")
                                    for k in (list(b)[:8] if isinstance(b, dict) else [])}}
    except Exception as exc:                                     # noqa: BLE001
        published = {"error": str(exc)}
    print("capra.json front structure:", json.dumps(published, indent=1)[:600])

    rows = {}
    for seed in S.SEEDS:
        z = np.load(f"outputs/killtest/seed{seed}/probs.npz")
        cal, curves = {}, {}
        for l in ("en", "vi"):
            cd, ed = S.conf_err(z[f"{l}_dev_probs"], z[f"{l}_dev_labels"])
            cal[l] = {"conf": cd, "err": ed}
            curves[l] = {"main": env_main_pipeline(cd, ed), "buggy": S.envelope_curve(cd, ed)}
        out = {}
        for conv in ("main", "buggy"):
            c = {l: curves[l][conv] for l in ("en", "vi")}
            taus = S.policies(cal, c)
            out[conv] = short(taus)
            pf = taus["CAPRA-parity"]
            out[conv]["design_D_parity_pp"] = round(100 * abs(
                S.risk_at(c["en"], len(cal["en"]["conf"]), pf["en"]) -
                S.risk_at(c["vi"], len(cal["vi"]["conf"]), pf["vi"])), 4)
            out[conv]["curve_at_parity_en_pp"] = round(
                100 * S.risk_at(c["en"], len(cal["en"]["conf"]), pf["en"]), 4)
            out[conv]["curve_at_parity_vi_pp"] = round(
                100 * S.risk_at(c["vi"], len(cal["vi"]["conf"]), pf["vi"]), 4)
        rows[seed] = out
        print(f"seed {seed}")
        for conv in ("main", "buggy"):
            t = rows[seed][conv]
            print(f"  {conv:6s} GLOBAL={t['GLOBAL']} EQUAL={t['EQUAL-COV']} "
                  f"PARITY={t['CAPRA-parity']} COMP={t['CAPRA-composition']} "
                  f"| curve at parity: en {t['curve_at_parity_en_pp']} pp, vi {t['curve_at_parity_vi_pp']} pp "
                  f"-> design D {t['design_D_parity_pp']} pp")

    same = all(rows[s]["main"]["CAPRA-parity"] == rows[s]["buggy"]["CAPRA-parity"] for s in S.SEEDS)
    print(f"\nparity-first shares identical under both conventions: {same}")
    with open("outputs/r3_envelope_convention.json", "w") as fh:
        json.dump({"per_seed": rows, "parity_identical": same}, fh, indent=1)
    print("wrote outputs/r3_envelope_convention.json")


if __name__ == "__main__":
    main()
