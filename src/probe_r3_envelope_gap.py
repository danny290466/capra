#!/usr/bin/env python3
"""Round-3 external review, item 2.2: how far does the running-minimum envelope sit BELOW the curve
the thresholds actually realise?

The manuscript reports the envelope as the object it optimises over (`envelope_curve`), and states
that at the reported operating points the raw curve gives the same disparity (Table B.1). What it
never quantifies is the size of the relaxation: envelope(s) <= raw(s) by construction, and a reviewer
is right that the envelope value at share s need not be attained by any policy that reviews s.

Outputs, per language x seed: max_s [raw(s) - envelope(s)] in pp and where it occurs, the same gap at
the review shares the paper reports (C = 5 %, 10 %, 25 %), and the raw/envelope pair at the peak.
"""
import json

import numpy as np

import sim_operational as S


def raw_curve(conf, err):
    order = np.argsort(-conf, kind="mergesort")
    e = err[order]
    k = np.arange(1, len(e) + 1)
    return np.cumsum(e) / k


def at_share(curve, s):
    n = len(curve)
    k = int(round(n * (1.0 - s)))
    k = min(max(k, 1), n)
    return float(curve[k - 1])


def main():
    out = {}
    for seed in S.SEEDS:
        z = np.load(f"outputs/killtest/seed{seed}/probs.npz")
        for l in ("en", "vi"):
            conf, err = S.conf_err(z[f"{l}_dev_probs"], z[f"{l}_dev_labels"])
            raw = raw_curve(conf, err)
            env = S.envelope_curve(conf, err)
            gap = raw - env                     # >= 0 by construction
            i = int(np.argmax(gap))
            n = len(raw)
            row = {
                "seed": seed, "n": n,
                "max_gap_pp": float(100 * gap.max()),
                "max_gap_review_share": float(1.0 - (i + 1) / n),
                "raw_at_peak_pp": float(100 * raw[i]),
                "env_at_peak_pp": float(100 * env[i]),
                "n_shares_with_gap": int((gap > 1e-12).sum()),
                "frac_shares_with_gap": float((gap > 1e-12).mean()),
                "gap_at_C_pp": 100 * (at_share(raw, S.C) - at_share(env, S.C)),
                "gap_at_2C_pp": 100 * (at_share(raw, 2 * S.C) - at_share(env, 2 * S.C)),
                "gap_at_5C_pp": 100 * (at_share(raw, 5 * S.C) - at_share(env, 5 * S.C)),
            }
            out.setdefault(l, []).append(row)
            print(f"{l} seed {seed}: n={n} max_gap={row['max_gap_pp']:.3f} pp at review share "
                  f"{row['max_gap_review_share']:.3f} (raw {row['raw_at_peak_pp']:.2f} vs env "
                  f"{row['env_at_peak_pp']:.2f}); gaps over {row['frac_shares_with_gap']*100:.1f}% of share grid; "
                  f"gap @C={row['gap_at_C_pp']:.3f} @2C={row['gap_at_2C_pp']:.3f} @5C={row['gap_at_5C_pp']:.3f} pp")

    summary = {}
    for l, rows in out.items():
        summary[l] = {k: float(np.mean([r[k] for r in rows]))
                      for k in ("max_gap_pp", "max_gap_review_share", "gap_at_C_pp")}
        summary[l]["max_gap_pp_worst_seed"] = float(max(r["max_gap_pp"] for r in rows))
    with open("outputs/r3_envelope_gap.json", "w") as fh:
        json.dump({"per_seed": out, "summary": summary}, fh, indent=1)
    print("\nSUMMARY", json.dumps(summary, indent=1))
    print("wrote outputs/r3_envelope_gap.json")


if __name__ == "__main__":
    main()
