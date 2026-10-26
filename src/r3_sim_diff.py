#!/usr/bin/env python3
"""Round-3 review, item 2.2 housekeeping: after fixing `sim_operational.envelope_curve` to the
paper's convention, prove the §5.6 numbers did NOT move (the two conventions select the same
thresholds on these data) by diffing the regenerated artefacts against the pre-fix copies.

Usage:  python src/r3_sim_diff.py                    (cwd = project root)
"""
import json
import sys

FILES = ("A_strong", "B_mild", "D_static")

def main():
    bad = 0
    for f in FILES:
        old = json.load(open(f"outputs/operational_sim_{f}_pre_r3.json"))
        new = json.load(open(f"outputs/operational_sim_{f}.json"))
        diffs = []
        for key in old:
            if key.startswith("_"):
                continue
            for metric, blk in old[key].items():
                if abs(blk["mean"] - new[key][metric]["mean"]) > 1e-12:
                    diffs.append((key, metric, blk["mean"], new[key][metric]["mean"]))
        po, pn = old.get("_paired_capped_minus_fixed", {}), new.get("_paired_capped_minus_fixed", {})
        for name in po:
            for metric, blk in po[name].items():
                if metric == "paired_runs":
                    continue
                if abs(blk["mean"] - pn[name][metric]["mean"]) > 1e-12:
                    diffs.append((f"paired:{name}", metric, blk["mean"], pn[name][metric]["mean"]))
        n_rows = sum(len(old[k]) for k in old if not k.startswith("_"))
        print(f"{f}: {n_rows} metric rows + paired block -> {len(diffs)} differ")
        for d in diffs[:8]:
            print(f"    {d[0]} | {d[1]}: {d[2]:.6f} -> {d[3]:.6f}")
        bad += len(diffs)
    print(f"\nTOTAL differing metrics after the envelope fix: {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
