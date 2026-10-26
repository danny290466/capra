#!/usr/bin/env python3
"""Envelope conventions on the release's own per-item scores (review round 2, item 2.4).

The manuscript prints two maxima for the monotone repair of a calibration risk curve: one for the
isotonic fit against the raw step function and one for the running-minimum envelope against the raw
function. Neither had an artefact in the release and neither stated its domain, so this script
recomputes both from `outputs/pool/*/seed*/probs.npz` and reports them per domain:

  all shares      every review share the calibration split can realise, 0 < s < 1
  operating band  the shares within [C, 5C] of the reported budget, where the papers read curves

Definitions follow Appendix B: sort the calibration items by descending confidence q = max(p, 1-p),
keep the top k, s = 1 - k/n, r(s) = mean error of the kept top-k. The envelope is the running minimum
from the low-review end; the isotonic fit is a non-increasing pool-adjacent-violators fit of r(s).

    /opt/homebrew/bin/python3.14 src/tools/envelope_convention_report.py --json outputs/r3_envelope_iso.json
"""
import argparse
import glob
import json
import os

import numpy as np


def curve(probs, labels):
    """r(s) for the raw step curve, exactly as the paper defines it (sim_operational.conf_err):
    confidence q = max(p, 1-p) and error = the decision p >= 0.5 disagrees with the label."""
    p1 = probs[:, 1]
    q = np.maximum(p1, 1.0 - p1)
    err = ((p1 >= 0.5).astype(int) != labels).astype(float)
    order = np.argsort(-q, kind="mergesort")
    err = err[order]                     # sort by descending confidence first
    n = len(err)
    k = np.arange(1, n + 1)
    r = np.cumsum(err) / k              # mean error of the top-k kept items
    s = 1.0 - k / n                     # share reviewed when top-k are kept
    return s, r


def envelope(s, r):
    """Running minimum from the low-review end: as s grows, the best risk seen so far."""
    return np.minimum.accumulate(r)


def isotonic_decreasing(s, r):
    """PAVA fit of r as a non-increasing function of s (same weight per grid point)."""
    # work in ascending s, y must be non-increasing -> fit -r as non-decreasing
    y = -np.asarray(r, float)
    w = np.ones_like(y)
    lvl, wt = [], []
    for yi, wi in zip(y, w):
        lvl.append(yi)
        wt.append(wi)
        while len(lvl) > 1 and lvl[-2] > lvl[-1]:
            v2, w2 = lvl.pop(), wt.pop()
            v1, w1 = lvl.pop(), wt.pop()
            tw = w1 + w2
            lvl.append((v1 * w1 + v2 * w2) / tw)
            wt.append(tw)
    out = np.empty_like(y)
    pos = 0
    for v, ww in zip(lvl, wt):
        span = int(round(ww))
        out[pos:pos + span] = v
        pos += span
    return -out


def stats(s, r, fit, lo, hi):
    mask = (s > lo) & (s < hi)
    if not mask.any():
        return None
    d = np.abs(fit - r)[mask] * 100.0
    j = int(np.argmax(d))
    return float(d[j]), float(s[mask][j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/pool/xlmr_base")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "seed*"))):
        f = os.path.join(d, "probs.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        seed = os.path.basename(d).replace("seed", "")
        rec = {"seed": seed}
        for lang in ("en", "vi"):
            s, r = curve(z[f"{lang}_dev_probs"], z[f"{lang}_dev_labels"])
            env = np.minimum.accumulate(r[::-1])[::-1]   # running min from the low-review end, as the paper does
            iso = isotonic_decreasing(s, r)
            rec[lang] = {
                "iso_vs_raw_all": stats(s, r, iso, 0.0, 1.0),
                "env_vs_raw_all": stats(s, r, env, 0.0, 1.0),
                "iso_vs_raw_band": stats(s, r, iso, args.budget * 0.99, args.budget * 5.01),
                "env_vs_raw_band": stats(s, r, env, args.budget * 0.99, args.budget * 5.01),
            }
        rows.append(rec)

    summary = {}
    for lang in ("en", "vi"):
        for key in ("iso_vs_raw_all", "env_vs_raw_all", "iso_vs_raw_band", "env_vs_raw_band"):
            vals = [r[lang][key][0] for r in rows if r[lang][key]]
            summary[f"{lang}:{key}"] = round(max(vals), 3) if vals else None

    out = {"root": args.root, "budget": args.budget, "per_seed": rows, "summary": summary}
    print(json.dumps(summary, indent=1))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print("json ->", args.json)


if __name__ == "__main__":
    main()
