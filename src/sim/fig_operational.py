#!/usr/bin/env python3
"""Figure 8 for CAPRA §5.6: load per review window with and without the per-window cap, and the
price of enforcement by policy.

Panel A is one arrival schedule (seed 42, strong-drift settings, 12 windows of 1 000 items) for the
parity-first policy; panel B reads the aggregated artefact of the 100-schedule x 5-seed run so the
bars match Table 15. Run on the frozen artefacts, next to sim_operational.py.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_operational as S                          # noqa: E402

BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#D55E00", "#009E73", "#6E6E6E", "#CC3311"
plt.rcParams.update({"font.size": 10.0, "axes.titlesize": 10.5, "axes.labelsize": 10.0,
                     "legend.fontsize": 8.6, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
                     "figure.dpi": 300, "savefig.bbox": "tight"})


def load_seed(seed="42"):
    z = np.load(f"outputs/killtest/seed{seed}/probs.npz")
    data = {}
    for l in ("en", "vi"):
        cd, ed = S.conf_err(z[f"{l}_dev_probs"], z[f"{l}_dev_labels"])
        ct, et = S.conf_err(z[f"{l}_test_probs"], z[f"{l}_test_labels"])
        data[l] = {"conf": ct, "err": et, "curve": S.envelope_curve(cd, ed),
                   "cal": {"conf": cd, "err": ed}}
    cal = {l: data[l]["cal"] for l in ("en", "vi")}
    curves = {l: data[l]["curve"] for l in ("en", "vi")}
    return data, S.policies(cal, curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="outputs/operational_sim_A_strong.json")
    ap.add_argument("--out", default="figures/fig_operational")
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--window", type=int, default=1000)
    args = ap.parse_args()

    data, taus = load_seed()
    S.N_WINDOW, S.N_WINDOWS = args.window, args.windows
    rng = np.random.default_rng(7000)
    mix = S.schedule_mix(rng, args.windows, 0.18, 0.25)

    t = taus["CAPRA-parity"]
    series = {}
    for regime in ("fixed", "capped"):
        rng2 = np.random.default_rng(7000)
        lang, conf, err, wl, rev = S.run_stream(data, t, mix, rng2, regime, args.window)
        series[regime] = wl
    print("per-window load (parity-first) fixed :", np.round(series["fixed"], 4).tolist())
    print("per-window load (parity-first) capped:", np.round(series["capped"], 4).tolist())
    print("mix (w_vi)                          :", np.round(mix, 3).tolist())

    agg = json.load(open(args.agg))
    names = ["GLOBAL", "EQUAL-COV", "CAPRA-parity"]
    # deltas are the differences of the values Table 15 prints, so the bars and the manuscript agree
    d_risk = [round(agg[f"{n}|capped"]["kept_risk_pp"]["mean"], 2)
              - round(agg[f"{n}|fixed"]["kept_risk_pp"]["mean"], 2) for n in names]
    d_disp = [round(agg[f"{n}|capped"]["disparity_pp"]["mean"], 2)
              - round(agg[f"{n}|fixed"]["disparity_pp"]["mean"], 2) for n in names]
    print("price of enforcement (risk pp):", np.round(d_risk, 3).tolist())
    print("price of enforcement (disp pp):", np.round(d_disp, 3).tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(1, args.windows + 1)
    ax1.axhline(100 * S.C, color=GREY, lw=0.9, ls="--")
    ax1.plot(x, 100 * series["fixed"], "-o", ms=3.2, lw=1.4, color=ORANGE, label="frozen thresholds")
    ax1.plot(x, 100 * series["capped"], "-s", ms=3.0, lw=1.4, color=BLUE, label="per-window cap")
    # the label sits in the empty right margin, with a white backing so the dashed rule does not
    # strike through it and the per-window cap line (flat at exactly 5 %) stays readable
    ax1.text(args.windows + 2.4, 100 * S.C, "C = 5 %", color=GREY, fontsize=8.4, va="center",
             ha="right", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax1.set_xlabel("review window")
    ax1.set_ylabel("share of traffic reviewed (%)")
    ax1.set_ylim(0, max(10.5, 105 * max(series["fixed"].max(), S.C)))
    ax1.set_xlim(0.5, args.windows + 2.6)
    ax1b = ax1.twinx()
    ax1b.plot(x, mix, color=GREY, lw=1.1, ls=":", label="Vietnamese share of traffic")
    ax1b.set_ylabel("Vietnamese share of traffic", color=GREY)
    ax1b.tick_params(axis="y", colors=GREY)
    ax1b.set_ylim(0, 1)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    ax1.set_title("A. Load per window (parity-first policy)")

    w = 0.36
    xs = np.arange(len(names))
    ax2.bar(xs - w / 2, d_risk, w, color=BLUE, label="kept-set risk")
    ax2.bar(xs + w / 2, d_disp, w, color=ORANGE, label="disparity")
    ax2.axhline(0, color="0.6", lw=0.8)
    for xi, v in zip(xs - w / 2, d_risk):
        ax2.text(xi, v + 0.02, f"{v:+.2f}", ha="center", fontsize=8.0)
    for xi, v in zip(xs + w / 2, d_disp):
        ax2.text(xi, v + (0.02 if v >= 0 else -0.10), f"{v:+.2f}", ha="center", fontsize=8.0)
    ax2.set_xticks(xs, ["pooled", "equalised\ncoverage", "parity-\nfirst"])
    ax2.set_ylabel("change from frozen thresholds (pp)")
    ax2.legend(frameon=False, loc="upper right")
    ax2.set_title("B. Price of enforcing the budget")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(args.out + ".png")
    fig.savefig(args.out + ".pdf")
    print("wrote", args.out + ".png", args.out + ".pdf")


if __name__ == "__main__":
    main()
