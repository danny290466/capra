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
    # Bars are the PAIRED differences Section 5.6 quotes (+x.xx +- s, 95 % empirical interval):
    # fixed and capped are routed from the same rng draw per (seed, schedule), so the paired
    # difference is the right estimator and these are the same numbers the text prints -- unlike
    # the difference of two marginal means, which drifts by up to 0.1 pp against the paired value.
    pair = agg["_paired_capped_minus_fixed"]
    d_risk = [pair[n]["kept_risk_pp"]["mean"] for n in names]
    e_risk = [[pair[n]["kept_risk_pp"]["mean"] - pair[n]["kept_risk_pp"]["ci"][0] for n in names],
              [pair[n]["kept_risk_pp"]["ci"][1] - pair[n]["kept_risk_pp"]["mean"] for n in names]]
    d_disp = [pair[n]["disparity_pp"]["mean"] for n in names]
    e_disp = [[pair[n]["disparity_pp"]["mean"] - pair[n]["disparity_pp"]["ci"][0] for n in names],
              [pair[n]["disparity_pp"]["ci"][1] - pair[n]["disparity_pp"]["mean"] for n in names]]
    print("paired price of enforcement (risk pp):", np.round(d_risk, 3).tolist())
    print("paired price of enforcement (disp pp):", np.round(d_disp, 3).tolist())
    print("paired sd (risk pp):", np.round([pair[n]["kept_risk_pp"]["sd"] for n in names], 3).tolist())
    print("paired sd (disp pp):", np.round([pair[n]["disparity_pp"]["sd"] for n in names], 3).tolist())

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
    ax2.bar(xs - w / 2, d_risk, w, color=BLUE, label="kept-set risk",
            yerr=e_risk, capsize=2.0, error_kw=dict(lw=0.8, ecolor="0.35"))
    ax2.bar(xs + w / 2, d_disp, w, color=ORANGE, label="disparity",
            yerr=e_disp, capsize=2.0, error_kw=dict(lw=0.8, ecolor="0.35"))
    ax2.axhline(0, color="0.6", lw=0.8)
    top = max(max(np.asarray(d_risk) + np.asarray(e_risk[1])),
              max(np.asarray(d_disp) + np.asarray(e_disp[1])))
    bot = min(min(np.asarray(d_risk) - np.asarray(e_risk[0])),
              min(np.asarray(d_disp) - np.asarray(e_disp[0])), 0.0)
    pad = 0.08 * max(top - bot, 0.4)
    # labels are aligned at the top of the TALLER series in each group, so a label never sits over
    # the neighbouring bar (round-3 review: '+0.38' landed inside the +0.73 bar)
    for i in range(len(names)):
        y_lab = max(d_risk[i] + e_risk[1][i], d_disp[i] + e_disp[1][i]) + 0.4 * pad
        ax2.text(xs[i] - w / 2, y_lab, f"{d_risk[i]:+.2f}", ha="center", va="bottom", fontsize=7.6)
        ax2.text(xs[i] + w / 2, y_lab, f"{d_disp[i]:+.2f}", ha="center", va="bottom", fontsize=7.6)
    ax2.set_ylim(bot - 1.6 * pad, top + 5.4 * pad)
    ax2.set_xticks(xs, ["pooled", "equalised\ncoverage", "parity-\nfirst"])
    ax2.set_ylabel("change from frozen thresholds (pp)")
    # legend sits inside the axes at the top, in the headroom the label row leaves free; outside the
    # frame it collided with the value labels and the title (round-3 review, item 3.2)
    ax2.legend(frameon=False, loc="upper center", ncol=2, fontsize=8.0,
               handlelength=1.4, columnspacing=1.2)
    ax2.set_title("B. Price of enforcing the budget")
    fig.tight_layout(w_pad=1.6)

    # machine check of the panel-B layout, since the figure is only inspected as a rendered PDF
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    leg = ax2.get_legend().get_window_extent(rend)
    collisions = []
    for t in ax2.texts:
        if t.get_text().strip() and t.get_window_extent(rend).overlaps(leg):
            collisions.append(f"label '{t.get_text()}' vs legend")
    for p in ax2.patches:
        if p.get_window_extent(rend).overlaps(leg):
            collisions.append(f"bar {p.get_height():+.2f} vs legend")
    for t in ax2.texts:
        if not t.get_text().strip():
            continue
        for p in ax2.patches:
            if t.get_window_extent(rend).overlaps(p.get_window_extent(rend)):
                collisions.append(f"label '{t.get_text()}' vs bar {p.get_height():+.2f}")
    print("panel B layout collisions:", collisions if collisions else "none")
    for t in ax2.texts:
        y = t.get_position()[1]
        assert 0 <= (y - ax2.get_ylim()[0]) / (ax2.get_ylim()[1] - ax2.get_ylim()[0]) <= 1.02, \
            f"label {t.get_text()} sits outside the axis: y={y}"
    fig.savefig(args.out + ".png")
    fig.savefig(args.out + ".pdf")
    print("wrote", args.out + ".png", args.out + ".pdf")


if __name__ == "__main__":
    main()
