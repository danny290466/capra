"""Figure: the L-group allocation on the seven-language benchmark (Appendix D).

Panel A: the trade-off frontier — the smallest worst pairwise residual-risk gap attainable when the queue
composition may deviate from the traffic shares by at most delta (design split). Panel B: the queue
composition of the L-group parity-first allocation (test split) against the traffic share every group
would get from a queue that mirrors traffic.

Reads the JSON written by capra_lgroup.py; CPU only. usage:
  python fig_lgroup.py --json-a outputs/textdetox/capra_lgroup_xlmr_base.json \
                       --json-b outputs/textdetox/capra_lgroup_xlmr_large.json --out figures/fig_lgroup
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.7,
                     "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-a", required=True)
    ap.add_argument("--json-b", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-a", default="XLM-R base")
    ap.add_argument("--label-b", default="XLM-R large")
    args = ap.parse_args()

    A = json.load(open(args.json_a))
    B = json.load(open(args.json_b)) if args.json_b else None
    langs = A["langs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.20, wspace=0.28)

    # ---- panel A: frontier
    for J, lab, col, mk in ((A, args.label_a, "#1f4e79", "o"), (B, args.label_b, "#a33", "s")):
        if J is None:
            continue
        d = [t["delta"] for t in J["tradeoff_frontier"]]
        g = [t["design_max_gap"] * 100 for t in J["tradeoff_frontier"]]
        ax1.plot(d, g, marker=mk, ms=3.2, lw=1.3, color=col, label=lab)
        ax1.annotate(f'{g[0]:.1f}', (d[0], g[0]), textcoords="offset points", xytext=(4, 4),
                     fontsize=7, color=col)
        ax1.annotate(f'{g[-1]:.1f}', (d[-1], g[-1]), textcoords="offset points", xytext=(-16, 4),
                     fontsize=7, color=col)
    ax1.set_xlabel("composition-deviation ceiling $\\delta$")
    ax1.set_ylabel("worst pairwise risk gap (pp)")
    ax1.set_title("A. The trade-off is a surface, not a point")
    ax1.set_xlim(-0.05, 1.08)
    lo = min(min(t["design_max_gap"] for t in J["tradeoff_frontier"]) for J in (A, B) if J)
    hi = max(max(t["design_max_gap"] for t in J["tradeoff_frontier"]) for J in (A, B) if J)
    ax1.set_ylim(lo * 100 - 1.2, hi * 100 + 1.2)
    ax1.grid(alpha=0.25, lw=0.5)
    ax1.legend(loc="upper right", frameon=False)

    # ---- panel B: queue composition at the parity-first point
    x = np.arange(len(langs))
    pf = A["parity_first"]["test"]["K"]
    po = A["pooled"]["test"]["K"]
    ax2.bar(x - 0.19, po, width=0.36, color="#9bb7d4", label="queue mirrors traffic (pooled rule)")
    ax2.bar(x + 0.19, pf, width=0.36, color="#1f4e79", label="L-group parity-first (test)")
    ax2.axhline(1.0 / len(langs), color="#555", lw=0.9, ls="--")
    ax2.annotate("traffic share 1/7", (len(langs) - 0.45, 1.0 / len(langs)), textcoords="offset points",
                 xytext=(-72, 4), fontsize=7, color="#555")
    ax2.set_xticks(x)
    ax2.set_xticklabels(langs)
    ax2.set_ylabel("share of the review queue")
    ax2.set_title("B. What parity-first costs in queue composition")
    ax2.set_ylim(0, max(max(pf), max(po)) * 1.22)
    ax2.grid(alpha=0.25, lw=0.5, axis="y")
    ax2.legend(loc="upper left", frameon=False)

    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=200)
    print("wrote", args.out + ".pdf", "and .png")
    # a figure whose every value is NaN still saves: assert the panels carry real data
    assert all(np.isfinite(v) for v in pf) and len(pf) == len(langs)
    assert all(t["design_max_gap"] > 0 for t in A["tradeoff_frontier"])


if __name__ == "__main__":
    main()
