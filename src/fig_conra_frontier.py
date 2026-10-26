"""Paper figure for CONRA-Parity — reads outputs/killtest/conra_frontier.json, writes PNG + PDF.

Panel A: risk gap vs review capacity, NAT (deployment-like mix).
Panel B: same for PAIR (same items, EN->VI translated) — the content-controlled contrast.
Panel C: who pays — VI share of the review queue vs capacity.
Panel D: the cost of parity — review load that exact risk parity needs at a fixed target risk,
         with the risk gap a single global threshold leaves when handed that same load.

Layout (review round 2, item #8): ONE figure-level legend under the panels instead of four
per-panel legends, which overlapped each other in A/B and sat on the curves in C; fewer x ticks on
the log axis; taller panels with the same page width, so each panel fills more of the page.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "conra_frontier.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fig_conra_frontier")

COLORS = {"GLOBAL": "#0072B2", "EQUAL-COV": "#E69F00", "CONRA": "#D55E00"}
LABELS = {"GLOBAL": "single global threshold", "EQUAL-COV": "equal coverage (equalized conformal)",
          "CONRA": "CONRA-Parity (risk parity)"}
MODES = [("NAT", "natural EN/VI mix"), ("PAIR", "paired EN->VI (same items)"),
         ("MAT", "prevalence-matched")]
MODE_STYLE = {"NAT": ("-", "o"), "MAT": ("-", "s"), "PAIR": ("--", "^")}
# legend text must stay SHORT: three long labels in one row stretch the tight bounding box wider
# than the axes, and the figure is then scaled down when it is placed at the page width — the panels
# get smaller, which is the opposite of the point of the redesign.
LEG_LABELS = {"GLOBAL": "pooled threshold", "EQUAL-COV": "equalised coverage", "CONRA": "CONRA-Parity"}
MODE_LEG = {"NAT": "NAT · natural mix", "MAT": "MAT · matched", "PAIR": "PAIR · paired"}
# 10 budgets on a log axis crowd their labels into a few millimetres; 5 ticks carry the same reading
XT = [1, 2, 5, 10, 20, 40]

d = json.load(open(JSON, encoding="utf-8"))
B = np.array(d["budgets"]) * 100

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight"})

fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.4))


def axis_labels(a, ylabel):
    a.set_xscale("log")
    a.set_xticks([t for t in XT if t <= B[-1]])
    a.set_xticklabels([f"{t:g}" for t in XT if t <= B[-1]])
    a.set_xlabel("review capacity C (% of traffic reviewable)", labelpad=1.5)
    a.set_ylabel(ylabel, labelpad=1.5)


def gap_panel(a, mode, title):
    m = d["modes"][mode]
    for rule in d["rules"]:
        a.plot(B, np.array(m["risk_gap"][rule]) * 100, "-o", ms=3.2, lw=1.5,
               color=COLORS[rule], label=LABELS[rule])
    a.axhline(0, color="0.35", lw=0.9, ls="-")
    axis_labels(a, "risk gap  r(EN) − r(VI)   (pp)")
    a.set_title(title, pad=5)


gap_panel(ax[0, 0], "NAT", "A — natural English–Vietnamese mix")
gap_panel(ax[0, 1], "PAIR", "B — paired, same items seen in both languages")

# C — who pays for the abstentions: colour = rule, line style = mix (solid NAT, dashed PAIR)
a = ax[1, 0]
for mode in ("NAT", "PAIR"):
    m = d["modes"][mode]
    for rule in ("GLOBAL", "CONRA"):
        ls, mk = MODE_STYLE[mode]
        a.plot(B, np.array(m["queue_vi"][rule]) * 100, ls, marker=mk, ms=3.2, lw=1.4,
               color=COLORS[rule], alpha=1.0 if mode == "NAT" else 0.65)
a.axhline(50, color="0.35", lw=0.9, ls=":")
a.text(B[0], 51.5, "parity with traffic share", fontsize=6.5, color="0.35")
a.set_ylim(45, 103)
axis_labels(a, "share of the review queue that is Vietnamese (%)")
a.set_title("C — who pays for the abstentions", pad=5)

# D — cost of parity
a = ax[1, 1]
for mode in ("NAT", "MAT", "PAIR"):
    if mode not in d["modes"]:
        continue
    ls, mk = MODE_STYLE[mode]
    pc = d["modes"][mode]["parity_cost"]
    r = np.array(pc["risks"]) * 100
    a.plot(r, np.array(pc["load_needed"]) * 100, ls, marker=mk, ms=3.4, lw=1.5)
for lvl in (5, 10, 20):
    a.axhline(lvl, color="#0072B2", lw=0.9, ls="--", alpha=0.55)
    a.text(10.2, lvl + 0.7, f"{lvl} % budget", fontsize=6.4, color="#0072B2", ha="right")
a.set_xlabel("target residual risk r (%)", labelpad=1.5)
a.set_ylabel("review load that exact risk parity needs (%)", labelpad=1.5)
a.set_title("D — the cost of language parity", pad=5)
a.set_ylim(0, 48)

# ONE shared legend for the whole figure: three rules by colour, three mixes by line style/marker
handles = [Line2D([], [], color=COLORS[r], lw=1.6, label=LEG_LABELS[r]) for r in d["rules"]]
handles += [Line2D([], [], color="0.35", lw=1.4, ls=MODE_STYLE[m][0], marker=MODE_STYLE[m][1],
                   ms=3.4, label=MODE_LEG[m]) for m, _ in MODES if m in d["modes"]]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.048), ncol=3,
           frameon=False, handlelength=1.8, columnspacing=1.6, handletextpad=0.6)

fig.tight_layout(h_pad=2.4, w_pad=1.8)
note = ("At r = 5 % exact risk parity needs "
        + " / ".join(
            f"{d['modes'][m]['parity_cost']['load_needed'][d['modes'][m]['parity_cost']['risks'].index(0.05)]*100:.1f}"
            for m in ("NAT", "MAT", "PAIR") if m in d["modes"])
        + " % review load (NAT / MAT / PAIR); handing a single global threshold that same load still leaves a "
        + " / ".join(
            f"{d['modes'][m]['parity_cost']['global_gap_at_that_load'][d['modes'][m]['parity_cost']['risks'].index(0.05)]*100:+.1f}"
            for m in ("NAT", "MAT", "PAIR") if m in d["modes"])
        + " pp language risk gap.")
# The note is the object that sets the figure's WIDTH: unwrapped it runs to ~8.7 in, so the whole
# figure was scaled down to 0.72x when placed at the 6.3 in page width and every panel shrank with it
# (review finding #8, "panels occupy only a small part of the page"). Wrap it inside the axes width.
note = "\n".join(textwrap.wrap(note, 76))
fig.text(0.5, -0.062, note, ha="center", va="top", fontsize=6.6, color="0.30", linespacing=1.35)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}")

img = np.asarray(plt.imread(f"{OUT}.png"))
assert img.size and float(img[..., :3].std()) > 0.01, "refusing to write a blank figure"
print("wrote", OUT + ".png", OUT + ".pdf", "| shape", img.shape, "| std", round(float(img[..., :3].std()), 3))
