"""Paper figure: the CAPRA trade-off front — disparity vs review-queue composition.

Reads a capra.json produced by capra.py and draws, per budget, the achievable (K, D) set with the
two endpoints marked: the parity-first point (min disparity) and the composition-first point
(K = traffic share). Left panel = cost-blind capacity, right panel = cost-aware capacity
(non-English review costs more time), so the two panels show how ignoring review cost inflates the
disparity you can achieve at the same nominal budget.

Usage: python fig_capra_front.py <capra.json> [out_prefix] [--lang-a en --lang-b de]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a.split("=")[0].lstrip("-"): (a.split("=")[1] if "=" in a else True)
         for a in sys.argv[1:] if a.startswith("--")}
JSON = args[0]
OUT = args[1] if len(args) > 1 else os.path.join(os.path.dirname(JSON), "fig_capra_front")
LA = flags.get("lang-a", "en")
LB = flags.get("lang-b", "vi")   # overridden from the JSON pair below

d = json.load(open(JSON, encoding="utf-8"))
k_star = d.get("composition_target", 0.5)
LA, LB = d.get("pair", [LA, LB])          # take the actual pair from the artefact

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight"})

budgets = sorted(d["budgets"].keys(), key=float)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(budgets)))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)

for j, tag in enumerate(("cost_blind", "cost_aware")):
    a = ax[j]
    for c, C in zip(colors, budgets):
        front = d["fronts"][C][tag]
        K = np.array([p["K"] for p in front]) * 100
        D = np.array([p["D_abs"] for p in front]) * 100
        ok = np.array([not p["clipped"] for p in front])
        a.plot(K[ok], D[ok], "-", lw=1.5, color=c)
        m = d["budgets"][C][tag]
        pf, cf = m["parity_first"], m["composition_first"]
        a.plot(pf["K"] * 100, pf["D_abs"] * 100, "o", ms=5, color=c, mec="white", mew=0.7, zorder=5)
        a.plot(cf["K"] * 100, cf["D_abs"] * 100, "s", ms=4.5, color=c, mec="white", mew=0.7, zorder=5)
        if j == 0:
            a.plot([], [], "-", lw=1.5, color=c, label=f"C = {float(C)*100:.0f} %")
    a.axvline(k_star * 100, color="0.3", lw=0.9, ls="--")
    a.text(k_star * 100 + 1.5, a.get_ylim()[1] * 0.96, "queue = traffic share", fontsize=6.3, color="0.3")
    a.set_xlabel(f"item composition: share of the review queue from {LB}   $K_Q$ (%)")
    a.set_title("A — cost-blind capacity" if j == 0 else
                f"B — cost-aware capacity ({LB} reviews at {d['costs'][LB]}× time)")
ax[0].plot([], [], "o", ms=5, color="0.4", label="parity-first")
ax[0].plot([], [], "s", ms=4.5, color="0.4", label="composition-first (K*)")
# outside the axes: inside, it covered the achievable-set curves
ax[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.28), ncol=2, frameon=False,
             fontsize=7, title="budget / objective", title_fontsize=7)
ax[0].set_ylabel(f"disparity  |r$_{{{LA}}}$ − r$_{{{LB}}}$|  (pp)")
fig.suptitle(f"CAPRA achievable set — {d['pair'][0]} vs {d['pair'][1]}, mode {d.get('mode', 'NAT')}",
             fontsize=9.5, y=1.04)
fig.text(0.5, -0.06,
         f"$K_Q$ = 100 % means the entire review queue is {LB} content; the parity-first point sits where the "
         f"disparity vanishes, the composition-first point where the queue mirrors the traffic mix.",
         ha="center", va="top", fontsize=6.4, color="0.30")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}")
print("wrote", OUT + ".png", OUT + ".pdf")
img = plt.imread(f"{OUT}.png")
print("png shape", img.shape, "std", float(np.asarray(img[..., :3]).std()))
