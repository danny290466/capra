#!/usr/bin/env python3
"""Four additional figures for the CAPRA manuscript, in the styles used by the reference papers:
system architecture (useraware Fig 1), confusion matrices (keywordbias Fig 5), a detector-by-language
heatmap (keywordbias Fig 4) and an ablation panel (useraware Figs 3-4).

All numbers come from the frozen artefacts; the confusion matrices re-derive the decision threshold on
the calibration split (the same rule the training used: the dev threshold with the best F1), so the
figure is reproducible from probs.npz alone.

Every figure asserts that it has finite data before it is written. v14 shipped a heatmap whose twelve
cells were all dashes and an ablation panel with no points at all, because a wrong key lookup produced
an all-NaN array and matplotlib happily drew an empty figure (review findings A2 and A3).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 10.0, "axes.titlesize": 10.5, "axes.labelsize": 10.0,
                     "legend.fontsize": 9.0, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6E6E6E"
OUT = "figures"
DET = "outputs/killtest"
CAL_ORDER = ["250", "500", "1000", "2000", "full"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def num(v):
    """A value from an artefact cell as a float, or NaN when the cell is missing/censored."""
    return float(v) if isinstance(v, (int, float)) else np.nan


def require_finite(name, **arrays):
    """Refuse to write a figure whose every value is missing: that is a bug, not a result."""
    for label, a in arrays.items():
        arr = np.asarray(a, dtype=float)
        if arr.size and not np.isfinite(arr).any():
            raise RuntimeError(f"{name}: '{label}' is entirely missing — refusing to write a blank "
                               f"figure. Check the artefact keys the panel reads.")


# ---------------------------------------------------------------- 1. system architecture
# MOVED TO DRAWIO (2026-09-24). Pipeline and schematic figures are always drawn in draw.io, never with
# matplotlib: `scripts/build_fig_architecture_drawio.py` builds figures/fig_architecture.drawio and
# exports fig_architecture.{png,pdf,svg}. The matplotlib version that used to live here produced a
# second, conflicting PNG for the same figure, so it was removed. Data figures (curves, matrices,
# heatmaps, panels) stay in this file.


# ---------------------------------------------------------------- 2. confusion matrices
def threshold_on_dev(conf, err):
    """Best-F1 threshold on a calibration split (the rule the trainers used)."""
    order = np.argsort(conf)
    c, e = conf[order], err[order]
    best, bt = -1.0, 0.5
    for t in np.unique(c):
        dec = c >= t
        tp = ((1 - e)[dec]).sum(); fp = (e[dec]).sum(); fn = ((1 - e)[~dec]).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best:
            best, bt = f1, float(t)
    return bt


def fig_confusion():
    seeds = sorted(glob.glob(os.path.join(DET, "seed*", "probs.npz")))
    if not seeds:
        raise RuntimeError("confusion: no probs.npz under " + DET)
    z = np.load(seeds[0])
    langs = [l for l in ("en", "vi") if f"{l}_dev_labels" in z.files and f"{l}_test_labels" in z.files]
    fig, axes = plt.subplots(1, len(langs), figsize=(7.2, 2.9))
    # Room for the panel titles (which now carry the threshold) and for the tick labels: the review's
    # complaint was a label running into its neighbour and off the page edge, so the margins are set
    # explicitly here instead of relying on the default layout.
    fig.subplots_adjust(top=0.80, bottom=0.24, left=0.075, right=0.90, wspace=0.42)
    axes = np.atleast_1d(axes)
    for ax, l in zip(axes, langs):
        conf_d, err_d = z[f"{l}_dev_probs"][:, 1], 1 - z[f"{l}_dev_labels"]
        tau = threshold_on_dev(conf_d, err_d)          # F1-optimal threshold in p(toxic) space
        conf_t, err_t = z[f"{l}_test_probs"][:, 1], 1 - z[f"{l}_test_labels"]
        dec = conf_t >= tau
        # ROWS ARE THE GROUND-TRUTH CLASSES (clean, toxic), COLUMNS THE DECISION (~dec = kept,
        # dec = flagged). The previous version built the array as [decision][class] and then
        # labelled the axes the other way round, so the printed matrix was the transpose of its
        # own caption and its class totals contradicted Table 2 (external review, 2026-10-13).
        cm = np.array([[(~dec & (err_t == 1)).sum(), (dec & (err_t == 1)).sum()],
                       [(~dec & (err_t == 0)).sum(), (dec & (err_t == 0)).sum()]])
        # Guard the class axis against the same class of bug: the toxic row must reproduce the
        # label prevalence, otherwise the two axes are the wrong way round again.
        implied, true_prev = cm[1].sum() / cm.sum(), float((err_t == 0).mean())
        assert abs(implied - true_prev) < 1e-9, (l, implied, true_prev)
        frac = cm / cm.sum()
        require_finite("fig_confusion", share=frac)
        im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=frac.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{frac[i, j]*100:.1f}%\n({cm[i, j]:,})", ha="center", va="center",
                        fontsize=9.0, color="white" if frac[i, j] > 0.55 * frac.max() else "0.15")
        ax.set_xticks([0, 1], ["Predicted\nclean", "Predicted\ntoxic"], fontsize=9.0)
        ax.set_yticks([0, 1], ["clean", "toxic"], fontsize=9.0)
        # The threshold and the state names both used to sit in one long x-label per panel, which ran the
        # two panels' labels into each other and past the page edge in the submitted PDF (round-2 external
        # review, item 5). The threshold now rides in the panel title, the axis is a short fixed label, and
        # the tick names say which axis is predicted (kept/ flagged belong to Section 3.1's reviewed /
        # auto-decided pair and are not the same partition). The caption keeps the threshold as well.
        ax.set_xlabel("Detector decision", fontsize=9.0)
        ax.set_title(f"{'English' if l == 'en' else 'Vietnamese'} — XLM-R base, seed 42, τ = {tau:.3f}",
                     fontsize=9.5)
        ax.grid(False)
    fig.colorbar(im, ax=axes[-1], shrink=0.75, label="share of test items")
    save(fig, "fig_confusion")


# ---------------------------------------------------------------- 3. detector x language heatmap
def fig_heatmap():
    src = {}
    for det, path in (("XLM-R base", "outputs/textdetox/xlmr_base/pair_parity.json"),
                      ("XLM-R large", "outputs/textdetox/xlmr_large/pair_parity.json"),
                      ("DeepSeek-flash (moderator)", "outputs/textdetox/llm_deepseek_flash/pair_parity.json")):
        if os.path.exists(path):
            src[det] = json.load(open(path, encoding="utf-8"))
    if not src:
        raise RuntimeError("heatmap: no pair_parity.json artefacts found")
    targets = ["de", "uk", "ru", "ar", "hi", "zh"]

    def capacity(node):
        """Parity capacity for one target, read exactly as Table 6 reads it (first seed).

        The artefact stores `min_parity_capacity` as a LIST of per-seed values with None for the
        censored seeds (e.g. [0.125, 0.0359, 0.0354]); Table 6 prints element [0], so the figures
        print the same element. v14 tested `isinstance(v, (int, float))`, which is False for a list,
        so every cell silently became NaN and the heatmap shipped blank.
        """
        if not isinstance(node, dict):
            return np.nan
        for key in ("min_parity_capacity", "parity_capacity"):
            v = node.get(key)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, (list, tuple)) and v:
                return num(v[0])            # seed 42, the column Table 6 prints; None = censored
        for m in node.values():             # nested per-mode blocks, older artefacts
            if isinstance(m, dict):
                got = capacity(m)
                if np.isfinite(got):
                    return got
        return np.nan

    def pick(d):
        out = {}
        for t in targets:
            node = d
            for step in ("pairs", t):
                node = (node or {}).get(step) if isinstance(node, dict) else None
            got = capacity(node)
            if np.isfinite(got):
                out[t] = got
        return out

    grid = [[pick(d).get(t, np.nan) * 100 for t in targets] for d in src.values()]
    M = np.array(grid, dtype=float)
    require_finite("fig_heatmap", grid=M)
    fig, ax = plt.subplots(figsize=(6.3, 2.4 + 0.3 * len(src)))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="OrRd", aspect="auto", vmin=0, vmax=40)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                # A censored cell is NOT a missing value: parity is not reached anywhere inside the
                # 0-40 % sweep, which is the same statement Figure 3 makes with a hatched bar. Drawing
                # it as an empty white cell with a dash (the earlier version) reads as "no data".
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                           edgecolor="0.45", hatch="////", lw=0.4, zorder=2))
                ax.text(j, i, "\u2265 40", ha="center", va="center", fontsize=8.0, color="0.2", zorder=3)
            else:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9.0, color="0.15",
                        zorder=3)
    ax.set_xticks(range(len(targets)), [f"{t}\n({s})" for t, s in
                                        zip(targets, ["Latin", "Cyrillic", "Cyrillic", "Arabic",
                                                      "Devanagari", "CJK"])], fontsize=9.0)
    ax.set_yticks(range(len(src)), list(src), fontsize=9.0)
    ax.set_xlabel("target language, paired with English")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="review capacity parity needs (% of traffic)")
    fig.text(0.5, -0.06, "hatched \u201c\u2265 40\u201d = two-sided parity is not reached inside the "
                         "0\u201340 % sweep", ha="center", va="top", fontsize=8.5, color="0.3")
    save(fig, "fig_heatmap")


# ---------------------------------------------------------------- 4. ablation panel
def fig_ablation():
    p = os.path.join(DET, "ablations_NAT.json")
    if not os.path.exists(p):
        raise RuntimeError("ablation: no artefact at " + p)
    a = json.load(open(p, encoding="utf-8"))
    r = (a.get("pairs") or {}).get("vi") or a
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    # (block, value key, error key, x label, y label). The calibration block stores D_test_mean /
    # D_test_sd — reading "parity_first_D_test" there is why v14's right panel was empty.
    panels = [("review_cost", "price_of_composition", None,
               "per-item review cost $c_{VI}$", "price of composition (pp)"),
              ("traffic_mix", "parity_first_D_test", None,
               "traffic mix (share of VI)", "parity-first $D$ test (pp)"),
              ("calibration_size", "D_test_mean", "D_test_sd",
               "calibration size $n$", "parity-first $D$ test (pp)")]
    for ax, (key, vkey, ekey, xlab, ylab) in zip(axes, panels):
        block = r.get(key) or {}
        if not block:
            raise RuntimeError(f"fig_ablation: the artefact has no '{key}' block")
        if key == "calibration_size":
            keys = [k for k in CAL_ORDER if k in block] + [k for k in block if k not in CAL_ORDER]
            # ordinal axis: "full" (the whole pool) is plotted last, not at x = 0 as in v14
            xs = np.arange(len(keys), dtype=float)
        else:
            keys = sorted(block, key=float)
            xs = np.array([float(k) for k in keys])
        vals = np.array([num((block[k] or {}).get(vkey)) for k in keys], dtype=float)
        y = vals * 100
        require_finite(f"fig_ablation[{key}]", values=y)
        errs = np.array([num((block[k] or {}).get(ekey)) for k in keys], dtype=float) \
            if ekey else np.full(len(keys), np.nan)
        yerr = errs * 100 if np.isfinite(errs).any() else None
        ax.errorbar(xs, y, yerr=yerr, fmt="-o", ms=3.4, lw=1.4, capsize=2.5, elinewidth=0.9,
                    color=ORANGE if key == "review_cost" else BLUE)
        ax.set_xlabel(xlab, fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.axhline(0, color="0.6", lw=0.8)
        if key == "calibration_size":
            ax.set_xticks(xs, [("full" if k == "full" else k) for k in keys], fontsize=8.6)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_ablation_small")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fails = 0
    for fn in (fig_confusion, fig_heatmap, fig_ablation):
        try:
            fn()
        except Exception as exc:                                            # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, type(exc).__name__, exc)
    raise SystemExit(1 if fails else 0)
