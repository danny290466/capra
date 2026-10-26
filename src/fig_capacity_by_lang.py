"""Figure 3 — what parity costs per target language (the script-distance claim, quantified).

Reads the pair_parity.json artefacts of the second-language-pair run and draws two panels:
  A. minimum parity capacity: the share of traffic a review team must see before the two languages'
     residual risks can be equalised (structural threshold); languages that never reach it inside the
     swept range are drawn as hatched ">= 40 %" bars — i.e. parity is not affordable for them.
  B. measured disparity at the fixed C = 5 % budget (GLOBAL pooled threshold), signed so that the
     direction of the disadvantage is visible.

Usage: python fig_capacity_by_lang.py --root outputs/textdetox --out figures/fig_capacity_by_lang
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

FAMILY = {"de": "Latin", "en": "Latin", "uk": "Cyrillic", "ru": "Cyrillic",
          "ar": "Arabic", "he": "Hebrew", "hi": "Devanagari", "zh": "CJK"}
CAP = 0.40          # the sweep's ceiling: beyond this we call parity unaffordable
ORDER = ["de", "uk", "ru", "ar", "he", "hi", "zh"]
PRETTY = {"xlmr_base": "XLM-R base", "xlmr_large": "XLM-R large",
          "llm_deepseek_flash": "DeepSeek-flash (moderator)",
          "llm_gemini31_flashlite": "Gemini-3.1-flash-lite (moderator)",
          "llm_qwen38_flash": "Qwen-3.8-flash (moderator)"}


def load(root):
    data = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "pair_parity.json"))):
        det = os.path.basename(os.path.dirname(f))
        d = json.load(open(f, encoding="utf-8"))
        data[det] = d.get("pairs", {})
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/textdetox")
    ap.add_argument("--out", default="figures/fig_capacity_by_lang")
    ap.add_argument("--budget-key", default="0.05")
    args = ap.parse_args()

    data = load(args.root)
    # The three detectors that Table 6 reports. The other two moderators did not score every item,
    # and the equal-n convention excludes them, so drawing them here would contradict the table
    # (v14 shipped a caption that promised "each detector" over a figure with a single series).
    dets = [d for d in ("xlmr_base", "xlmr_large", "llm_deepseek_flash") if d in data] or sorted(data)
    targets = [t for t in ORDER if any(t in v for v in data.values())]
    if not targets:
        raise SystemExit("no pair artefacts found")

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 6.4), sharex=True)
    w = 0.8 / max(len(dets), 1)
    xs = np.arange(len(targets))

    for i, det in enumerate(dets):
        caps, disps, infeas = [], [], []
        for t in targets:
            e = data[det].get(t, {})
            cap = e.get("min_parity_capacity", [None])[0]
            caps.append(CAP * 100 if cap is None else cap * 100)
            infeas.append(cap is None)
            disps.append(e.get("at_c_ref", {}).get("GLOBAL", {}).get("risk_gap", np.nan) * 100)
        off = (i - (len(dets) - 1) / 2) * w
        b = axes[0].bar(xs + off, caps, width=w, label=PRETTY.get(det, det))
        for rect, inf in zip(b, infeas):
            if inf:
                rect.set_hatch("///")
                rect.set_alpha(0.55)
                axes[0].text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.6, "≥", ha="center",
                             fontsize=9)
        axes[1].bar(xs + off, disps, width=w, label=PRETTY.get(det, det))

    axes[0].axhline(5, color="crimson", ls="--", lw=1.2)
    axes[0].text(len(targets) - 0.4, 5.4, "typical budget C = 5 %", color="crimson", fontsize=9, ha="right")
    axes[0].set_ylabel("minimum parity capacity (% of traffic)")
    axes[0].set_title("A — what parity costs: review capacity needed to equalise residual risk\n"
                      "(hatched “≥” = not reached inside the 0–40 % sweep)", fontsize=11, loc="left")
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("disparity at C = 5 % (pp)")
    # the sign convention belongs in the caption, not in a title that gets clipped at this width
    axes[1].set_title("B — disparity at the same fixed budget", fontsize=11, loc="left")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"{t}\n({FAMILY.get(t, '?')})" for t in targets])
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=170)
    print("wrote", f"{args.out}.png/.pdf")


if __name__ == "__main__":
    main()
