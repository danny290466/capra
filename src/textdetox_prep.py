"""Second language pair(s) for the parity study — prepare textdetox/multilingual_toxicity_dataset.

Why this dataset: 15 languages, 5 000 comments each, annotated under ONE shared protocol (TextDetox
2024), which removes the annotation-guideline confound that a Jigsaw-style mix would introduce.
Languages chosen: en + de (Latin, high-resource control) + uk (Cyrillic) + ar (Arabic script, RTL).

Emits our standard JSONL schema ({text, label}) plus a stratified 70/15/15 train/dev/test split, so
the existing trainer/analyser stack can consume it unchanged once --data-dir is pointed here.

Run on a box with `datasets` + `pyarrow` installed (ubuntu-2 env toxkgvi-gpu-ft):
  python textdetox_prep.py --out data/textdetox
"""
import argparse
import json
import os
import random

from datasets import load_dataset

D = "textdetox/multilingual_toxicity_dataset"
# Latin, Cyrillic x2, Arabic (RTL), Devanagari, CJK. Hebrew is deliberately absent: the dataset ships
# only 2 011 items for `he` (all others 5 000), which cannot satisfy the equal-n convention.
LANGS = ["en", "de", "uk", "ru", "ar", "hi", "zh"]


def load_lang(lang):
    """The repo stores one parquet per language; the HF split name IS the language code."""
    for attempt in (dict(path=D, split=lang), dict(path=D, name="default", split=lang)):
        try:
            return load_dataset(**attempt)
        except Exception as e:                                    # noqa: BLE001
            last = e
    raise RuntimeError(f"cannot load {lang}: {last}")


def strat_split(rows, dev_frac, test_frac, seed):
    rng = random.Random(seed)
    by_lab = {}
    for r in rows:
        by_lab.setdefault(r["label"], []).append(r)
    parts = {"train": [], "dev": [], "test": []}
    for lab, rs in by_lab.items():
        rng.shuffle(rs)
        n = len(rs)
        n_dev, n_test = int(round(n * dev_frac)), int(round(n * test_frac))
        parts["dev"] += rs[:n_dev]
        parts["test"] += rs[n_dev:n_dev + n_test]
        parts["train"] += rs[n_dev + n_test:]
    for p in parts.values():
        rng.shuffle(p)
    return parts


def strat_subsample(rows, n, seed):
    """Label-stratified cap to exactly `n` rows (same n for every language -> comparable pairs)."""
    if n <= 0 or len(rows) <= n:
        return rows
    rng = random.Random(seed)
    by_lab = {}
    for r in rows:
        by_lab.setdefault(r["label"], []).append(r)
    keep, used = [], 0
    labs = sorted(by_lab)
    for i, lab in enumerate(labs):
        rs = by_lab[lab][:]
        rng.shuffle(rs)
        want = int(round(n * len(rs) / len(rows)))
        if i == len(labs) - 1:
            want = n - used
        keep += rs[:want]
        used += len(keep) - used
    rng.shuffle(keep)
    return keep[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/textdetox")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--common-n", type=int, default=4500,
                    help="cap EVERY language to the same raw n (stratified) so pairs share n; 0 = off")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for lang in [s.strip() for s in args.langs.split(",") if s.strip()]:
        ds = load_lang(lang)
        cols = ds.column_names
        text_col = "text" if "text" in cols else cols[0]
        lab_col = "label" if "label" in cols else cols[1]
        seen, rows = set(), []
        for r in ds:
            t = (r[text_col] or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            rows.append({"text": t, "label": int(r[lab_col])})
        n_dedup = len(rows)
        rows = strat_subsample(rows, args.common_n, args.seed)
        if args.common_n and n_dedup < args.common_n:
            print(f"  [WARN] {lang}: only {n_dedup} items after dedup -> cannot reach common n "
                  f"{args.common_n}; this breaks the equal-n convention", flush=True)
        parts = strat_split(rows, args.dev_frac, args.test_frac, args.seed)
        for name, rs in parts.items():
            with open(os.path.join(args.out, f"{lang}_{name}.jsonl"), "w", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[lang] = {"raw": len(ds), "dedup": n_dedup, "used": len(rows),
                         **{k: len(v) for k, v in parts.items()},
                         "prev": round(sum(r["label"] for r in rows) / max(len(rows), 1), 4),
                         "cols": cols}
        print(f"[{lang}] raw={len(ds)} dedup={n_dedup} used={len(rows)} train/dev/test="
              f"{len(parts['train'])}/{len(parts['dev'])}/{len(parts['test'])} "
              f"prev={summary[lang]['prev']}", flush=True)


    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": D, "args": vars(args), "langs": summary}, f, indent=2)
    print("->", args.out)


if __name__ == "__main__":
    main()
