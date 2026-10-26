"""Detector-pool trainer for ParityAbstain (task 5).

Generalises killtest_train.py so the same recipe (sample-balanced training, natural prevalence,
fp16, 3 epochs, dev-calibrated thresholds, probs.npz artefact) can be applied to:

  1. a bigger shared multilingual model      --model FacebookAI/xlm-roberta-large --langs en,vi
  2. an English-only specialist              --model roberta-base            --langs en
  3. a Vietnamese-only specialist            --model vinai/phobert-base      --langs vi --seg-vi
     (the two specialists are merged into one virtual detector by pool_pair.py, so the analysis
      scripts can treat the pair exactly like a single model)

Every eval set present in data/killtest is scored, so probs.npz keeps the same key schema as the
kill-test artefacts (en_dev/en_test/vi_dev/vi_test/tr_dev/tr_test).

Output per seed -> <out>/probs.npz, <out>/metrics.json   (resumable: skips if probs.npz exists)
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data", "killtest")

# Eval sets are discovered from the data dir (see discover_sets); tr_* is an alias for trans_*.jsonl
EXTRA_SETS = ["tr_dev", "tr_test"]


def set_path(data_dir, name):
    fname = "trans_" + name.split("_", 1)[1] if name.startswith("tr_") else name
    return os.path.join(data_dir, f"{fname}.jsonl")


def discover_sets(data_dir, langs):
    names = []
    for lang in langs:
        for part in ("dev", "test"):
            if os.path.exists(set_path(data_dir, f"{lang}_{part}")):
                names.append(f"{lang}_{part}")
    for extra in EXTRA_SETS:
        if os.path.exists(set_path(data_dir, extra)):
            names.append(extra)
    return names


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def segment_vi(text):
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")


class ToxDS(Dataset):
    def __init__(self, rows, tok, max_len):
        self.enc = tok([r["text"] for r in rows], truncation=True, max_length=max_len, padding=False)
        self.labels = [int(r["label"]) for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        d["labels"] = torch.tensor(self.labels[i])
        return d


def build_train(data_dir, langs, per_lang, seed, seg_vi):
    rows = []
    for lang in langs:
        f = os.path.join(data_dir, f"{lang}_train.jsonl")
        rs = read_jsonl(f)
        rng = random.Random(seed + sum(ord(ch) for ch in lang))   # independent sample per language
        rs = rng.sample(rs, min(per_lang, len(rs)))
        if seg_vi and lang == "vi":
            rs = [{"text": segment_vi(r["text"]), "label": r["label"]} for r in rs]
        rows += rs
        print(f"[data] {lang}: {len(rs)} training rows", flush=True)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--langs", default="en,vi", help="comma list of TRAINING languages")
    ap.add_argument("--eval-langs", default="",
                    help="comma list of languages to SCORE but never TRAIN on. Used by the "
                         "leave-one-out controls (LOLO/LOSO): the held-out language is absent from "
                         "the training mixture and from model selection, which follows the first "
                         "TRAINING language's dev split.")
    ap.add_argument("--data-dir", default=DATA, help="dir holding {lang}_{train,dev,test}.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--per-lang", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seg-vi", action="store_true", help="underthesea word-segment VI text (PhoBERT)")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]

    out = args.out
    probs_path = os.path.join(out, "probs.npz")
    if os.path.exists(probs_path):
        print(f"[skip] {probs_path} exists")
        return
    os.makedirs(out, exist_ok=True)

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {args.model} ({n_par:.0f}M params) langs={langs} seg_vi={args.seg_vi}", flush=True)

    train_rows = build_train(args.data_dir, langs, args.per_lang, args.seed, args.seg_vi)
    n_tox = sum(r["label"] for r in train_rows)
    print(f"[data] train n={len(train_rows)} toxic={n_tox} prev={n_tox / len(train_rows):.4f}", flush=True)

    ds = {}
    eval_extra = [s.strip() for s in args.eval_langs.split(",") if s.strip()]
    if eval_extra:
        print(f"[eval-extra] scoring {eval_extra} without training on them", flush=True)
    for name in discover_sets(args.data_dir, langs + eval_extra):
        f = set_path(args.data_dir, name)
        if not os.path.exists(f):
            print(f"[data] {name}: MISSING -> skipped", flush=True)
            continue
        lang = name.split("_", 1)[0]
        rows = read_jsonl(f)
        if args.seg_vi and lang == "vi":
            rows = [{"text": segment_vi(r["text"]), "label": r["label"]} for r in rows]
        ds[name] = ToxDS(rows, tok, args.max_len)
        print(f"[data] {name}: n={len(ds[name])}", flush=True)

    def compute_metrics(p):
        logits, labels = p
        preds = logits.argmax(-1)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"f1_toxic": f1, "acc": float((preds == labels).mean())}

    best = f"eval_{langs[0]}_dev_f1_toxic"
    targs = TrainingArguments(
        output_dir=os.path.join(out, "ckpt"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=args.fp16,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=best,
        greater_is_better=True,
        dataloader_num_workers=2,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ToxDS(train_rows, tok, args.max_len),
                      eval_dataset=ds, compute_metrics=compute_metrics,
                      data_collator=DataCollatorWithPadding(tok))
    trainer.train()

    arrays, metrics = {}, trainer.evaluate()
    metrics = {k.replace("eval_", ""): v for k, v in metrics.items()}
    for name, d in ds.items():
        logits = trainer.predict(d).predictions
        arrays[f"{name}_probs"] = softmax(logits)
        arrays[f"{name}_labels"] = np.array(d.labels)
    np.savez_compressed(probs_path, **arrays)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "n_train": len(train_rows), "train_prev": n_tox / len(train_rows),
                   "params_m": n_par, "eval": metrics}, f, indent=2)
    print("[done]", out, flush=True)
    for k in sorted(metrics):
        if k.startswith(("en_", "vi_", "tr_")) and "f1" in k:
            print(f"  {k} = {metrics[k]:.4f}", flush=True)


if __name__ == "__main__":
    main()
