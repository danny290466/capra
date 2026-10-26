"""Kill-test trainer for ParityAbstain: ONE multilingual XLM-R serving BOTH languages.

Design choice (recorded in the result note): a single shared model is the only setup in which
"one global confidence threshold for both languages" is a real deployment object. Training is
sample-balanced across languages (--per-lang each) but keeps each language's NATURAL label
prevalence (no class weighting) so that per-language score distributions stay comparable to
deployment.

Outputs per seed  ->  <out>/probs.npz  (softmax probs + labels for every eval set)
                      <out>/metrics.json
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
MODEL = "xlm-roberta-base"

# eval set name -> (jsonl file, is_paired_vi)
EVAL_SETS = {
    "en_dev": ("en_dev.jsonl", False),
    "en_test": ("en_test.jsonl", False),
    "vi_dev": ("vi_dev.jsonl", False),
    "vi_test": ("vi_test.jsonl", False),
    "tr_dev": ("trans_dev.jsonl", True),
    "tr_test": ("trans_test.jsonl", True),
}


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


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


def build_train(per_lang, seed):
    en = read_jsonl(os.path.join(DATA, "en_train.jsonl"))
    vi = read_jsonl(os.path.join(DATA, "vi_train.jsonl"))
    rng = random.Random(seed)
    en = rng.sample(en, min(per_lang, len(en)))
    vi = rng.sample(vi, min(per_lang, len(vi)))
    rows = en + vi
    rng.shuffle(rows)
    return rows


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--per-lang", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = args.out
    probs_path = os.path.join(out, "probs.npz")
    if os.path.exists(probs_path):
        print(f"[skip] {probs_path} exists")
        return
    os.makedirs(out, exist_ok=True)

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    train_rows = build_train(args.per_lang, args.seed)
    n_tox = sum(r["label"] for r in train_rows)
    print(f"[data] train n={len(train_rows)} toxic={n_tox} prev={n_tox / len(train_rows):.4f}", flush=True)

    ds = {}
    for name, (fname, _) in EVAL_SETS.items():
        path = os.path.join(DATA, fname)
        if os.path.exists(path):
            ds[name] = ToxDS(read_jsonl(path), tok, args.max_len)
            print(f"[data] {name}: n={len(ds[name])}", flush=True)
        else:
            print(f"[data] {name}: MISSING ({fname}) -> skipped", flush=True)

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

    targs = TrainingArguments(
        output_dir=os.path.join(out, "ckpt"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=128,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_en_dev_f1_toxic",
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
                   "eval": metrics}, f, indent=2)
    print("[done]", out)
    for k in sorted(metrics):
        if k.startswith(("en_", "vi_", "tr_")) and "f1" in k:
            print(f"  {k} = {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
