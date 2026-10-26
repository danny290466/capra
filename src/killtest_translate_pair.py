"""Paired EN->VI control for the ParityAbstain kill-test.

Translates the EN eval sets (Civil Comments dev/test, original labels) into Vietnamese with
Helsinki-NLP/opus-mt-en-vi, so the SAME items + SAME labels exist in both languages.
That removes the domain/content/base-rate confound of comparing Civil Comments vs ViHSD:
any remaining EN-vs-VI gap at matched risk is attributable to the model's language behaviour.

Source data stays untouched. Output: <proj>/data/killtest/trans_{dev,test}.jsonl
"""
import argparse
import json
import os

import torch
from transformers import MarianMTModel, MarianTokenizer

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data", "killtest")
MODEL = "Helsinki-NLP/opus-mt-en-vi"


def translate(texts, tok, model, device, bs=64, max_len=256):
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=max_len, num_beams=1)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        if (i // bs) % 10 == 0:
            print(f"  translated {min(i + bs, len(texts))}/{len(texts)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).to(device).eval()
    model.half() if device == "cuda" else None
    print(f"loaded {MODEL} on {device} (fp16={device == 'cuda'})", flush=True)

    for sp in args.splits:
        src = os.path.join(DATA, f"en_{sp}.jsonl")
        dst = os.path.join(DATA, f"trans_{sp}.jsonl")
        if os.path.exists(dst) and not args.force:
            print(f"skip {dst} (exists)")
            continue
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        vi = translate([r["text"] for r in rows], tok, model, device)
        with open(dst, "w", encoding="utf-8") as f:
            for r, t in zip(rows, vi):
                f.write(json.dumps({"id": "tr_" + r["id"], "src_id": r["id"], "text": t,
                                    "label": int(r["label"])}, ensure_ascii=False) + "\n")
        n_tox = sum(r["label"] for r in rows)
        print(f"{sp}: {len(rows)} items translated (toxic={n_tox}, prev={n_tox / len(rows):.4f}) -> {dst}")


if __name__ == "__main__":
    main()
