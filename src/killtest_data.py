"""Kill-test data prep for ParityAbstain (selective-parity).

Normalizes two PUBLIC sources into one binary-toxicity JSONL schema:
  EN: Civil Comments 60k/5k/5k  -> data/cc/{train,dev,test}.jsonl  (label = tox_frac >= 0.5)
  VI: ViHSD                      -> data/vihsd/{train,dev,test}.csv (label = label_id in {1,2})

READ-ONLY on both sources. Output: <proj>/data/killtest/{en,vi}_{split}.jsonl
Rows: {"id": str, "text": str, "label": 0|1}
"""
import argparse
import csv
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC_DIR = "data/cc"
VIHSD_DIR = "data/vihsd"
OUT_DIR = os.path.join(PROJ, "data", "killtest")

SPLITS = ["train", "dev", "test"]


def load_en():
    out = {}
    for sp in SPLITS:
        rows = []
        with open(os.path.join(CC_DIR, f"{sp}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({
                    "id": r["post_id"],
                    "text": r["text"],
                    "label": int(float(r["tox_frac"]) >= 0.5),
                })
        out[sp] = rows
    return out


def load_vi():
    out = {}
    for sp in SPLITS:
        rows = []
        with open(os.path.join(VIHSD_DIR, f"{sp}.csv"), newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                rows.append({
                    "id": f"vihsd_{sp}_{i}",
                    "text": r["free_text"].strip(),
                    "label": int(r["label_id"].strip() in ("1", "2")),
                })
        out[sp] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = {}
    for lang, loader in (("en", load_en), ("vi", load_vi)):
        data = loader()
        for sp in SPLITS:
            path = os.path.join(OUT_DIR, f"{lang}_{sp}.jsonl")
            if os.path.exists(path) and not args.force:
                continue
            rows = [r for r in data[sp] if r["text"].strip()]
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[lang] = {
            sp: {
                "n": len(data[sp]),
                "n_toxic": sum(r["label"] for r in data[sp]),
                "prev": round(sum(r["label"] for r in data[sp]) / max(1, len(data[sp])), 4),
            }
            for sp in SPLITS
        }

    print(json.dumps(summary, indent=2))
    with open(os.path.join(OUT_DIR, "SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote ->", OUT_DIR)


if __name__ == "__main__":
    main()
