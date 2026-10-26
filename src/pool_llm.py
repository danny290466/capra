"""LLM moderator as an external confidence source for the ParityAbstain detector pool (task 5).

No training: the LLM scores every dev/test item with a verbalised P(toxic) in [0,1], which the
selective-prediction analysis consumes exactly like a classifier's softmax (p1 = P(toxic),
conf = max(p, 1-p)). Prompts are in English for every language on purpose — the deployment case is
one hosted model behind one API call, not a per-language prompt-engineering exercise.

Output: <out>/probs.npz with the same key schema as the trainer artefacts + <out>/usage.json.
Resumable: raw per-item records are appended to <out>/raw_<set>.jsonl and skipped on restart.
"""
import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "killtest")
EVAL_SETS = ["en_dev", "en_test", "vi_dev", "vi_test", "tr_dev", "tr_test"]

SYSTEM = ("You are a content-moderation classifier. You are shown one social-media comment. Judge "
          "whether it is hateful or toxic toward a person or group (1) or not (0). Answer with JSON "
          "only, no prose.")
USER = ('Comment:\n"""\n{text}\n"""\n\n'
        'Return exactly: {{"prob_toxic": <integer 0-100, your probability that the comment IS hateful '
        'or toxic>, "label": <0 or 1, your decision>}}')

_lock = threading.Lock()


def chat(base_url, key, model, messages, timeout=120, thinking="none", json_mode=True,
         opencode=False, session=None, max_tokens=60, temperature=0.0):
    """Minimal OpenAI-compatible chat call over stdlib urllib (no SDK needed on the Mac).

    opencode=True targets the opencode gateway (https://opencode.ai/zen/go/v1): it sits behind
    Cloudflare, so it needs a non-default User-Agent plus an `x-opencode-session` UUID or it answers
    403/400 (plan-included models, no per-token billing).
    """
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if thinking != "none":                      # DeepSeek V4 only; harmless fields break some routers
        payload["thinking"] = {"type": thinking}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if opencode:
        headers["User-Agent"] = "opencode/1.0"
        headers["x-opencode-session"] = session or str(uuid.uuid4())
        headers["x-opencode-request"] = str(uuid.uuid4())
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions",
                                data=json.dumps(payload).encode(), method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    msg = d["choices"][0]["message"]
    # thinking models (glm-5.3*, kimi, minimax, deepseek-v4-pro) may leave `content` empty and put
    # everything in `reasoning_content`; the caller parses whichever holds the JSON.
    return (msg.get("content") or ""), (msg.get("reasoning_content") or ""), d.get("usage", {})


def text_hash(t):
    """Stable short hash of an item's text — binds a cached LLM score to the exact input it scored."""
    import hashlib
    return hashlib.md5(t.encode("utf-8")).hexdigest()[:12]


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        p = float(d["prob_toxic"])
        p = min(max(p / 100.0, 1e-4), 1 - 1e-4)
        return p, int(d.get("label", 1 if p >= 0.5 else 0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--thinking", default="none", choices=["none", "disabled", "enabled"],
                    help="DeepSeek V4 turns thinking ON by default and needs an explicit 'disabled'; "
                         "other providers (OpenRouter/Qwen) must not receive the field at all")
    ap.add_argument("--json-mode", action="store_true", default=True,
                    help="send response_format=json_object (unsupported by some OpenRouter providers)")
    ap.add_argument("--no-json-mode", dest="json_mode", action="store_false")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--key-env", default="DEEPSEEK_API_KEY",
                    help="env var holding the API key (use OPENCODE_GO_API_KEY for the opencode gateway)")
    ap.add_argument("--opencode", action="store_true",
                    help="add the opencode gateway headers (UA + x-opencode-session/request)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only N items per set")
    ap.add_argument("--sets", default=",".join(EVAL_SETS))
    ap.add_argument("--data-dir", default=DATA, help="dir with {set}.jsonl (e.g. data/textdetox)")
    args = ap.parse_args()

    key = os.environ.get(args.key_env)
    assert key, f"{args.key_env} not in env"
    os.makedirs(args.out, exist_ok=True)

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    session = str(uuid.uuid4())          # one routing session per run (opencode gateway)
    stats = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "parse_fail": 0}
    arrays = {}
    t0 = time.time()

    counts = {}
    for name in sets:
        fname = name.replace("tr_", "trans_") if name.startswith("tr_") else name
        f = os.path.join(args.data_dir, f"{fname}.jsonl")
        if not os.path.exists(f):
            print(f"[skip] {name}", flush=True)
            continue
        rows = read_jsonl(f)
        if args.limit:
            rows = rows[:args.limit]
        raw_path = os.path.join(args.out, f"raw_{name}.jsonl")
        done = {}
        if os.path.exists(raw_path):
            for line in open(raw_path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    # the cache is keyed by ITEM INDEX, so it is only valid if the text matches:
                    # re-prepping the data (new sample, new split) would otherwise silently mismap labels
                    if 0 <= r["i"] < len(rows) and r.get("h") == text_hash(rows[r["i"]]["text"]):
                        done[r["i"]] = r
                except Exception:
                    pass

        todo = [i for i in range(len(rows)) if i not in done]
        print(f"[{name}] n={len(rows)} cached={len(done)} todo={len(todo)}", flush=True)

        def work(i):
            text = rows[i]["text"][:4000]
            last = None
            for attempt in range(4):
                try:
                    content, reasoning, u = chat(args.base_url, key, args.model,
                                                [{"role": "system", "content": SYSTEM},
                                                 {"role": "user", "content": USER.format(text=text)}],
                                                thinking=args.thinking, json_mode=args.json_mode,
                                                opencode=args.opencode, session=session,
                                                max_tokens=args.max_tokens, temperature=args.temperature)
                    parsed = parse(content) or parse(reasoning)
                    if parsed is None:
                        last = f"parse_fail: content={content[:80]!r} reasoning={reasoning[:80]!r}"
                        time.sleep(1 + attempt)
                        continue
                    with _lock:
                        stats["calls"] += 1
                        stats["prompt_tokens"] += u.get("prompt_tokens", 0)
                        stats["completion_tokens"] += u.get("completion_tokens", 0)
                    return {"i": i, "p": parsed[0], "lab": parsed[1], "label_true": int(rows[i]["label"])}
                except urllib.error.HTTPError as e:          # rate limit, 5xx, bad request: keep the body
                    try:
                        body = e.read().decode()[:200]
                    except Exception:
                        body = ""
                    last = f"HTTP {e.code}: {body}"
                    time.sleep(2 + 3 * attempt)
                except Exception as e:                       # timeouts, connection resets
                    last = str(e)[:200]
                    time.sleep(2 + 3 * attempt)
            with _lock:
                stats["parse_fail"] += 1
                if last and ("data_inspection" in last or "inappropriate" in last):
                    stats["refused_by_policy"] = stats.get("refused_by_policy", 0) + 1
            print(f"  [fail] {name}#{i}: {last}", flush=True)
            with open(os.path.join(args.out, f"refused_{name}.jsonl"), "a", encoding="utf-8") as rf:
                rf.write(json.dumps({"i": i, "reason": last, "label_true": int(rows[i]["label"])},
                                    ensure_ascii=False) + "\n")
            return None

        with ThreadPoolExecutor(max_workers=args.workers) as ex, open(raw_path, "a", encoding="utf-8") as fh:
            for j, res in enumerate(ex.map(work, todo)):
                if res is not None:
                    res["h"] = text_hash(rows[res["i"]]["text"])   # bind the record to its text
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                if (j + 1) % 500 == 0:
                    el = time.time() - t0
                    print(f"  {name}: {j+1}/{len(todo)} ({el:.0f}s, {stats['prompt_tokens']} in-tok)", flush=True)

        recs = {}
        for line in open(raw_path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("h") == text_hash(rows[r["i"]]["text"]):      # ignore stale/mismatched records
                recs[r["i"]] = r
        # An unscored item must never become a plausible number. p = 0.5 has confidence 0.5, i.e. the
        # MINIMUM, so every defaulted item was guaranteed to enter the review queue and every table built
        # on it was contaminated (review findings R3-05, R3-06). Unscored items are stored as NaN and
        # flagged; a detector with unscored items is not usable in a table.
        p1 = np.full((len(rows), 2), np.nan)
        scored = np.zeros(len(rows), dtype=bool)
        labels = np.array([int(r["label"]) for r in rows])
        by_set = {}
        for i in range(len(rows)):
            if i in recs:
                p1[i, 1] = recs[i]["p"]
                p1[i, 0] = 1 - recs[i]["p"]
                scored[i] = True
            else:
                s = str(rows[i].get("set", rows[i].get("split", "?")))
                by_set[s] = by_set.get(s, 0) + 1
        missing = int((~scored).sum())
        if missing:
            print(f"  [ERROR] {name}: {missing} items UNSCORED ({by_set}); stored as NaN, not as a score. "
                  f"Exclude this detector from any comparison (review finding R3-05).", flush=True)
        counts[name] = {"n": len(rows), "unscored": missing, "unscored_by_set": by_set,
                        "usable_in_tables": missing == 0}
        arrays[f"{name}_probs"] = p1
        arrays[f"{name}_labels"] = labels
        arrays[f"{name}_scored"] = scored

    np.savez_compressed(os.path.join(args.out, "probs.npz"), **arrays)
    with open(os.path.join(args.out, "usage.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "stats": stats, "elapsed_s": round(time.time() - t0, 1),
                   "sets": sets, "counts": counts}, f, indent=2)
    print("[done]", args.out, "stats:", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
