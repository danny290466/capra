#!/usr/bin/env python3
"""Deployment-feasibility numbers per detector: parameters, training rows, measured training time.

Reads each detector's `metrics.json` (parameters come from the loader, training row count from the run)
and the run logs for `train_runtime` / `train_samples_per_second`, which Trainer writes into its log
line but not into metrics.json. Writes outputs/detector_costs.json — the artefact behind the
compute sentence in the manuscript (reviewer pattern R1-C2: parameters, speed, resources).

usage: python tools/detector_cost_table.py --proj .
"""
import argparse
import glob
import json
import os
import re

RUNTIME_RE = re.compile(r"'train_runtime': ([0-9.]+).*?'train_samples_per_second': ([0-9.]+)", re.S)


def runtimes_by_log(proj):
    out = {}
    for path in glob.glob(os.path.join(proj, "logs", "*.log")):
        txt = open(path, encoding="utf-8", errors="ignore").read()
        hits = RUNTIME_RE.findall(txt)
        if hits:
            out[os.path.basename(path)] = [{"train_runtime_s": float(a), "samples_per_s": float(b)}
                                           for a, b in hits]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=".")
    ap.add_argument("--out", default="outputs/detector_costs.json")
    args = ap.parse_args()
    proj = os.path.abspath(args.proj)

    rows = {}
    for path in sorted(glob.glob(os.path.join(proj, "outputs", "*", "*", "seed42", "metrics.json"))) \
            + sorted(glob.glob(os.path.join(proj, "outputs", "*", "seed42", "metrics.json"))):
        rel = os.path.relpath(path, proj)
        d = json.load(open(path, encoding="utf-8"))
        a = d.get("args", {})
        name = rel[:-len("/seed42/metrics.json")].replace(os.sep, "/")
        pm = d.get("params_m")
        rows[name] = {"path": rel, "model": a.get("model"), "langs": a.get("langs"),
                      "params_m": round(float(pm), 1) if pm else None,
                      "n_train": d.get("n_train"), "epochs": a.get("epochs"),
                      "batch": a.get("bs"), "lr": a.get("lr"), "max_len": a.get("max_len")}
    res = {"detectors": rows, "train_runtime_by_log": runtimes_by_log(proj),
           "note": "params_m is what the loader reported; train_runtime comes from the Trainer log line "
                   "(metrics.json does not carry it)"}
    out = os.path.join(proj, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
    for k, v in rows.items():
        pm = "?" if v['params_m'] is None else f"{v['params_m']:.1f}"
        print(f"{k:34s} {str(v['model']):26s} params {pm:>7} M  n_train {str(v['n_train']):>6}  "
              f"langs {v['langs']}")
    print("\ntraining runtimes found in logs:")
    for log, hits in res["train_runtime_by_log"].items():
        print(f"  {log}: " + ", ".join(f"{h['train_runtime_s']:.0f}s @ {h['samples_per_s']:.0f} smp/s"
                                        for h in hits))
    print("json ->", out)


if __name__ == "__main__":
    main()
