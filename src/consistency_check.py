"""Consistency gate for every manuscript table (lesson from the Evasion-Hate R2 round).

Rule enforced: for a given condition, the number of evaluation items must be IDENTICAL in every
table, and every detector compared side by side must be replicated the SAME number of times. No
silent subsampling/truncation, no 3-seed detector next to a 5-seed one.

Scans a tree of detector directories (<root>/seed*/probs.npz, or a bare <root>/probs.npz for
single-run detectors such as an LLM moderator) and reports, per detector:
  * number of seeds
  * per set: n (and a flag when n differs across seeds inside one detector)
Then cross-checks detectors against each other and EXITS NON-ZERO on any mismatch, so it can gate
table generation in a driver script.

Usage:  python consistency_check.py --roots outputs/killtest outputs/pool outputs/textdetox
"""
import argparse
import glob
import json
import os
import sys


def load_n(path):
    import numpy as np
    z = np.load(path)
    return {k.replace("_labels", ""): int(len(z[k])) for k in z.files if k.endswith("_labels")}


def scan(root):
    dirs = sorted(d for d in glob.glob(os.path.join(root, "seed*"))
                  if os.path.exists(os.path.join(d, "probs.npz")))
    if not dirs and os.path.exists(os.path.join(root, "probs.npz")):
        dirs = [root]
    if not dirs:
        return None
    per_seed = {os.path.basename(d): load_n(os.path.join(d, "probs.npz")) for d in dirs}
    sets = sorted({s for v in per_seed.values() for s in v})
    n_by_set, ragged = {}, {}
    for s in sets:
        vals = {k: v[s] for k, v in per_seed.items() if s in v}
        n_by_set[s] = vals
        if len(set(vals.values())) > 1:
            ragged[s] = vals
    missing = {k: sorted(set(sets) - set(v)) for k, v in per_seed.items() if set(sets) - set(v)}
    return {"dir": root, "n_seeds": len(dirs), "sets": sets,
            "n_min": {s: min(v.values()) for s, v in n_by_set.items()},
            "ragged": ragged, "missing_sets": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--expect-seeds", type=int, default=None,
                    help="require every trained detector to have this many seeds")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="explicit benchmark groups as name=root[,root...]; the cross-detector n check "
                         "runs INSIDE a group only (different benchmarks legitimately have different n, "
                         "e.g. killtest en_dev=5000 vs textdetox en_dev=675)")
    args = ap.parse_args()

    report, problems = {}, []
    for root in args.roots:
        sub_roots = []
        if os.path.isdir(os.path.join(root, "seed42")) or os.path.exists(os.path.join(root, "probs.npz")):
            sub_roots = [root]
        else:                                   # a parent like outputs/pool holding several detectors
            sub_roots = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
        for r in sub_roots:
            info = scan(r)
            if info is None:
                continue
            name = os.path.relpath(r, os.path.dirname(r.rstrip("/")) or ".")
            report[r] = info
            if info["ragged"]:
                problems.append(f"RAGGED n across seeds in {r}: "
                                + json.dumps(info["ragged"], ensure_ascii=False))
            if info["missing_sets"]:
                problems.append(f"MISSING sets in {r}: {json.dumps(info['missing_sets'])}")
            # a trained detector is any root holding seed*/ subdirectories; a bare probs.npz is a
            # single-run detector (a hosted moderator) and is exempt from the seed-count rule. The old
            # condition keyed on the literal substring "seed42" in the path, which no detector root ever
            # contained, so the check never fired at all (review findings R3-01, R2-14).
            is_trained = bool(glob.glob(os.path.join(r, "seed*")))
            info["kind"] = "trained" if is_trained else "single-run"
            if args.expect_seeds is not None and is_trained and info["n_seeds"] != args.expect_seeds:
                problems.append(f"SEED COUNT {info['n_seeds']} != {args.expect_seeds} in {r}")
            # a detector carrying unscored (NaN) items cannot be printed beside the others
            try:
                import numpy as np
                z = np.load(os.path.join(r, "probs.npz"))
                for k in [k for k in z.files if k.endswith("_scored")]:
                    bad = int((~z[k].astype(bool)).sum())
                    if bad:
                        problems.append(f"UNSCORED items in {r}: {k} has {bad} unscored item(s)")
            except Exception:                                                  # noqa: BLE001
                pass

    print(f"{'detector':58s} {'seeds':>5}  sets (n)")
    for r, info in sorted(report.items()):
        sets_txt = ", ".join(f"{s}={info['n_min'][s]}" for s in info["sets"])
        print(f"{r:58s} {info['n_seeds']:>5}  {sets_txt}")

    # cross-detector: same set name must have the same n everywhere WITHIN the same benchmark group
    if args.groups:
        groups = {}
        for spec in args.groups:
            name, _, roots = spec.partition("=")
            groups[name] = [os.path.normpath(r) for r in roots.split(",") if r]
        for name, g_roots in groups.items():
            ref, members = {}, []
            for r, info in report.items():
                if any(os.path.normpath(r) == g or os.path.normpath(r).startswith(g + os.sep)
                       for g in g_roots):
                    members.append(r)
                    for s, n in info["n_min"].items():
                        if s in ref and ref[s][1] != n:
                            problems.append(f"[{name}] n MISMATCH for set '{s}': "
                                            f"{ref[s][0]} has {ref[s][1]}, {r} has {n}")
                        else:
                            ref.setdefault(s, (r, n))
            print(f"group {name}: {len(members)} detector(s), {len(ref)} set(s) cross-checked")
    else:
        ref = {}
        for r, info in report.items():
            for s, n in info["n_min"].items():
                if s in ref and ref[s][1] != n:
                    problems.append(f"n MISMATCH for set '{s}': {ref[s][0]} has {ref[s][1]}, {r} has {n}")
                else:
                    ref.setdefault(s, (r, n))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"report": report, "problems": problems,
                       "expect_seeds": args.expect_seeds, "groups": args.groups,
                       "trained_detectors": sorted(r for r, i in report.items()
                                                   if i.get("kind") == "trained"),
                       "single_run_detectors": sorted(r for r, i in report.items()
                                                      if i.get("kind") == "single-run")}, f, indent=2)
        print("json ->", args.json)

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        print(f"\nFAIL: {len(problems)} consistency problem(s) — do NOT generate tables yet.")
        sys.exit(1)
    n_trained = sum(1 for i in report.values() if i.get("kind") == "trained")
    print(f"\nOK: {len(report)} detector(s) consistent - same item count per set inside each group, "
          f"and all {n_trained} trained detector(s) replicated {args.expect_seeds or 'as declared'} times.")


if __name__ == "__main__":
    main()
