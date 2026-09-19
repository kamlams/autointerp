#!/usr/bin/env python
"""Side-by-side table of several autointerp runs: detection / fuzzing per feature.

    python compare_runs.py RUN_DIR [RUN_DIR ...] [--labels]

A run directory is any directory of `feature<ID>.json` files. Columns are named after the last
two path components, so `by_explainer/kimi/L7` and `by_explainer/qwen/L7` stay distinguishable.
"""
import argparse
import glob
import json
import math
import os


def load_run(run_dir: str) -> dict:
    out = {}
    for path in glob.glob(os.path.join(run_dir, "feature*.json")):
        r = json.load(open(path))
        out[r["feature_id"]] = r
    return out


def run_name(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    return "/".join(parts[-2:])


def score(result: dict, key: str) -> float | None:
    """Balanced accuracy, or None when the scorer had no probes (very sparse features can lack
    strong-firing pairs for fuzzing) or the value is NaN."""
    s = result.get(key) or {}
    v = s.get("balanced_acc")
    if v is None or (isinstance(v, float) and math.isnan(v)) or not s.get("n_pos"):
        return None
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", action="store_true", help="also print the LLM label of every run")
    args = ap.parse_args()
    runs = {run_name(r): load_run(r) for r in args.runs}
    fids = sorted({fid for r in runs.values() for fid in r})
    human = {fid: next((r[fid].get("human_label") for r in runs.values() if fid in r), None) or ""
             for fid in fids}
    width = max(16, max(len(n) for n in runs) + 2)

    head = f"{'feature':<8} {'human label':<22}" + "".join(f"{name:>{width}}" for name in runs)
    print(head)
    print("-" * len(head))
    for fid in fids:
        row = f"{fid:<8} {human[fid][:22]:<22}"
        for r in runs.values():
            if fid not in r:
                row += f"{'—':>{width}}"
                continue
            d, f = score(r[fid], "detection"), score(r[fid], "fuzzing")
            cell = f"{'  — ' if d is None else f'{d:.2f}'}/{'  — ' if f is None else f'{f:.2f}'}"
            err = (r[fid].get("detection") or {}).get("n_parse_errors", 0)
            row += f"{cell + (f' ({err}err)' if err else ''):>{width}}"
        print(row)
    if args.labels:
        for fid in fids:
            print(f"\n[{fid}]" + (f" human: {human[fid]}" if human[fid] else ""))
            for name, r in runs.items():
                if fid in r:
                    print(f"  {name:<{width}} {r[fid].get('label')}")

    def mean(r: dict, key: str) -> float:
        vals = [v for v in (score(x, key) for x in r.values()) if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\nmean det/fuzz: " + "  ".join(f"{n}={mean(r, 'detection'):.2f}/{mean(r, 'fuzzing'):.2f}"
                                          for n, r in runs.items() if r))


if __name__ == "__main__":
    main()
