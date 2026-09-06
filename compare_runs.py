#!/usr/bin/env python
"""Side-by-side table of several autointerp runs.

    python compare_runs.py outputs/full5_dsflash outputs/v3_dsflash_reasoning [--labels]
"""
import argparse
import glob
import json
import os


def load_run(run_dir: str) -> dict:
    out = {}
    for path in glob.glob(os.path.join(run_dir, "feature*.json")):
        r = json.load(open(path))
        out[r["feature_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", action="store_true", help="also print LLM labels")
    args = ap.parse_args()
    runs = {os.path.basename(r.rstrip("/")): load_run(r) for r in args.runs}
    fids = sorted({fid for r in runs.values() for fid in r})
    human = {fid: next((r[fid]["human_label"] for r in runs.values() if fid in r), "")
             for fid in fids}

    head = f"{'feature':<8} {'human':<22}" + "".join(f"{name[:22]:>24}" for name in runs)
    print(head)
    print("-" * len(head))
    for fid in fids:
        row = f"{fid:<8} {human[fid][:22]:<22}"
        for r in runs.values():
            if fid in r:
                d, f = r[fid]["detection"], r[fid]["fuzzing"]
                row += f"{d['balanced_acc']:>12.2f}/{f['balanced_acc']:<4.2f}"
                row += f"{'(' + str(d['n_parse_errors']) + 'err)' if d['n_parse_errors'] else '':>7}"
            else:
                row += f"{'—':>24}"
        print(row)
    if args.labels:
        for fid in fids:
            print(f"\n[{fid}] human: {human[fid]}")
            for name, r in runs.items():
                if fid in r:
                    print(f"  {name[:22]:<22} {r[fid]['label']}")
    means = {name: (sum(r[f]['detection']['balanced_acc'] for f in r) / len(r),
                    sum(r[f]['fuzzing']['balanced_acc'] for f in r) / len(r))
             for name, r in runs.items() if r}
    print("\nmean det/fuzz: " + "  ".join(f"{n[:22]}={d:.2f}/{f:.2f}" for n, (d, f) in means.items()))


if __name__ == "__main__":
    main()
