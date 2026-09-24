#!/usr/bin/env python
"""Every number in README.md, computed from the files in this folder only.

    python results/paper_protocol_260920/analyze.py
"""
import json
import math
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = [("deepseek-v4-flash", "DeepSeek-V4-Flash"),
          ("deepseek-v4-flash-reasoning", "DeepSeek-V4-Flash-reasoning"),
          ("kimi-k3", "Kimi-K3"),
          ("qwen3.8-27b", "Qwen3.8-27B")]
LAYERS = ("L0", "L7", "L15")


def load(sub: str) -> dict:
    """{(layer, feature_id): record} for every feature file under HERE/sub."""
    out = {}
    for layer in LAYERS:
        for p in sorted((HERE / sub / layer).glob("feature*.json")):
            r = json.loads(p.read_text())
            out[(layer, r["feature_id"])] = r
    return out


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float("nan")


def ci(diffs, seed=0):
    d = np.asarray(diffs)
    boots = np.random.default_rng(seed).choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    expl = {k: load(f"explanations/{k}") for k, _ in MODELS}
    scored = {k: load(f"scores/{k}") for k, _ in MODELS}
    floor = load("scores/floor-shuffled-explanation")
    base = load("baseline-published-exam/deepseek-v4-flash")
    ids = sorted(set().union(*[set(v) for v in expl.values()]))
    print(f"features: {len(ids)} ({', '.join(f'{L} {sum(1 for x in ids if x[0] == L)}' for L in LAYERS)})\n")

    print("## 1. The unmodified published protocol (explainer and examiner DeepSeek-V4-Flash)")
    p0 = [r for r in expl["deepseek-v4-flash"].values() if r.get("detection") and r.get("fuzzing")]
    print(f"   n={len(p0)}  detection {mean(r['detection']['accuracy'] for r in p0):.3f}  "
          f"fuzzing {mean(r['fuzzing']['accuracy'] for r in p0):.3f}  (paper, Gemma-2-9b: 0.74 / 0.76)")
    det_tnr = mean(r['detection']['tnr'] for r in p0)
    fuz = [r["fuzzing"] for r in p0]
    print(f"   detection TNR {det_tnr:.3f};  fuzzing TPR {mean(f['tpr'] for f in fuz):.3f}  TNR {mean(f['tnr'] for f in fuz):.3f}\n")

    print("## 2. Coverage: which features each explainer described")
    for k, name in MODELS:
        unscoreable = [x for x in ids if x in expl[k] and expl[k][x].get("not_scoreable")]
        gone = [x for x in ids if x not in expl[k]]
        described = [x for x in ids if x in expl[k] and not expl[k][x].get("not_scoreable")
                     and (expl[k][x].get("explanation") or "").strip()]
        print(f"   {name:28s} described {len(described):2d}/{len(ids)}"
              f"   not scoreable {['/'.join(map(str, x)) for x in unscoreable]}"
              f"   no final reply {['/'.join(map(str, x)) for x in gone]}")
    print()

    print("## 3. Single-mark exam: four explainers, examiner DeepSeek-V4-Flash-reasoning")
    ok = lambda k, x: x in scored[k] and scored[k][x].get("fuzzing")
    print("   own set (each explainer over every feature it described), balanced accuracy:")
    for k, name in MODELS:
        rs = [scored[k][x]["fuzzing"] for x in ids if ok(k, x)]
        per = "  ".join(f"{L} {mean(scored[k][x]['fuzzing']['balanced_acc'] for x in ids if x[0] == L and ok(k, x)):.3f}"
                        for L in LAYERS)
        print(f"   {name:28s} n={len(rs):2d}  BA {mean(f['balanced_acc'] for f in rs):.3f}  TPR {mean(f['tpr'] for f in rs):.3f}"
              f"  TNR A {mean(f['tnr_negA'] for f in rs):.3f}  TNR B {mean(f['tnr_negB'] for f in rs):.3f}   {per}")
    common = [x for x in ids if all(ok(k, x) for k, _ in MODELS)]
    print(f"\n   paired, on the {len(common)} features all four described:")
    for k, name in MODELS:
        rs = [scored[k][x]["fuzzing"] for x in common]
        print(f"   {name:28s} BA {mean(f['balanced_acc'] for f in rs):.3f}  TPR {mean(f['tpr'] for f in rs):.3f}"
              f"  TNR A {mean(f['tnr_negA'] for f in rs):.3f}  TNR B {mean(f['tnr_negB'] for f in rs):.3f}")
    ref = "deepseek-v4-flash"
    for k, name in MODELS:
        if k == ref:
            continue
        m, lo, hi = ci([scored[k][x]["fuzzing"]["balanced_acc"] - scored[ref][x]["fuzzing"]["balanced_acc"] for x in common])
        print(f"   {name} - DeepSeek-V4-Flash: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if lo > 0 or hi < 0 else ""))
    fl = [x for x in common if x in floor and floor[x].get("fuzzing")]
    print(f"\n   floor (each feature scored with another feature's DeepSeek-V4-Flash explanation), n={len(fl)}:"
          f"  BA {mean(floor[x]['fuzzing']['balanced_acc'] for x in fl):.3f}"
          f"  TPR {mean(floor[x]['fuzzing']['tpr'] for x in fl):.3f}"
          f"  TNR A {mean(floor[x]['fuzzing']['tnr_negA'] for x in fl):.3f}"
          f"  TNR B {mean(floor[x]['fuzzing']['tnr_negB'] for x in fl):.3f}\n")

    print("## 4. What the modification changes: same DeepSeek-V4-Flash explanations, same reasoning examiner")
    both = [x for x in ids if x in base and base[x].get("fuzzing") and ok(ref, x)]
    b = [base[x]["fuzzing"] for x in both]; s = [scored[ref][x]["fuzzing"] for x in both]
    print(f"   n={len(both)}")
    print(f"   published fuzzing exam (all marks):  BA {mean(f['balanced_acc'] for f in b):.3f}  TPR {mean(f['tpr'] for f in b):.3f}  TNR {mean(f['tnr'] for f in b):.3f}")
    print(f"   single-mark exam:                    BA {mean(f['balanced_acc'] for f in s):.3f}  TPR {mean(f['tpr'] for f in s):.3f}"
          f"  TNR A {mean(f['tnr_negA'] for f in s):.3f}  TNR B {mean(f['tnr_negB'] for f in s):.3f}")
    m, lo, hi = ci([scored[ref][x]["fuzzing"]["balanced_acc"] - base[x]["fuzzing"]["balanced_acc"] for x in both])
    print(f"   single-mark - published: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    counts = [(f.get("n_pos"), f.get("n_negA"), f.get("n_negB")) for f in s]
    print(f"   realized probes per feature on the single-mark exam: positives {sorted({c[0] for c in counts})},"
          f" A {sorted({c[1] for c in counts})}, B min {min(c[2] for c in counts)} max {max(c[2] for c in counts)}")


if __name__ == "__main__":
    main()
