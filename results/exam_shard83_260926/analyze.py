#!/usr/bin/env python
"""Our 30-word explanations against the published-protocol explanations of the same explainer, both
scored on the single-mark exam built from corpus shard 83 (their evidence came from shard 84).

    python results/exam_shard83_260926/analyze.py [--markdown]

Reads <side>/<model>/<set>/<explainer>/L*/feature<id>.json of this folder (side = own | delphi), pairs
the latents both sides scored (Qwen3 L24: ours explain 8066, the published-protocol run 8006, so
neither enters the pairing), and prints balanced accuracy and its parts (TPR on the firing tokens,
TNR on type-A and type-B wrong marks), ours vs published, with the paired BA difference and a
bootstrap 95% interval (10,000 resamples, seed 0). With the shard-84 exam results present
(../own_pipeline_260926/<model>/<set>/paper-exam-single-mark/ and the scores/ of the two
paper_protocol folders), the same comparison on shard 84 is printed for reference.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent
EXPLAINERS = [("kimi-k3", "Kimi-K3"), ("qwen3.8-27b", "Qwen3.8-27B"),
              ("deepseek-v4-flash-reasoning", "DeepSeek-V4-Flash-reasoning")]
UNITS = [("olmoe-1b-7b-0924", "active-on-to", "L0"), ("olmoe-1b-7b-0924", "active-on-to", "L7"),
         ("olmoe-1b-7b-0924", "active-on-to", "L15"), ("olmoe-1b-7b-0924", "expert-linked", "L1"),
         ("olmoe-1b-7b-0924", "expert-linked", "L15"), ("qwen3-30b-a3b", "expert-linked", "L24")]
METRICS = [("balanced_acc", "BA"), ("tpr", "TPR"), ("tnr_negA", "TNR A"), ("tnr_negB", "TNR B")]


def shard83(side, model, lset, key, layer):
    return HERE / side / model / lset / key / layer


def shard84(side, model, lset, key, layer):
    if side == "own":
        return RES / "own_pipeline_260926" / model / lset / "paper-exam-single-mark" / key / layer
    if lset == "active-on-to":
        return RES / "paper_protocol_260920" / "scores" / key / layer
    return RES / "paper_protocol_260924" / model / "scores" / key / layer


def load(folder: Path) -> tuple[dict, int, int]:
    """feature_id -> fuzzing dict for scored latents; also the counts of files and unscoreable latents."""
    out, files, unscoreable = {}, 0, 0
    for p in sorted(folder.glob("feature*.json")):
        files += 1
        r = json.loads(p.read_text())
        f = r.get("fuzzing") or {}
        ba = f.get("balanced_acc")
        if r.get("not_scoreable") or ba is None or (isinstance(ba, float) and math.isnan(ba)):
            unscoreable += 1
            continue
        out[int(p.stem[7:])] = f
    return out, files, unscoreable


def mean(rs, key):
    xs = [r[key] for r in rs if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
    return statistics.mean(xs) if xs else float("nan")


def ci(d, seed=0):
    d = np.asarray(d, dtype=float)
    if len(d) < 3:
        return float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan")
    b = np.random.default_rng(seed).choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def compare(where, key):
    """Per-unit rows and the pooled row of one explainer; None when a side has no results at all."""
    rows, allo, allr = [], [], []
    for model, lset, layer in UNITS:
        ours, fo, uo = load(where("own", model, lset, key, layer))
        ref, fr, ur = load(where("delphi", model, lset, key, layer))
        common = sorted(set(ours) & set(ref))
        o = [ours[i] for i in common]; r = [ref[i] for i in common]
        allo += o; allr += r
        rows.append(dict(unit=f"{model} {lset} {layer}", n=len(common), files=(fo, fr), unscoreable=(uo, ur),
                         ours={m: mean(o, m) for m, _ in METRICS}, ref={m: mean(r, m) for m, _ in METRICS},
                         d=ci([a["balanced_acc"] - b["balanced_acc"] for a, b in zip(o, r)])))
    if not allo:
        return None
    pooled = dict(unit="all", n=len(allo), ours={m: mean(allo, m) for m, _ in METRICS},
                  ref={m: mean(allr, m) for m, _ in METRICS},
                  d=ci([a["balanced_acc"] - b["balanced_acc"] for a, b in zip(allo, allr)]))
    return rows, pooled


def fmt_d(d):
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" + ("  *" if d[1] > 0 or d[2] < 0 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="the pooled comparison as a markdown table")
    args = ap.parse_args()
    if args.markdown:
        print("| explainer | exam | n | BA ours vs published | TPR | TNR A | TNR B | BA ours - published [95% CI] |")
        print("|---|---|---|---|---|---|---|---|")
    per_unit = []
    for key, name in EXPLAINERS:
        for label, where in (("shard 83", shard83), ("shard 84", shard84)):
            got = compare(where, key)
            if got is None:
                continue
            rows, p = got
            if args.markdown:
                parts = " | ".join(f"{p['ours'][m]:.3f} vs {p['ref'][m]:.3f}" for m, _ in METRICS)
                print(f"| {name} | {label} | {p['n']} | {parts} | {fmt_d(p['d']).strip()} |")
                if label == "shard 83":
                    per_unit.append((name, rows))
                continue
            print(f"# {name}, exam on {label}" + ("" if label == "shard 83" else " (for reference)"))
            for r in rows:
                line = "  ".join(f"{lab} {r['ours'][m]:.3f} vs {r['ref'][m]:.3f}" for m, lab in METRICS)
                extra = (f" (files ours/published {r['files'][0]}/{r['files'][1]}, unscoreable "
                         f"{r['unscoreable'][0]}/{r['unscoreable'][1]})") if label == "shard 83" else ""
                print(f"  {r['unit']}: n={r['n']}{extra}  {line}  | BA diff {fmt_d(r['d'])}")
            line = "  ".join(f"{lab} {p['ours'][m]:.3f} vs {p['ref'][m]:.3f}" for m, lab in METRICS)
            print(f"  ALL: n={p['n']}  {line}  | BA ours - published {fmt_d(p['d'])}\n")
    if args.markdown and per_unit:
        print("\nShard-83 exam per latent set and layer, BA ours vs published (paired n) and the difference:\n")
        print("| set | " + " | ".join(n for n, _ in per_unit) + " |")
        print("|---|" + "---|" * len(per_unit))
        for i, (model, lset, layer) in enumerate(UNITS):
            cells = []
            for _, rows in per_unit:
                r = rows[i]
                cells.append(f"{r['ours']['balanced_acc']:.3f} vs {r['ref']['balanced_acc']:.3f} ({r['n']}), "
                             f"{fmt_d(r['d']).strip()}" if r["n"] else "-")
            print(f"| {model} {lset} {layer} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
