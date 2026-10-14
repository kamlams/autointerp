#!/usr/bin/env python
"""This pipeline's explanations on the single-mark fuzzing exam, against the published-protocol
explanations of the same explainer on the same exam (same seed, same windows).

    python results/own_pipeline_260926/analyze_s1.py

Ours:      <model>/<set>/paper-exam-single-mark/<explainer>/L*/feature<id>.json (this folder)
Reference: ../paper_protocol_260920/scores/<explainer>/L*/ (latents active on " to") and
           ../paper_protocol_260924/<model>/scores/<explainer>/L*/ (expert-linked latents; Qwen3 L24
           8006 there and 8066 here are different latents and are left out of the pairing).
"""
import json
import math
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent
EXPLAINERS = [("kimi-k3", "Kimi-K3"), ("qwen3.8-27b", "Qwen3.8-27B"),
              ("deepseek-v4-flash-reasoning", "DeepSeek-V4-Flash-reasoning")]
SETS = [("olmoe-1b-7b-0924", "active-on-to", ["L0", "L7", "L15"], lambda k, L: RES / "paper_protocol_260920" / "scores" / k / L),
        ("olmoe-1b-7b-0924", "expert-linked", ["L1", "L15"], lambda k, L: RES / "paper_protocol_260924" / "olmoe-1b-7b-0924" / "scores" / k / L),
        ("qwen3-30b-a3b", "expert-linked", ["L24"], lambda k, L: RES / "paper_protocol_260924" / "qwen3-30b-a3b" / "scores" / k / L)]
METRICS = [("balanced_acc", "BA"), ("tpr", "TPR"), ("tnr_negA", "TNR A"), ("tnr_negB", "TNR B")]


def fuzz(path: Path):
    if not path.exists():
        return None
    f = json.loads(path.read_text()).get("fuzzing") or {}
    ba = f.get("balanced_acc")
    return f if ba is not None and not (isinstance(ba, float) and math.isnan(ba)) else None


def load(folder: Path) -> dict:
    return {int(p.stem[7:]): f for p in folder.glob("feature*.json") if (f := fuzz(p))}


def mean(rs, key):
    return statistics.mean(r[key] for r in rs) if rs else float("nan")


def ci(d, seed=0):
    d = np.asarray(d)
    b = np.random.default_rng(seed).choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    for key, name in EXPLAINERS:
        print(f"# {name}")
        allo, allr = [], []
        for model, lset, layers, ref_dir in SETS:
            for L in layers:
                ours = load(HERE / model / lset / "paper-exam-single-mark" / key / L)
                ref = load(ref_dir(key, L))
                common = sorted(set(ours) & set(ref))
                o = [ours[i] for i in common]; r = [ref[i] for i in common]
                allo += o; allr += r
                line = "  ".join(f"{lab} {mean(o, m):.3f} vs {mean(r, m):.3f}" for m, lab in METRICS)
                d = ci([a["balanced_acc"] - b["balanced_acc"] for a, b in zip(o, r)]) if len(common) > 2 else (float("nan"),) * 3
                print(f"  {model} {lset} {L}: n={len(common)} (ours {len(ours)}, reference {len(ref)})  {line}"
                      f"  | BA ours - reference {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        if allo:
            d = ci([a["balanced_acc"] - b["balanced_acc"] for a, b in zip(allo, allr)])
            line = "  ".join(f"{lab} {mean(allo, m):.3f} vs {mean(allr, m):.3f}" for m, lab in METRICS)
            print(f"  ALL: n={len(allo)}  {line}  | BA ours - reference {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
                  + ("  *" if d[1] > 0 or d[2] < 0 else ""))
        print()


if __name__ == "__main__":
    main()
