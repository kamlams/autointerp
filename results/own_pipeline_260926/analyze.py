#!/usr/bin/env python
"""Every number in README.md, computed from the files in this folder only.

    python results/own_pipeline_260926/analyze.py
"""
import json
import math
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPLAINERS = [("kimi-k3", "Kimi-K3"), ("qwen3.8-27b", "Qwen3.8-27B"),
              ("deepseek-v4-flash-reasoning", "DeepSeek-V4-Flash-reasoning")]
EXAMS = [("explanations-and-scores", "DeepSeek-V4-Flash-reasoning"),
         ("rescored-by-deepseek-v4-flash", "DeepSeek-V4-Flash")]
LIMIT = 30


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def mean(xs):
    xs = [x for x in xs if ok(x)]
    return statistics.mean(xs) if xs else float("nan")


def ci(d, seed=0):
    d = np.asarray(d)
    b = np.random.default_rng(seed).choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    F = json.loads((HERE / "features.json").read_text())
    for base in ("olmoe-1b-7b-0924", "qwen3-30b-a3b"):
        for lset, layers in F[base]["sets"].items():
            print(f"# {base} / {lset}")
            for layer, groups in layers.items():
                ids = [i for g in groups.values() for i in g]
                rec = {}   # (exam, explainer, id) -> record
                for exam, _ in EXAMS:
                    for key, _ in EXPLAINERS:
                        for i in ids:
                            r = load(HERE / base / lset / exam / key / layer / f"feature{i}.json")
                            if r:
                                rec[exam, key, i] = r
                print(f"\n## {layer}: {len(ids)} latents")
                for key, name in EXPLAINERS:
                    got = [rec["explanations-and-scores", key, i] for i in ids if ("explanations-and-scores", key, i) in rec]
                    miss = [i for i in ids if ("explanations-and-scores", key, i) not in rec]
                    words = [r.get("explanation_words") for r in got if r.get("explanation_words") is not None]
                    short = sum(1 for r in got if "explanation_first" in r)
                    over = [r["feature_id"] for r in got if r.get("over_limit")]
                    print(f"   {name:28s} explained {len(got)}/{len(ids)}" + (f" (missing {miss})" if miss else "")
                          + f"; words mean {mean(words):.1f} max {max(words) if words else '-'}; shortened {short}"
                          + (f"; still over {LIMIT}: {over}" if over else ""))
                for exam, ename in EXAMS:
                    print(f"   examiner {ename}:")
                    for gname, gids in list(groups.items()) + ([("all", ids)] if len(groups) > 1 else []):
                        for key, name in EXPLAINERS:
                            rs = [rec[exam, key, i] for i in gids if (exam, key, i) in rec]
                            if not rs:
                                continue
                            det = mean(r["detection"].get("balanced_acc") for r in rs)
                            fz = [r["fuzzing"].get("balanced_acc") for r in rs if r["fuzzing"].get("n_pos")]
                            print(f"     {gname:22s} {name:28s} n={len(rs):2d}  detection {det:.3f}  fuzzing {mean(fz):.3f}"
                                  f" (n={len(fz)})")
                common = [i for i in ids if all(("explanations-and-scores", k, i) in rec for k, _ in EXPLAINERS)]
                for metric in ("detection", "fuzzing"):
                    for key, name in EXPLAINERS[1:]:
                        pairs = [(rec["explanations-and-scores", key, i][metric].get("balanced_acc"),
                                  rec["explanations-and-scores", "kimi-k3", i][metric].get("balanced_acc")) for i in common]
                        pairs = [(a, b) for a, b in pairs if ok(a) and ok(b)]
                        if len(pairs) > 2:
                            m, lo, hi = ci([a - b for a, b in pairs])
                            print(f"   paired {metric:9s} {name} - Kimi-K3 (n={len(pairs)}): {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                                  + ("  *" if lo > 0 or hi < 0 else ""))
                short_fz = sorted({i for i in ids for k, _ in EXPLAINERS
                                   if ("explanations-and-scores", k, i) in rec
                                   and (rec["explanations-and-scores", k, i]["fuzzing"].get("n_pos") or 0) < 20})
                if short_fz:
                    print(f"   latents with fewer than 20 fuzzing pairs (few strong firings): {short_fz}")
            print()


if __name__ == "__main__":
    main()
