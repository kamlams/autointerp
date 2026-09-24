#!/usr/bin/env python
"""Every number in README.md, computed from the files in this folder only.

    python results/paper_protocol_260924/analyze.py
"""
import json
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPLAINERS = [("kimi-k3", "Kimi-K3"), ("qwen3.8-27b", "Qwen3.8-27B"),
              ("deepseek-v4-flash-reasoning", "DeepSeek-V4-Flash-reasoning")]
METRICS = ("balanced_acc", "tpr", "tnr_negA", "tnr_negB")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def ci(diffs, seed=0):
    d = np.asarray(diffs)
    boots = np.random.default_rng(seed).choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def row(scores):
    return "  ".join(f"{m.replace('balanced_acc', 'BA').replace('tpr', 'TPR').replace('tnr_neg', 'TNR ')} "
                     f"{statistics.mean(s[m] for s in scores):.3f}" for m in METRICS)


def main():
    features = json.loads((HERE / "features.json").read_text())
    for base in ("olmoe-1b-7b-0924", "qwen3-30b-a3b"):
        info = features[base]
        if not (HERE / base / "scores").exists():
            print(f"# {base}: no results yet\n")
            continue
        print(f"# {base} ({info['model']}, SAE {info['sae_run']})")
        for layer, groups in info["groups"].items():
            s = json.loads((HERE / base / "sampling" / f"{layer}.json").read_text())
            print(f"\n## {layer}: {sum(map(len, groups.values()))} latents; SAE FVU "
                  f"{s.get('fvu_first_batches', 'not recorded')}; corpus {s['tokens']} tokens")
            fuzz = {}   # (explainer, latent) -> fuzzing record
            for key, name in EXPLAINERS:
                missing = []
                for ids in groups.values():
                    for i in ids:
                        e = load(HERE / base / "explanations" / key / layer / f"feature{i}.json")
                        f = load(HERE / base / "scores" / key / layer / f"feature{i}.json")
                        if not e or not (e.get("explanation") or "").strip():
                            missing.append(i)
                        elif f and f.get("fuzzing"):
                            fuzz[key, i] = f["fuzzing"]
                print(f"   {name:28s} explained {sum(map(len, groups.values())) - len(missing)}"
                      f"/{sum(map(len, groups.values()))}" + (f"; no final reply: {missing}" if missing else ""))
            for group, ids in groups.items():
                print(f"\n   {group}")
                for key, name in EXPLAINERS:
                    sc = [fuzz[key, i] for i in ids if (key, i) in fuzz]
                    if sc:
                        print(f"     {name:28s} n={len(sc):2d}  {row(sc)}")
            ids = [i for g in groups.values() for i in g]
            common = [i for i in ids if all((k, i) in fuzz for k, _ in EXPLAINERS)]
            print(f"\n   all {layer} latents, paired on the {len(common)} every explainer described:")
            for key, name in EXPLAINERS:
                print(f"     {name:28s} {row([fuzz[key, i] for i in common])}")
            for key, name in EXPLAINERS[1:]:
                m, lo, hi = ci([fuzz[key, i]["balanced_acc"] - fuzz["kimi-k3", i]["balanced_acc"] for i in common])
                print(f"     {name} - Kimi-K3: BA {m:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if lo > 0 or hi < 0 else ""))
            short = [(i, s["features"][str(i)]["n_firings"]) for i in ids
                     if (load(HERE / base / "scores" / "kimi-k3" / layer / f"feature{i}.json") or {})
                     .get("n_windows", {}).get("test", 100) < 100]
            if short:
                print(f"   latents with fewer than 100 test windows (too few activating windows): "
                      + ", ".join(f"{i}" for i, _ in short))
        print()


if __name__ == "__main__":
    main()
