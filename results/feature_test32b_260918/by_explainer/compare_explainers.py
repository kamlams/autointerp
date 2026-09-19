#!/usr/bin/env python
"""Reproduce the tables of README.md in this directory from the committed per-feature JSON.

Three explainers wrote an explanation for the same 96 latents; every explanation set was scored by
two examiners:

    <explainer>/L<layer>/feature<ID>.json        scored by DeepSeek-V4-Flash-reasoning
    <explainer>/check/L<layer>/feature<ID>.json  the same explanations scored by DeepSeek-V4-Flash

Run from anywhere: python compare_explainers.py
"""
import glob
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"DeepSeek-V4-Flash-reasoning": "deepseek", "Kimi-K3": "kimi", "Qwen3.8-27B": "qwen"}
LAYERS = (0, 7, 15)
STOPWORDS = set("the a an of and or to in for with on detector feature token tokens word words".split())


def load(arm: str, check: bool) -> dict:
    """{(layer, latent): (detection, fuzzing or None, explanation seconds, label)}"""
    out = {}
    for layer in LAYERS:
        folder = os.path.join(HERE, arm, "check" if check else "", f"L{layer}")
        for path in glob.glob(os.path.join(folder, "feature*.json")):
            d = json.load(open(path))
            fz = d.get("fuzzing") or {}
            fuzz = fz.get("balanced_acc")
            if fuzz is None or (isinstance(fuzz, float) and math.isnan(fuzz)) or not fz.get("n_pos"):
                fuzz = None  # no strong-firing pairs to build fuzzing probes from
            out[(layer, int(d["feature_id"]))] = (d["detection"]["balanced_acc"], fuzz, d.get("elapsed_s"), d.get("label") or "")
    return out


def both(row):
    return None if row[1] is None else min(row[0], row[1])


def main():
    data = {name: (load(arm, False), load(arm, True)) for name, arm in ARMS.items()}
    keys = sorted(set.intersection(*[set(v[0]) & set(v[1]) for v in data.values()]))
    print(f"{len(keys)} latents explained by all three models and scored by both examiners\n")

    for title, idx in (("examiner: DeepSeek-V4-Flash-reasoning", 0), ("examiner: DeepSeek-V4-Flash (wrote none of the explanations)", 1)):
        print(title)
        print(f"  {'explainer':<30}{'detection':>10}{'fuzzing':>9}{'min':>8}{'both>=0.7':>11}{'expl. time':>12}")
        for name, sets in data.items():
            rows = [sets[idx][k] for k in keys]
            fuzz = [r[1] for r in rows if r[1] is not None]
            mins = [both(r) for r in rows if both(r) is not None]
            secs = [sets[0][k][2] for k in keys if sets[0][k][2]]  # time is a property of the explanation run
            print(f"  {name:<30}{statistics.mean(r[0] for r in rows):>10.3f}{statistics.mean(fuzz):>9.3f}"
                  f"{statistics.mean(mins):>8.3f}{sum(m >= 0.7 for m in mins):>7}/{len(keys):<3}{statistics.mean(secs):>10.0f} s")
        print()

    names = list(data)
    print("paired per-latent differences of min(detection, fuzzing), external examiner")
    for a, b in ((names[0], names[1]), (names[0], names[2]), (names[2], names[1])):
        diffs = [both(data[a][1][k]) - both(data[b][1][k]) for k in keys
                 if both(data[a][1][k]) is not None and both(data[b][1][k]) is not None]
        mean, se = statistics.mean(diffs), statistics.stdev(diffs) / math.sqrt(len(diffs))
        print(f"  {a} - {b}: {mean:+.3f}  95% CI [{mean - 1.96 * se:+.3f}, {mean + 1.96 * se:+.3f}]  "
              f"first is better on {sum(d > 0.001 for d in diffs)}/{len(diffs)}")

    print("\nper layer, external examiner, min(detection, fuzzing)")
    for name, sets in data.items():
        cells = []
        for layer in LAYERS:
            vals = [both(sets[1][k]) for k in keys if k[0] == layer and both(sets[1][k]) is not None]
            cells.append(f"L{layer} {statistics.mean(vals):.3f}")
        print(f"  {name:<30}" + "   ".join(cells))

    print("\nhow often two explainers' labels share a content word")
    def words(label):
        return {w.strip(",.\"'()").lower() for w in label.split()} - STOPWORDS
    for a, b in ((names[0], names[1]), (names[0], names[2]), (names[2], names[1])):
        hits = sum(bool(words(data[a][0][k][3]) & words(data[b][0][k][3])) for k in keys)
        print(f"  {a} vs {b}: {hits}/{len(keys)} = {hits / len(keys):.0%}")


if __name__ == "__main__":
    main()
