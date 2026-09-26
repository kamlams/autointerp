#!/usr/bin/env python
"""Launch the shard-83 single-mark exam for every (explainer, set, layer, side) unit, Kimi-K3 first,
with at most BUDGET examiner calls in flight (feature workers x examiner workers per process).

    python results/exam_shard83_260926/run_exam.py [--budget 60]

Run from the repository root (run.sh does). Each unit is one run_paper_protocol.py process writing to
outputs/exam_shard83/<side>/<model>/<set>/<explainer>/<layer>/; a latent whose JSON exists is skipped,
so the launcher can be restarted. On 2026-09-26 about 150 calls in flight overloaded the shared
gateway (calls per minute fell from about 300 to about 115), while 50-60 kept it near 200-300.
"""
import argparse
import os
import subprocess
import sys
import time

EXAMINER = "deepseek-ai/DeepSeek-V4-Flash-reasoning"
EXPLAINERS = ["kimi-k3", "qwen3.8-27b", "deepseek-v4-flash-reasoning"]
# model, set, layer, shard-84 data directory (its _shard83 twin holds the exam), latents scored at once
UNITS = [("olmoe-1b-7b-0924", "active-on-to", "L0", "moein_L0_test32b", 4),
         ("olmoe-1b-7b-0924", "active-on-to", "L7", "moein_L7_test32b", 4),
         ("olmoe-1b-7b-0924", "active-on-to", "L15", "moein_L15_test32b", 4),
         ("olmoe-1b-7b-0924", "expert-linked", "L1", "moein_L1_260924", 4),
         ("olmoe-1b-7b-0924", "expert-linked", "L15", "moein_L15_260924", 7),
         ("qwen3-30b-a3b", "expert-linked", "L24", "qwen3_moein_L24_260926", 3)]
WORKERS = 2  # examiner calls per latent at once


def explanations(side: str, model: str, lset: str, explainer: str, layer: str) -> str:
    if side == "own":
        return f"results/own_pipeline_260926/{model}/{lset}/explanations-and-scores/{explainer}/{layer}"
    if lset == "active-on-to":
        return f"results/paper_protocol_260920/explanations/{explainer}/{layer}"
    return f"results/paper_protocol_260924/{model}/explanations/{explainer}/{layer}"


def jobs():
    for explainer in EXPLAINERS:
        for model, lset, layer, data, fw in UNITS:
            for side in ("own", "delphi"):
                src = explanations(side, model, lset, explainer, layer)
                out = f"outputs/exam_shard83/{side}/{model}/{lset}/{explainer}/{layer}"
                yield dict(name=f"{side}/{model}/{lset}/{explainer}/{layer}", src=src, out=out,
                           ready=f"outputs/feature-data/{data}_shard83/sampling.json",
                           calls=fw * WORKERS, cmd=[sys.executable, "run_paper_protocol.py",
                                                    "--data-dir", f"outputs/feature-data/{data}_shard83",
                                                    "--arm", "S1", "--explanations", src,
                                                    "--scorer-model", EXAMINER, "--scorers", "fuzz",
                                                    "--feature-workers", str(fw), "--workers", str(WORKERS),
                                                    "--out", out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60, help="examiner calls in flight, all processes together")
    args = ap.parse_args()
    todo, running = [], []
    for j in jobs():
        if not os.path.isdir(j["src"]):
            print(f"{time.strftime('%F %T')} skip {j['name']}: no explanations at {j['src']}", flush=True)
            continue
        todo.append(j)
    os.makedirs("outputs/exam_shard83/logs", exist_ok=True)
    while todo or running:
        for j in [j for j in running if j["proc"].poll() is not None]:
            running.remove(j)
            n = len([f for f in os.listdir(j["out"]) if f.startswith("feature")]) if os.path.isdir(j["out"]) else 0
            print(f"{time.strftime('%F %T')} done {j['name']} (exit {j['proc'].returncode}, {n} latents)", flush=True)
        load = sum(j["calls"] for j in running)
        # in priority order (Kimi-K3 first). A unit whose shard-83 data is still being extracted keeps its
        # share of the budget reserved, so lower-priority units cannot take it in the meantime.
        reserved = 0
        for j in list(todo):
            if not os.path.exists(j["ready"]):
                reserved += j["calls"]
                continue
            if running and load + reserved + j["calls"] > args.budget:
                break
            todo.remove(j)
            os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
            log = open(f"outputs/exam_shard83/logs/{j['name'].replace('/', '_')}.txt", "a")
            j["proc"] = subprocess.Popen(j["cmd"], stdout=log, stderr=subprocess.STDOUT)
            running.append(j)
            load += j["calls"]
            print(f"{time.strftime('%F %T')} start {j['name']} ({j['calls']} calls; {load} in flight)", flush=True)
        time.sleep(15)
    print(f"{time.strftime('%F %T')} all units done", flush=True)


if __name__ == "__main__":
    main()
