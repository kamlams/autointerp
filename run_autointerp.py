#!/usr/bin/env python
"""Run the autointerp pipeline over feature activation dumps.

Example:
    python run_autointerp.py --data-dir ../autointerp --model moonshotai/Kimi-K3
    python run_autointerp.py --features ../autointerp/feature28617__to__260904.pt
"""
import argparse
import json
import os
import sys
import time

from autointerp_pipeline.data import discover_feature_files, load_corpus, load_feature
from autointerp_pipeline.explainer import explain
from autointerp_pipeline.llm import LLMClient
from autointerp_pipeline.sampler import SamplerConfig, build_examples
from autointerp_pipeline.scorer import detection, fuzzing

DEFAULT_DATA_DIR = "/hs/work0/home/users/lingqi.zhang/workspace/wengang/autointerp"


def process_feature(path: str, corpus, client: LLMClient, cfg: SamplerConfig,
                    workers: int, out_dir: str, scorer_client: LLMClient | None = None,
                    explain_max_tokens: int = 32768) -> dict:
    t0 = time.time()
    scorer_client = scorer_client or client
    feat = load_feature(path)
    ex = build_examples(feat, corpus, cfg)
    expl = explain(client, ex, cfg, max_tokens=explain_max_tokens)
    det = detection(scorer_client, expl["explanation"], ex, workers=workers, seed=cfg.seed)
    fuz = fuzzing(scorer_client, expl["explanation"], ex, workers=workers, seed=cfg.seed)
    result = {
        "feature_id": feat.feature_id,
        "human_label": feat.human_label,
        "file": os.path.basename(path),
        "stats": ex.stats,
        "label": expl["label"],
        "explanation": expl["explanation"],
        "detection": {k: v for k, v in det.items() if k not in ("preds", "labels")},
        "fuzzing": {k: v for k, v in fuz.items() if k not in ("preds", "labels")},
        "elapsed_s": round(time.time() - t0, 1),
        "examples_shown": [s.render(True) for s in ex.explainer[:6]],
        "stats_card": ex.card,
    }
    with open(os.path.join(out_dir, f"feature{feat.feature_id}.json"), "w") as f:
        json.dump(result | {"detection_raw": det, "fuzzing_raw": fuz,
                            "explainer_prompt": expl["prompt"]}, f,
                  ensure_ascii=False, indent=1, default=str)
    return result


def write_report(results: list[dict], out_dir: str, model: str):
    lines = [f"# Autointerp run — model `{model}`", ""]
    lines.append("| feature | human label | density | LLM label | det bal.acc | fuzz bal.acc |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['feature_id']} | {r['human_label']} | {r['stats']['density']:.2%} "
            f"| {r['label']} | {r['detection']['balanced_acc']:.2f} "
            f"| {r['fuzzing']['balanced_acc']:.2f} |")
    lines.append("")
    for r in results:
        lines.append(f"## feature {r['feature_id']} (human: {r['human_label']})")
        lines.append(f"- **LLM label**: {r['label']}")
        lines.append(f"- **explanation**: {r['explanation']}")
        d = r["detection"]
        lines.append(f"- **detection**: bal.acc {d['balanced_acc']:.2f} "
                     f"(TPR {d['tpr']:.2f} / TNR {d['tnr']:.2f}, parse_err {d['n_parse_errors']})")
        lines.append(f"  - TPR by strength decile: "
                     + ", ".join(f"{k}:{v:.2f}" for k, v in d.get("tpr_by_strength", {}).items()))
        f_ = r["fuzzing"]
        lines.append(f"- **fuzzing**: bal.acc {f_['balanced_acc']:.2f} "
                     f"(TPR {f_['tpr']:.2f} / TNR {f_['tnr']:.2f}, parse_err {f_['n_parse_errors']})")
        lines.append(f"- density {r['stats']['density']:.2%}, max_act {r['stats']['max_act']:.2f}, "
                     f"elapsed {r['elapsed_s']}s")
        lines.append("")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--features", nargs="*", help="explicit .pt paths (default: all in data-dir)")
    ap.add_argument("--token-list", default=None, help="token list pkl (default: auto-discover)")
    ap.add_argument("--model", default=os.environ.get("RIVAULT_MODEL", "moonshotai/Kimi-K3"))
    ap.add_argument("--scorer-model", default=None,
                    help="lighter model for detection/fuzzing calls (default: same as --model)")
    ap.add_argument("--out", default=None, help="output dir (default outputs/<ts>_<model>)")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent scorer calls; 12 runs all scoring batches in parallel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-explainer", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=20, help="activating and negative probes each")
    ap.add_argument("--explain-max-tokens", type=int, default=32768,
                    help="completion budget for the explainer call; reasoning models spend "
                         "15-30k tokens thinking on this task")
    args = ap.parse_args()

    token_pkl = args.token_list
    if token_pkl is None:
        cands = [f for f in os.listdir(args.data_dir) if f.startswith("token_list")]
        assert cands, f"no token_list*.pkl in {args.data_dir}"
        token_pkl = os.path.join(args.data_dir, sorted(cands)[-1])
    feature_paths = args.features or discover_feature_files(args.data_dir)
    assert feature_paths, "no feature files found"

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs",
        time.strftime("%y%m%d_%H%M%S") + "_" + args.model.split("/")[-1].replace(":", "_"))
    os.makedirs(out_dir, exist_ok=True)

    cfg = SamplerConfig(seed=args.seed, n_explainer=args.n_explainer,
                        n_test_act=args.n_test, n_test_neg=args.n_test)
    client = LLMClient(model=args.model, log_path=os.path.join(out_dir, "calls.jsonl"))
    scorer_client = None
    if args.scorer_model and args.scorer_model != args.model:
        scorer_client = LLMClient(model=args.scorer_model,
                                  log_path=os.path.join(out_dir, "calls.jsonl"))

    print(f"loading corpus {token_pkl} ...", flush=True)
    corpus = load_corpus(token_pkl)
    print(f"corpus: {corpus.n_batches}x{corpus.n_prompts}x{corpus.n_tokens} "
          f"= {corpus.total_tokens} tokens", flush=True)

    results = []
    for path in feature_paths:
        print(f"[{len(results)+1}/{len(feature_paths)}] {os.path.basename(path)}", flush=True)
        try:
            r = process_feature(path, corpus, client, cfg, args.workers, out_dir,
                                scorer_client=scorer_client,
                                explain_max_tokens=args.explain_max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}", file=sys.stderr, flush=True)
            continue
        results.append(r)
        print(f"  -> label={r['label']!r} det={r['detection']['balanced_acc']:.2f} "
              f"fuzz={r['fuzzing']['balanced_acc']:.2f} ({r['elapsed_s']}s)", flush=True)

    write_report(results, out_dir, args.model)
    usage = {"explainer": client.usage.summary()}
    if scorer_client:
        usage["scorer"] = scorer_client.usage.summary()
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({"model": args.model, "scorer_model": args.scorer_model,
                   "sampler": vars(cfg), "features": feature_paths,
                   "usage": usage}, f, indent=1, default=str)
    print(f"\nusage: {usage}")
    print(f"report: {out_dir}/report.md")


if __name__ == "__main__":
    main()
