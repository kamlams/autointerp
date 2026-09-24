#!/usr/bin/env python
"""Published autointerp protocol (Paulo et al. 2024 / delphi) and its ablation arms.

    # write explanations with one model (no scoring), then score them on the single-mark exam
    python run_paper_protocol.py --data-dir outputs/feature-data/moein_L7_test32b --arm P0 \
        --explainer-model Qwen/Qwen3.8-27B --scorers none --out outputs/paper/explain_qwen/L7
    python run_paper_protocol.py --data-dir outputs/feature-data/moein_L7_test32b --arm S1 \
        --explanations outputs/paper/explain_qwen/L7 --scorer-model deepseek-ai/DeepSeek-V4-Flash-reasoning \
        --scorers fuzz --out outputs/paper/score_qwen/L7
    python run_paper_protocol.py --data-dir ... --arm V8 --explanations results/.../moein_L7_test32b --out ...

An arm is a preset of PaperConfig plus the explainer model. The protocol and its results are
documented in results/paper_protocol_260920/README.md.
Runs resume: a latent whose JSON exists is skipped.
"""
import argparse
import json
import os
import sys
import time

from autointerp_pipeline import paper
from autointerp_pipeline.paper import NotScoreable
from autointerp_pipeline.data import discover_feature_files, load_corpus, load_feature
from autointerp_pipeline.llm import LLMClient, map_concurrent
from autointerp_pipeline.stats import feature_stats_card

FLASH = "deepseek-ai/DeepSeek-V4-Flash"
ARMS = {
    "P0": {},
    "P0b": {"seed": 2},
    "A1": {"train_type": "quantiles"},
    "A2": {"train_type": "top"},
    "A3": {"train_type": "bins_dedup"},
    "A4": {"n_train": 24},
    "A5": {"show_activations": False},
    "A6": {"stats_card": True},
    "A7": {"explainer": FLASH + "-reasoning"},
    "A8": {"drop_pos0": True},
    "NEG": {"reuse": "shifted"},   # another latent's explanation: negative control
    "V8": {"reuse": "same"},       # explanations written under another protocol, scored here
    # explainer x scorer 2 x 2 over "reasoning or not". The explanations are reused, never rewritten,
    # so the scorer is the only thing that changes between P0/A9 and their reasoning-scored twins.
    "A9": {"reuse": "same"},       # A7's reasoning explanations, scored by the reasoning model
    "A10": {"reuse": "same"},      # P0's Flash explanations, scored by the reasoning model
    # Exam built from strong firings only: the same explanations, asked only about windows where
    # the latent genuinely fires rather than where TopK padded it in.
    "HI": {"reuse": "same", "test_high_only": True},
    "HI_V8": {"reuse": "same", "test_high_only": True},   # v8's explanations on the strong-firing exam
    # delphi's exam untouched except that the fake marks are as numerous as the real ones: the
    # one-variable control for whether the examiner follows mark counts instead of the explanation
    "P0M": {"reuse": "same", "match_mark_counts": True},
    # delphi's exam untouched except that the fake marks are scattered instead of pinned: prices
    # the position cue, the larger of the two
    "P0R": {"reuse": "same", "random_mark_pos": True},
    # the floor under the matched-count rule: a wrong explanation on that exam
    "P0M_NEG": {"reuse": "shifted", "match_mark_counts": True},
    # one mark per probe, negatives half unrelated windows and half wrong tokens of firing windows
    "S1": {"reuse": "same", "single_mark": True},
    "S1_NEG": {"reuse": "shifted", "single_mark": True},   # its floor
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--explanations", default=None, help="run directory to take explanations from (NEG, V8)")
    ap.add_argument("--scorer-model", default=FLASH)
    ap.add_argument("--explainer-model", default=None,
                    help="override the arm's explainer; the rest of the arm is unchanged, so several "
                         "models can be compared on identical examples")
    ap.add_argument("--workers", type=int, default=8, help="concurrent scorer calls per latent")
    ap.add_argument("--feature-workers", type=int, default=3, help="latents processed at once")
    ap.add_argument("--limit", type=int, default=0, help="first N latents only (smoke test)")
    ap.add_argument("--explain-max-tokens", type=int, default=32768,
                    help="explainer token budget, reasoning included. The 2026-09-20 run gave 32768 only to "
                         "models named '*reasoning*' and 4096 to the rest, which cut off Qwen3.8-27B")
    ap.add_argument("--scorers", default="detect,fuzz",
                    help="which scorers to run; 'none' writes the explanation and stops, so a later "
                         "run can score it with --explanations")
    args = ap.parse_args()

    arm = dict(ARMS[args.arm])
    explainer_model = args.explainer_model or arm.pop("explainer", FLASH)
    arm.pop("explainer", None)
    reuse = arm.pop("reuse", None)
    assert bool(reuse) == bool(args.explanations), "--explanations is required for NEG/V8 and only for them"
    cfg = paper.PaperConfig(**arm)
    os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.out, "calls.jsonl")
    explainer = LLMClient(model=explainer_model, log_path=log)
    scorer = explainer if args.scorer_model == explainer_model else LLMClient(model=args.scorer_model, log_path=log)

    token_pkl = sorted(f for f in os.listdir(args.data_dir) if f.startswith("token_list"))[-1]
    corpus = load_corpus(os.path.join(args.data_dir, token_pkl))
    paths = discover_feature_files(args.data_dir)[: args.limit or None]
    ids = [load_feature(p).feature_id for p in paths]
    donor = {fid: (ids[(i + 1) % len(ids)] if reuse == "shifted" else fid) for i, fid in enumerate(ids)}

    def one(path):
        feat = load_feature(path)
        target = os.path.join(args.out, f"feature{feat.feature_id}.json")
        if os.path.exists(target):
            return json.load(open(target))
        t0 = time.time()
        try:
            train, test, negatives = paper.sample(feat, corpus, cfg)
            if reuse:
                src = json.load(open(os.path.join(args.explanations, f"feature{donor[feat.feature_id]}.json")))
                expl = {"explanation": src["explanation"], "parsed": True, "raw": None, "prompt": None}
            else:
                card = feature_stats_card(feat, corpus)[0] if cfg.stats_card else None
                expl = paper.explain(explainer, train, cfg, card, max_tokens=args.explain_max_tokens)
            smt = 16384 if "reasoning" in args.scorer_model else 2048
            want = {x.strip() for x in args.scorers.split(",")}
            det = (paper.detection(scorer, expl["explanation"], test, negatives, args.workers, smt)
                   if "detect" in want else {})
            # the strong-firing exam marks tokens by one absolute cut, so true and fake marks are
            # counted the same way and their number no longer betrays the class
            strong_abs = (cfg.strong_floor * paper.build_windows(feat, corpus, cfg)[0].max_act
                          if cfg.test_high_only else None)
            if "fuzz" not in want:
                fuz = {}
            elif cfg.single_mark:
                fuz = paper.fuzzing_single(scorer, expl["explanation"], test, negatives,
                                           paper.build_windows(feat, corpus, cfg), args.workers,
                                           smt, cfg.seed * 100003 + feat.feature_id)
            else:
                fuz = paper.fuzzing(scorer, expl["explanation"], test, negatives, args.workers, smt,
                                    strong_abs, cfg.match_mark_counts, cfg.random_mark_pos)
        except NotScoreable as e:
            # Recorded, not silently dropped: a latent this protocol cannot score is a result.
            with open(target, "w") as f:
                json.dump({"feature_id": feat.feature_id, "arm": args.arm, "not_scoreable": str(e),
                           "detection": {}, "fuzzing": {}}, f, indent=1)
            print(f"  feature {feat.feature_id} NOT SCOREABLE: {e}", flush=True)
            return None
        except Exception as e:  # noqa: BLE001
            print(f"  feature {feat.feature_id} FAILED: {str(e)[:200]}", file=sys.stderr, flush=True)
            return None
        strip = lambda d: {k: v for k, v in d.items() if k not in ("preds", "labels")}
        result = {"feature_id": feat.feature_id, "arm": args.arm, "label": expl["explanation"][:60],
                  "explanation": expl["explanation"], "explanation_parsed": expl["parsed"],
                  "explanation_from": donor[feat.feature_id] if reuse else None,
                  "n_windows": {"train": len(train), "test": len(test), "negatives": len(negatives)},
                  "detection": strip(det), "fuzzing": strip(fuz), "elapsed_s": round(time.time() - t0, 1)}
        with open(target, "w") as f:
            json.dump(result | {"detection_raw": det, "fuzzing_raw": fuz, "explainer_prompt": expl["prompt"],
                                "explainer_reply": expl["raw"]}, f, ensure_ascii=False, indent=1)
        sc = " ".join(f"{k}={v['accuracy']:.2f}" for k, v in (("det", det), ("fuzz", fuz)) if v)
        print(f"  {feat.feature_id}: {sc} ({result['elapsed_s']}s) {expl['explanation'][:90]!r}", flush=True)
        return result

    results = [r for r in map_concurrent(one, paths, args.feature_workers) if r]
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({"arm": args.arm, "config": vars(cfg), "explainer_model": explainer_model,
                   "explain_max_tokens": None if reuse else args.explain_max_tokens,
                   "scorer_model": args.scorer_model, "explanations": args.explanations,
                   "data_dir": os.path.relpath(args.data_dir), "n_done": len(results), "n_features": len(paths),
                   "usage": explainer.usage.summary()}, f, indent=1)
    def mean(k):
        v = [r[k]["accuracy"] for r in results if r[k]]
        return sum(v) / len(v) if v else float("nan")
    print(f"{args.arm} {os.path.basename(args.data_dir)} -> {args.out}: {len(results)}/{len(paths)} "
          f"latents, detection {mean('detection'):.3f}, fuzzing {mean('fuzzing'):.3f}")


if __name__ == "__main__":
    main()
