# This repository's pipeline with a 30-word explanation, on the latents of both published-protocol runs (2026-09-26)

**What this is.** The latents of `../paper_protocol_260920/` and `../paper_protocol_260924/`,
explained and scored again with this repository's own pipeline (`run_autointerp.py`: statistics
card + 24 snippets, `LABEL` + `EXPLANATION`, detection + fuzzing), with one change: the
`EXPLANATION` is limited to **30 words**. One folder per base model.

Reproduce every number below: `python results/own_pipeline_260926/analyze.py`.

## Setting

| folder | model | SAE (TopK, k = 32, 32,768 latents, output of `post_attention_layernorm`) | latents |
|---|---|---|---|
| `olmoe-1b-7b-0924/active-on-to/` | allenai/OLMoE-1B-7B-0924 | `OLMoE_moein_unitnormFalse_lrwarmup0_lr0.001_seed0_sample200M` | L0, L7, L15: 32 each, the latents active on " to" in "When Mary and John went to the store" (the 96 of `paper_protocol_260920/` and `feature_test32b_260918/`) |
| `olmoe-1b-7b-0924/expert-linked/` | same | same | L1 expert 9: 20 positive + 9 negative; L15 expert 39: 32 positive + 23 negative |
| `qwen3-30b-a3b/expert-linked/` | Qwen/Qwen3-30B-A3B | `Qwen3_moein_unitnormFalse_lrwarmup0_lr0.001_seed0_sample200M` | L24 expert 9: 10 promotive + 10 suppressive, with the collaborator's correction of 2026-09-26 (**8066** replaces 8006) |

`features.json` lists every latent by model, set, layer and group. Corpus: SmolLM2 shard 84,
825 x 16 x 128 = 1,689,600 tokens, tokenized per model. The Qwen3 activations were extracted
again with the corrected list (`extract_features.py --model Qwen/Qwen3-30B-A3B`, one GPU, about
five minutes; SAE FVU 0.498 on the first 8 batches, as before).

- Explainers: Kimi-K3 (run first), Qwen3.8-27B, DeepSeek-V4-Flash-reasoning; 32,768-token budget.
- Examiner: DeepSeek-V4-Flash-reasoning. Every explanation is then re-scored by DeepSeek-V4-Flash
  (`rescored-by-deepseek-v4-flash/`), as in `../feature_test32b_260918/by_explainer/`.

## What changed from the pipeline run of 2026-09-18

Everything is as in `../feature_test32b_260918/` (same sampler, statistics card, prompts, seed,
scorers and examiners) except the explanation length:

- the explainer prompt's last line asks for `EXPLANATION: <at most 30 words: ...>` instead of
  "one to three sentences" (`run_autointerp.py --max-explanation-words 30`);
- an `EXPLANATION` still over 30 words gets one follow-up turn in the same conversation asking the
  explainer to rewrite it in at most 30 words; the record keeps both versions
  (`explanation_first`, `explanation`) and the word count (`explanation_words`, `over_limit`).

The examiner sees only the explanation, never the label.

## Explanation length

| explainer | mean words | shortened by the follow-up | still over 30 after it |
|---|---|---|---|
| Kimi-K3 | 28.1-28.6 per layer | 66 of 200 | 5: L7/7959 (32), L1/22703 and L1/846 (up to 33), L15/17958 and L15/30290 (31, expert-linked) |
| Qwen3.8-27B | 22.0-24.1 | 0 | 0 |
| DeepSeek-V4-Flash-reasoning | 25.5-26.4 | 0 | 0 |

## Results

Balanced accuracy (chance 0.5), examiner **DeepSeek-V4-Flash-reasoning**, detection / fuzzing:

| set | layer / group | n | Kimi-K3 | Qwen3.8-27B | DeepSeek-V4-Flash-reasoning |
|---|---|---|---|---|---|
| OLMoE, active on " to" | L0 | 32 | 0.713 / 0.671 | 0.698 / 0.662 | 0.699 / 0.682 |
| | L7 | 32 | 0.698 / 0.670 | 0.726 / 0.685 (31) | 0.690 / 0.658 |
| | L15 | 32 | 0.684 / 0.622 | 0.696 / 0.640 | 0.684 / 0.653 |
| OLMoE, expert-linked | L1 expert 9 positive | 20 | 0.789 / 0.829 | 0.782 / 0.840 | 0.785 / 0.826 |
| | L1 expert 9 negative | 9 | 0.817 / 0.893 | 0.808 / 0.957 | 0.819 / 0.942 |
| | L15 expert 39 positive | 32 | 0.769 / 0.595 | 0.821 / 0.577 (31) | 0.792 / 0.583 |
| | L15 expert 39 negative | 23 | 0.776 / 0.868 | 0.788 / 0.855 | 0.772 / 0.847 |
| Qwen3-30B-A3B, expert-linked | L24 expert 9 promotive | 10 | 0.762 / 0.826 | 0.760 / 0.844 | 0.770 / 0.803 |
| | L24 expert 9 suppressive | 10 | 0.825 / 0.750 | 0.833 / 0.783 | 0.820 / 0.757 |

The same explanations re-scored by **DeepSeek-V4-Flash**:

| set | layer / group | Kimi-K3 | Qwen3.8-27B | DeepSeek-V4-Flash-reasoning |
|---|---|---|---|---|
| OLMoE, active on " to" | L0 | 0.640 / 0.603 | 0.658 / 0.595 | 0.646 / 0.593 |
| | L7 | 0.611 / 0.604 | 0.648 / 0.636 | 0.625 / 0.613 |
| | L15 | 0.610 / 0.591 | 0.616 / 0.572 | 0.620 / 0.583 |
| OLMoE, expert-linked | L1 positive | 0.728 / 0.702 | 0.705 / 0.684 | 0.726 / 0.695 |
| | L1 negative | 0.797 / 0.865 | 0.753 / 0.887 | 0.792 / 0.874 |
| | L15 positive | 0.663 / 0.580 | 0.675 / 0.560 | 0.688 / 0.574 |
| | L15 negative | 0.767 / 0.778 | 0.757 / 0.755 | 0.747 / 0.801 |
| Qwen3-30B-A3B, expert-linked | L24 promotive | 0.685 / 0.758 | 0.693 / 0.714 | 0.715 / 0.749 |
| | L24 suppressive | 0.800 / 0.690 | 0.810 / 0.699 | 0.790 / 0.675 |

Paired on the latents all three explainers described, against Kimi-K3 (bootstrap 95% CI,
reasoning examiner), every difference in every layer has a CI crossing zero except one: detection
at OLMoE expert-linked L15, Qwen3.8-27B +0.035 [+0.013, +0.057]. No explainer is consistently
better, as on 2026-09-18. The Flash examiner scores every cell lower than the reasoning examiner,
by 0.01 to 0.16.

## Coverage and known issues

- **Missing explanations (2 of 600).** Qwen3.8-27B wrote nothing for OLMoE active-on-to L7/32709
  (both attempts ended without text) and OLMoE expert-linked L15/21329 (no text at 32k, 64k and
  128k tokens, then a second attempt stopped after 30 minutes). The logged calls are in
  `missing_explanations.json`.
- **Fewer than 20 fuzzing pairs.** Fuzzing needs strong firings (at least 30% of the maximum);
  sparse latents have few. 29 of the 200 latents get fewer than 20 pairs (listed by
  `analyze.py`); Qwen3 L24/23481 gets none and has no fuzzing score, L24/5492 three, L24/8066
  twelve. Detection always has its full 20 + 20 probes.
- **How the run went** (`run.sh`). Phase 2 (Qwen3.8-27B and DeepSeek-V4-Flash-reasoning) was
  started while phase 1 was still finishing one slow half (Kimi-K3 on dense layer-0 latents). Two
  latents (expert-linked L15/21638 and L15/22810) were also explained by a parallel helper run for
  Qwen3.8-27B; those copies were discarded and the main run's records kept.

## Layout

| path | contents |
|---|---|
| `features.json` | the latents by model, set, layer and group |
| `<model>/<set>/explanations-and-scores/<explainer>/L*/feature<id>.json` | label, explanation, word count, both scores with every probe, statistics card, explainer prompt; `run_config_<part>.json` per process |
| `<model>/<set>/rescored-by-deepseek-v4-flash/<explainer>/L*/` | the same explanations scored by DeepSeek-V4-Flash |
| `<model>/<set>/sampling/L*.json` | extraction record (firing counts, max activation; SAE path redacted) |
| `missing_explanations.json` | the logged calls behind the two missing explanations |
| `run.sh`, `analyze.py`, `MANIFEST.json` | how the run was made; every number above; every file's origin and sha256 |

Raw run directories with the full request logs stay in `outputs/own260926/` (gitignored).
