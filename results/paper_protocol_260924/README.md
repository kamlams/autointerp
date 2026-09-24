# The 2026-09-20 protocol on the latents of 2026-09-24: OLMoE layers 1 and 15, Qwen3-30B-A3B layer 24

**What this is.** The protocol of `../paper_protocol_260920/` (the published explainer unchanged,
the single-mark fuzzing exam with type-A and type-B negatives) applied to the latents the
collaborator listed on 2026-09-24 (`features.json`): latents tied to one expert of an MoE layer,
split by the sign of their effect on that expert. One folder per base model.

Reproduce every number below: `python results/paper_protocol_260924/analyze.py`.

## Setting

| | `olmoe-1b-7b-0924/` | `qwen3-30b-a3b/` |
|---|---|---|
| model | allenai/OLMoE-1B-7B-0924 | Qwen/Qwen3-30B-A3B |
| SAE (TopK, k = 32, 32768 latents, output of `post_attention_layernorm`) | `OLMoE_moein_unitnormFalse_lrwarmup0_lr0.001_seed0_sample200M` | `Qwen3_moein_unitnormFalse_lrwarmup0_lr0.001_seed0_sample200M` |
| latents | L1, expert 9: 20 positive + 9 negative; L15, expert 39: 32 positive + 23 negative | L24, expert 9: 10 promotive + 10 suppressive |
| corpus | SmolLM2 shard 84, 825 x 16 x 128 = 1,689,600 tokens (the corpus of 09-20) | the same 6,600 documents, tokenized with the Qwen3 tokenizer: 1,689,600 tokens |
| SAE fit on this corpus | not recorded (extracted before the check existed) | FVU 0.498 on the first 8 batches; 32,481 of 32,768 latents live |

Every listed latent fires on the corpus.

## Differences from the 2026-09-20 run

| | 2026-09-20 | here |
|---|---|---|
| explainers | DeepSeek-V4-Flash, DeepSeek-V4-Flash-reasoning, Kimi-K3, Qwen3.8-27B | Kimi-K3 (primary), Qwen3.8-27B, DeepSeek-V4-Flash-reasoning |
| explainer budget | 32768 tokens for models named `*reasoning*`, 4096 for the rest (this cut off Qwen3.8-27B) | 32768 for every explainer (`--explain-max-tokens`); the client doubles it up to 131072 when a reply ends for length |
| empty replies | not retried | each explanation run makes two passes; the second retries the features that came back empty |
| a model other than OLMoE | - | `extract_features.py --model Qwen/Qwen3-30B-A3B` |

Everything else is identical: 40 random activating 32-token windows per latent with the same seed
for every explainer, delphi's prompt, 10 test windows per decile of the window maximum, one marked
token per probe, examiner DeepSeek-V4-Flash-reasoning, five probes per query.

## Results

Balanced accuracy on the single-mark fuzzing exam (chance 0.5); TPR on the positives, TNR on the
type-A (silent window) and type-B (silent token of a firing window) negatives.

| model / layer | group | explainer | n | BA | TPR | TNR A | TNR B |
|---|---|---|---|---|---|---|---|
| OLMoE L1 | expert 9 positive | Kimi-K3 | 20 | 0.803 | 0.628 | 0.979 | 0.975 |
| | | Qwen3.8-27B | 20 | 0.788 | 0.716 | 0.872 | 0.849 |
| | | DeepSeek-V4-Flash-reasoning | 20 | 0.769 | 0.744 | 0.801 | 0.788 |
| | expert 9 negative | Kimi-K3 | 9 | 0.912 | 0.839 | 0.982 | 0.986 |
| | | Qwen3.8-27B | 9 | 0.912 | 0.854 | 0.976 | 0.965 |
| | | DeepSeek-V4-Flash-reasoning | 9 | 0.899 | 0.878 | 0.916 | 0.923 |
| OLMoE L15 | expert 39 positive | Kimi-K3 | 32 | 0.593 | 0.263 | 0.966 | 0.881 |
| | | Qwen3.8-27B | 32 | 0.594 | 0.395 | 0.842 | 0.743 |
| | | DeepSeek-V4-Flash-reasoning | 32 | 0.558 | 0.546 | 0.590 | 0.549 |
| | expert 39 negative | Kimi-K3 | 23 | 0.793 | 0.661 | 0.974 | 0.876 |
| | | Qwen3.8-27B | 23 | 0.814 | 0.756 | 0.937 | 0.809 |
| | | DeepSeek-V4-Flash-reasoning | 23 | 0.798 | 0.785 | 0.857 | 0.764 |
| Qwen3 L24 | expert 9 promotive | Kimi-K3 | 10 | 0.719 | 0.496 | 0.961 | 0.925 |
| | | Qwen3.8-27B | 9 | 0.747 | 0.560 | 0.970 | 0.900 |
| | | DeepSeek-V4-Flash-reasoning | 10 | 0.741 | 0.585 | 0.908 | 0.885 |
| | expert 9 suppressive | Kimi-K3 | 10 | 0.744 | 0.690 | 0.934 | 0.660 |
| | | Qwen3.8-27B | 10 | 0.762 | 0.814 | 0.872 | 0.550 |
| | | DeepSeek-V4-Flash-reasoning | 10 | 0.684 | 0.730 | 0.779 | 0.500 |

Per layer, paired on the latents all three explainers described (BA, difference from Kimi-K3 with
bootstrap 95% CI):

| model / layer | n | Kimi-K3 | Qwen3.8-27B | DeepSeek-V4-Flash-reasoning |
|---|---|---|---|---|
| OLMoE L1 | 29 | 0.837 | 0.827 (-0.010 [-0.036, +0.018]) | 0.809 (-0.027 [-0.051, -0.004]) |
| OLMoE L15 | 55 | 0.677 | 0.686 (+0.009 [-0.006, +0.025]) | 0.658 (-0.019 [-0.035, -0.001]) |
| Qwen3 L24 | 19 | 0.732 | 0.755 (+0.023 [-0.001, +0.049]) | 0.711 (-0.021 [-0.072, +0.019]) |

As on 2026-09-20, the reasoning explainer writes the broadest explanations (highest TPR, lowest
TNR) and is the only one below Kimi-K3 with a CI excluding zero (OLMoE layers). The weakest group
is OLMoE L15 expert 39 positive: every explainer misses most of the strongest firings (Kimi-K3
TPR 0.263).

## Coverage and known issues

- **Missing explanations.** One: Qwen3.8-27B on Qwen3 L24/15048. Both passes ended after 18,108
  tokens without an answer, the reasoning degenerating into a repeated ", is"
  (`qwen3-30b-a3b/logs/empty_replies.jsonl`). Every other latent has an explanation from every
  explainer; the second pass recovered all other empty replies.
- **Latents with few activating windows** get fewer windows than 40 + 100, by the rule of the
  09-20 code (explainer windows = min(40, windows / 3), test windows 10 per decile of what is left):

  | latent | activating windows | explainer windows | test windows (positives / A / B) |
  |---|---|---|---|
  | OLMoE L1/15468 | 129 | 40 | 80 / 40 / 40 |
  | OLMoE L15/5165 | 126 | 40 | 80 / 40 / 40 |
  | Qwen3 L24/20369 | 130 | 40 | 90 / 45 / 40 |
  | Qwen3 L24/754 | 107 | 35 | 70 / 35 / 35 |
  | Qwen3 L24/23481 | 61 | 20 | 40 / 20 / 20 |
  | Qwen3 L24/8006 | 57 | 19 | 30 / 15 / 15 |
  | Qwen3 L24/5492 | 52 | 17 | 30 / 15 / 15 |

  Four of the ten Qwen3 suppressive latents are among them; their scores rest on 30 to 90 probes.
- **Qwen3 SAE fit.** FVU 0.498 is at the upper end of the range seen for a matching SAE (0.02-0.5;
  a wrong model or side gives values far above 1). The base model was confirmed by the collaborator
  as Qwen3-30B-A3B (not -Base); the fit was not investigated further.

## Layout

| path | contents |
|---|---|
| `features.json` | the latent lists, by model, layer and group |
| `<model>/run.sh` | how the run was made (extraction for Qwen3, explanations, scoring) |
| `<model>/explanations/<explainer>/L*/` | explanations, one JSON per latent, and `run_config.json` |
| `<model>/scores/<explainer>/L*/` | single-mark fuzzing scores (examiner DeepSeek-V4-Flash-reasoning) |
| `<model>/sampling/L*.json` | the extraction record (firing counts, max activation, SAE path redacted) |
| `<model>/logs/empty_replies.jsonl` | the logged calls behind every missing explanation |
| `<model>/MANIFEST.json` | every file's origin in `outputs/` and sha256 |

Raw run directories, including the full request logs, stay in
`outputs/paper260924_olmoe-1b-7b-0924/` and `outputs/paper260924_qwen3-30b-a3b/` (gitignored).
