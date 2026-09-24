# The published protocol with the fewest changes that make it usable here (2026-09-20 .. 09-21)

**What this is.** The automatic-interpretation protocol of Paulo et al., *Automatically
Interpreting Millions of Features in Large Language Models* (ICML 2025), run as its reference
implementation (EleutherAI/delphi) runs it, with **as few modifications as possible**. The
explainer side is the published protocol unchanged. The evaluation keeps the published fuzzing
task and changes only how probes are marked and where negatives come from, because measurements
on these features showed the published construction scores something other than the
explanation (below). Every change is listed, with the measurement that forced it.

This is the result set the paper section in `../../../paper/` reports. It supersedes, for that
purpose, the earlier run in `../feature_test32b_260918/`, which used this project's own pipeline
(statistics card, 24 snippets) rather than the published protocol.

Reproduce every number below: `python results/paper_protocol_260920/analyze.py`.

## Setting

- Model OLMoE-1B-7B-0924; TopK SAEs (32,768 latents, k = 32) on the input of the MoE blocks
  (`moein`, `post_attention_layernorm` output), collaborator run
  `OLMoE_moein_unitnormFalse_lrwarmup0_lr0.001_seed0_sample200M`.
- 96 features: the 32 active on ` to` in "When Mary and John went to the store" at each of layers
  0, 7 and 15 (`feature_to_test.txt` of 2026-09-18).
- Reference corpus: 1,689,600 tokens (825 x 16 x 128), the same corpus as
  `../feature_test32b_260918/`.

## What is kept from the published protocol

| | published protocol (delphi) | here |
|---|---|---|
| unit | non-overlapping 32-token window; activating if the feature fires anywhere in it | same |
| explainer evidence | 40 activating windows sampled uniformly at random | same |
| highlighting | tokens strictly above 0.3 x the window's own maximum wrapped in `<< >>` | same |
| activation values | listed after each window, 0-10, scaled by the feature's corpus maximum | same |
| explainer prompt | delphi's system prompt and three few-shot examples, no chain of thought | same, verbatim (`autointerp_pipeline/delphi_prompts/`) |
| test set | 100 activating windows, ten per decile of the window maximum | same |
| fuzzing prompt | delphi's instructions and three few-shot exchanges, five probes per query | same, verbatim |
| metric | accuracy | balanced accuracy (equal to accuracy when classes are balanced) |

The 0.3 threshold, the fixed fake-mark count and the 50/50 test size are defaults of the delphi
code; the paper text does not state them. Where the paper and the code differ we follow the code,
except for the test size, where we follow the paper (100 + 100).

## What changed, and why

| # | change | published | here | measured reason (on these 96 features) |
|---|---|---|---|---|
| 1 | scorers reported | detection and fuzzing | fuzzing only | detection scores 0.580 against the paper's 0.74; the gap is all on the negatives (TNR 0.524) and tracks how common the trigger token is (Spearman 0.40): whether a feature fires *anywhere* in 32 tokens cannot separate a comma feature from a passage with commas |
| 2 | marks per probe | every token above 0.3 x the window max; negatives get a fixed count at fixed positions | exactly one, positives and negatives alike | the question "is every mark right" is a conjunction of per-token judgements: TPR 0.635 at one mark, 0.336 above six (Spearman -0.50), and dense features carry 7.8 marks per window against 1.8 for sparse ones; negatives also carry more marks (5.58 vs 4.30 in all 95 features), and matching the counts alone costs 0.049 accuracy, so the examiner follows the count |
| 3 | negative probes | a window where the feature never fires | half such windows (**A**), half a silent token of a window where it does fire (**B**) | tests localisation inside relevant text as well as rejection of unrelated text; reported separately |

Smaller, declared departures: explainer and test windows are disjoint (delphi may reuse a window);
the marked positive token is the window's strongest; a feature with no silent window is recorded
as not scoreable rather than scored (L0/2565 fires in 52,798 of 52,800 windows); explainers and the
examiner are different models (the paper used one model for both).

## Models

- Explainers: DeepSeek-V4-Flash, DeepSeek-V4-Flash-reasoning, Kimi-K3, Qwen3.8-27B. Each wrote its
  explanations once, from the same 40 windows per feature (same seed).
- Examiner: DeepSeek-V4-Flash-reasoning, for every explainer.

## Results

**1. The unmodified protocol** (explainer and examiner DeepSeek-V4-Flash, 95 features):
detection 0.580, fuzzing 0.665 (paper, Gemma-2-9b SAE: 0.74 / 0.76). Fuzzing TPR 0.516 against
TNR 0.813.

**2. What the modification changes**, holding explanations (DeepSeek-V4-Flash) and examiner
(DeepSeek-V4-Flash-reasoning) fixed, 95 features:

| exam | BA | TPR | TNR |
|---|---|---|---|
| published fuzzing (all marks) | 0.698 | 0.521 | 0.876 |
| single mark | 0.680 | 0.634 | 0.733 (A) / 0.717 (B) |

The total moves by -0.019 [-0.032, -0.006]; its composition changes: the published number rested
on negatives that were easy for the wrong reason.

**3. Four explainers on the single-mark exam** (balanced accuracy, each explainer over every
feature it described):

| explainer | n | BA | TPR | TNR A | TNR B | L0 | L7 | L15 |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 95 | 0.680 | 0.634 | 0.733 | 0.717 | 0.656 | 0.692 | 0.690 |
| DeepSeek-V4-Flash-reasoning | 94 | 0.654 | 0.713 | 0.608 | 0.584 | 0.621 | 0.681 | 0.660 |
| Kimi-K3 | 95 | 0.684 | 0.610 | 0.779 | 0.737 | 0.679 | 0.700 | 0.672 |
| Qwen3.8-27B | 90 | 0.693 | 0.643 | 0.760 | 0.724 | 0.654 | 0.722 | 0.700 |

Paired on the 89 features all four described, against DeepSeek-V4-Flash (bootstrap 95% CI):
reasoning -0.022 [-0.035, -0.010], Kimi +0.005 [-0.011, +0.022], Qwen +0.011 [-0.003, +0.026].
Only the reasoning explainer differs; it has the best TPR and the worst TNR, i.e. broad
explanations. Floor (each feature scored with another feature's explanation, 88 features):
BA 0.578, TPR 0.427, TNR A 0.739, TNR B 0.717.

## Coverage and known defects

| explainer | described | not scoreable | no final reply |
|---|---|---|---|
| DeepSeek-V4-Flash | 95 / 96 | L0/2565 | - |
| DeepSeek-V4-Flash-reasoning | 94 / 96 | L0/2565 | L7/19319 |
| Kimi-K3 | 95 / 96 | L0/2565 | - |
| Qwen3.8-27B | 90 / 96 | L0/2565 | L0/856, L0/32469, L7/10602, L15/907, L15/21663 |

The log lines showing the empty replies are in `logs/empty_replies.jsonl`.

- **Qwen3.8-27B ran with a 4,096-token budget.** `run_paper_protocol.py` gives 32,768 tokens only
  to models whose name contains "reasoning"; Qwen3.8-27B reasons without that word in its name.
  225 of its 265 explainer calls hit the cap and were retried with a doubled budget, which the
  client does only when no text came back at all. For L7/29038 about 39 tokens of answer came
  back, so the reply was accepted cut off mid-sentence ("...between an agent and a") and scored
  as it stands. No other explanation from any model is cut off.
- Type-B negatives are skipped without refill when a window has no silent token: 41 to 50 per
  feature instead of 50.

## Layout

| folder | contents | run name in routerinterp |
|---|---|---|
| `explanations/deepseek-v4-flash/` | explanations; also that model's **unmodified-protocol** scores (detection + fuzzing, examiner DeepSeek-V4-Flash) | `P0` |
| `explanations/deepseek-v4-flash-reasoning/` | explanations (plus scores under the unmodified protocol, examiner DeepSeek-V4-Flash) | `A7` |
| `explanations/kimi-k3/`, `explanations/qwen3.8-27b/` | explanations only | `EXPL_kimi`, `EXPL_qwen` |
| `scores/<explainer>/` | single-mark fuzzing scores, examiner DeepSeek-V4-Flash-reasoning | `S1_flash`, `S1_reasoning`, `S1_kimi`, `S1_qwen` |
| `scores/floor-shuffled-explanation/` | the floor | `S1_neg` |
| `baseline-published-exam/deepseek-v4-flash/` | DeepSeek-V4-Flash explanations on the published fuzzing exam, examiner DeepSeek-V4-Flash-reasoning | `G_P0_flash` |
| `logs/empty_replies.jsonl` | the 47 log lines behind "no final reply" | from `calls.jsonl` |
| `MANIFEST.json` | every file's origin and sha256 | |

`P0` and `A7` are experiment-arm codes ("paper baseline", "ablation 7"), not layer numbers.

The runs were made in the `routerinterp` development repository between 2026-09-20 and 09-21
and copied here byte for byte on 2026-09-25 (`MANIFEST.json`). Paths inside each
`run_config.json` are relative to that repository. The full request/response logs
(`calls.jsonl`, about 200 MB) were not copied; they remain in `routerinterp/outputs/paper/`.

## Re-run

```bash
# explanations, one model (no scoring)
python run_paper_protocol.py --data-dir outputs/feature-data/moein_L7_test32b --arm P0 \
    --explainer-model Qwen/Qwen3.8-27B --scorers none --out outputs/paper/explain_qwen/L7
# single-mark exam on those explanations
python run_paper_protocol.py --data-dir outputs/feature-data/moein_L7_test32b --arm S1 \
    --explanations outputs/paper/explain_qwen/L7 \
    --scorer-model deepseek-ai/DeepSeek-V4-Flash-reasoning --scorers fuzz --out outputs/paper/score_qwen/L7
```

`--arm P0` is the published explainer; `--arm S1` the single-mark exam; `--arm S1_NEG` its floor.
Activation data (`outputs/feature-data/moein_L*_test32b/`) is made by `extract_features.py`.
