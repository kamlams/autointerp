# Three explainers on the same 96 latents (2026-09-18 batch)

The same 96 latents (layers 0, 7, 15 of the `moein` SAE) explained three times, by
`DeepSeek-V4-Flash-reasoning` (the original run), `moonshotai/Kimi-K3` and `Qwen/Qwen3.8-27B`.
Everything else is identical: same activations, same corpus, same sampler and seed. Each set of
explanations is then scored twice — once by `DeepSeek-V4-Flash-reasoning` (the examiner the
original run used) and once by `DeepSeek-V4-Flash` (non-reasoning), which wrote none of the
explanations and so is external to all three arms.

Run on 2026-09-18 23:20 - 2026-09-19 16:24 across two allocations; `run_explainers.sh` started it,
`resume_explainers.sh` finished it after the first allocation ended. 96/96 explained in every arm.

## Scores (96 features)

| examiner | explainer | detection | fuzzing | min(det,fuzz) | both >= 0.7 | explanation time |
|---|---|---|---|---|---|---|
| DeepSeek-reasoning | DeepSeek-reasoning | 0.710 | 0.675 | 0.640 | 30/96 | 159 s |
| DeepSeek-reasoning | Kimi-K3 | 0.694 | 0.668 | 0.632 | 29/96 | 236 s |
| DeepSeek-reasoning | Qwen3.8-27B | 0.708 | 0.664 | 0.636 | 33/96 | 258 s |
| DeepSeek-Flash (external) | DeepSeek-reasoning | 0.632 | 0.630 | 0.582 | 17/96 | 159 s |
| DeepSeek-Flash (external) | Kimi-K3 | 0.624 | 0.598 | 0.566 | 17/96 | 236 s |
| DeepSeek-Flash (external) | Qwen3.8-27B | 0.627 | 0.612 | 0.569 | 16/96 | 258 s |

Paired per-feature differences under the external examiner, with 95% confidence intervals:

| comparison | mean difference | 95% CI | wins |
|---|---|---|---|
| DeepSeek - Kimi | +0.016 | [-0.003, +0.035] | 43/96 |
| DeepSeek - Qwen | +0.013 | [-0.005, +0.032] | 49/96 |
| Qwen - Kimi | +0.003 | [-0.017, +0.023] | 46/96 |

Per layer (external examiner, min(det,fuzz)): DeepSeek 0.586 / 0.567 / 0.593, Kimi 0.572 / 0.551 /
0.576, Qwen 0.564 / 0.607 / 0.535 for layers 0 / 7 / 15.

## Reading

- **The three explainers are indistinguishable on this task.** Every confidence interval crosses
  zero, and each model wins about half the features. An earlier 6-latent pilot had suggested a DeepSeek advantage of
  +0.07; on 96 latents it is +0.013 to +0.016 and not significant.
- **Cost is the only clear difference.** DeepSeek explains a feature in 159 s against 236 s (Kimi)
  and 258 s (Qwen): 1.5-1.6x more compute for no measurable gain.
- The examiner still matters far more than the explainer: moving from the reasoning examiner to the
  non-reasoning one costs every arm 0.06-0.07, four times the largest gap between explainers.
- The three arms label the same feature differently about half the time (a shared content word in
  46-49% of features), yet score the same. Different wordings of the same feature are scored
  equally well, which is what one wants from the metric, but it also means a single explanation
  should not be read as *the* description of a feature.

Every number above is reproduced by `python compare_explainers.py` in this directory.

Per-feature data: `<explainer>/L<layer>/` (explanations plus reasoning-examiner scores),
`<explainer>/check/L<layer>/` (the external-examiner scores), `<explainer>/features_L<layer>.md`
(readable tables).
