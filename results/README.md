# Results — full pipeline runs on the accepted protocol

Each directory is a complete record of one end-to-end run (fresh explanations +
detection + fuzzing) of the pipeline at the stated commit: `report.md`, one
`feature{ID}.json` per feature (explanation, scores, per-probe predictions, diagnostics),
`calls.jsonl` (every LLM call: prompt, reply, reasoning excerpt, token usage),
`run_config.json`, `run.log`. Scratch experiments live in the gitignored `outputs/`.

Data: the 5 hand-labeled sample features in `../autointerp` (1.69M-token corpus).
Scores are balanced accuracy (chance = 0.5): detection / fuzzing.

| run | date | pipeline commit | explainer = examiner | 28617 to | 12298 after_a | 25337 preceded_by_a_verb | 14825 ubiquitous | 24792 content (93%) | wall | calls | reasoning tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v8_ds-reasoning` | 2026-09-05 | 95621ad | deepseek-ai/DeepSeek-V4-Flash-reasoning | 0.80 / 1.00 | 0.75 / 0.90 | 0.62 / 0.72 | 0.62 / 0.68 | 0.65 / 0.55 | 505 s | 88 | 280k |
| `v9_ds-flash` | 2026-09-06 | bffb5ce | deepseek-ai/DeepSeek-V4-Flash (no reasoning) | 0.68 / 0.88 | 0.50 / 0.55 | 0.57 / 0.57 | 0.65 / 0.68 | 0.55 / 0.57 | 33 s | 87 | 0 |

Sampler settings shared by all runs (see `run_config.json`): 24 explainer snippets
(8 top + 16 stratified over rank/value bins), context 31/8 tokens, `<<strong>>` marks at
≥ 30% of the feature max, dense branch at ≥ 10% density (P75 mark floor + 6 contrast
snippets), 20 + 20 detection probes with the probed token in `[[ ]]`, 20 fuzzing pairs with
exactly one mark per probe, seed 0, temperature 0.

## Reading the v8 explanations

- **28617 / 12298** reproduce the human labels ("to" with the leading-space tokenization
  detail; "token after a/an/another", incl. position-0 and spillover behaviour).
  Detection misses are confined to weak spillover firings that the explanations rightly
  exclude (`tpr_by_strength`).
- **25337** is described as "the token following a short consonant-final stem, most
  reliably after cut/pick/click/dump/break/jump" — closer to the corpus ground truth
  (form of the preceding token: -ck/-t/-p stems) than the human label "preceded by a
  verb", but still verb-centred; fuzzing TPR 0.45 measures the remaining gap.
- **14825** = religious register with God/Lord/Allah peaks and a diffuse weak background;
  the weak background is what the detection score cannot capture.
- **24792** (92.6% density) is explained through where it stays silent (sequence-initial
  tokens, mid-sentence articles); single-mark fuzzing on a dense feature is only
  discriminative if the explanation pins down all silent positions (TNR 0.20).

## Reading the v9 explanations (non-reasoning Flash)

Same prompts and probes as v8; only the model differs. 15x faster (33 s), but the
explainer misreads the evidence on two of five features: **12298** becomes "reduplicated
article 'a a'" — the statistics card lists ' a' as the preceding token with a 100%
conditional rate and the model turns that into a fired-on-token claim (detection TPR 0.00;
the same failure appeared in the pre-audit run v2, so it is a stable weakness of the
non-reasoning model, not noise); **25337** stays verb-centred (TPR 0.15). **28617**,
**14825**, **24792** are described correctly and score close to the reasoning run. As an
examiner the non-reasoning model is usable (fuzzing TNR 0.75–1.00), as an explainer it is
not reliable on features whose mechanism sits on the preceding token.
