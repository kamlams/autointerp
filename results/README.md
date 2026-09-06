# Results — full pipeline runs on the accepted protocol

Each directory is a complete record of one end-to-end run (fresh explanations +
detection + fuzzing) of the pipeline at the stated commit: `report.md`, one
`feature{ID}.json` per feature (explanation, scores, per-probe predictions, diagnostics),
`calls.jsonl` (every LLM call: prompt, reply, reasoning excerpt, token usage),
`run_config.json`, `run.log`. Scratch experiments live in the gitignored `outputs/`.

Data: the 5 hand-labeled sample features in `feature-data/` (1.69M-token corpus).
Scores are balanced accuracy (chance = 0.5): detection / fuzzing.

| run | date | pipeline commit | explainer = examiner | 28617 to | 12298 after_a | 25337 preceded_by_a_verb | 14825 ubiquitous | 24792 content (93%) | wall | calls | reasoning tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v8_ds-reasoning` | 2026-09-05 | 95621ad | deepseek-ai/DeepSeek-V4-Flash-reasoning | 0.80 / 1.00 | 0.75 / 0.90 | 0.62 / 0.72 | 0.62 / 0.68 | 0.65 / 0.55 | 505 s | 88 | 280k |
| `v9_ds-flash` | 2026-09-06 | bffb5ce | deepseek-ai/DeepSeek-V4-Flash (no reasoning) | 0.68 / 0.88 | 0.50 / 0.55 | 0.57 / 0.57 | 0.65 / 0.68 | 0.55 / 0.57 | 33 s | 87 | 0 |
| `v10_kimi-k3` | 2026-09-06 | bffb5ce (+af0e66d for 14825) | moonshotai/Kimi-K3 | 0.80 / 1.00 | 0.90 / 1.00 | 0.62 / 0.68 | 0.60 / 0.53 | 0.75 / 0.57 | 10280 s | 118 | 714k |

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

## Reading the v10 explanations (Kimi-K3)

Best explanations of the three models, at ~20x the wall time of the reasoning DeepSeek run
(2.9 h, 714k reasoning tokens, 3 workers). **12298** is fully characterised (a/an/another
and word-initial subword "a" → next token; "the" never triggers; sequence-initial "a" fires
on itself) and scores 0.90 / 1.00. **25337** is the first explanation to state the true
mechanism — "the token following a short stem ending in a stop consonant (-t, -ck, -k,
-p, -mp, and voiced -d/-g)" — yet scores no higher than the verb-centred v8 version
(fuzz TPR 0.40): the examiner rejects correct probes such as compound nouns ending in
-cks, so the remaining gap is in how strictly the explanation is applied, not in the
mechanism. **24792** is read as a context-gated surprisal signal (silent at position 0 and
on highly predictable tokens such as " the": 61% vs 93% baseline) and gets the best
detection score on the dense feature (0.75). **14825** = sequence openings + religious
vocabulary, matching the corpus statistics; its low scores come from the diffuse weak
background that no explanation can predict (detection TPR is 1.0 at strengths ≥ 4 and ~0
below). The first attempt lost 14825 when the gateway dropped 7 of 8 parallel Kimi
streams; the rerun under the class-based retry policy at 3 workers hit only
budget-doubling retries (3, all recovered) and no drops.

## Three-model summary

| | DeepSeek-V4-Flash-reasoning (v8) | DeepSeek-V4-Flash (v9) | Kimi-K3 (v10) |
|---|---|---|---|
| mean detection / fuzzing | 0.69 / 0.77 | 0.59 / 0.65 | 0.73 / 0.76 |
| wall time, 5 features | 505 s | 33 s | 10280 s |
| reasoning tokens | 280k | 0 | 714k |
| explanation failures | none | 12298 misread (stats-card columns confused), 25337 verb-only | none |

Non-reasoning Flash is a usable examiner but not a reliable explainer; reasoning Flash is
the cost-effective default; Kimi-K3 gives the most complete explanations at a cost that
only makes sense for hard features or spot checks.
