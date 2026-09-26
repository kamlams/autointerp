# The single-mark exam on a held-out shard: explanation evidence and exam windows disjoint (2026-09-26)

**What this is.** The 30-word explanations of `../own_pipeline_260926/` ("ours") and the
published-protocol explanations of `../paper_protocol_260920/` and `../paper_protocol_260924/`
("published"), all scored again on the single-mark fuzzing exam of `../paper_protocol_260920/`,
with the exam rebuilt from a **different corpus shard** than the one the explanations were written
from.

Reproduce every number below: `python results/exam_shard83_260926/analyze.py` (add `--markdown`
for the tables).

## Why

Both kinds of explanation were written from SmolLM2-135M-10B shard 84. On shard 84, the exam's 100
positive windows per latent are disjoint from the published protocol's 40 explanation windows, but
not from our explanations' evidence: the statistics card counts every firing of the corpus,
including those inside the exam windows, and the 24 snippets are not kept out of them. Excluding
the exam windows from the card is not possible without rewriting the explanations, since any
activating window is part of the card.

Here the exam is built from **shard 83** instead: explanation evidence (shard 84) and exam windows
(shard 83) come from different documents by construction, for both kinds of explanation. Neither
shard was used to train any SAE here (the OLMoE SAEs were trained on shards 0-8; the Qwen3 SAE's
training data is not recorded in its checkpoint, which applies equally to shard 84).

## Setting

- Corpus: SmolLM2-135M-10B `train-00083-of-00085.parquet`, cut exactly like shard 84 (825 x 16 x
  128 = 1,689,600 tokens, the collaborator's `chunk_and_tokenize`), once with the OLMoE tokenizer
  and once with the Qwen3 tokenizer (`outputs/feature-data/corpus_shard83_825x16x128[_Qwen3-30B-A3B]`).
- Activations: `extract_features.py --shard 83 --latents ...` with the same SAE, layer and input side
  as each shard-84 data directory (read from its `sampling.json`), one GPU. Every latent fires on
  shard 83. SAE FVU on the first 8 batches (0.02-0.5 = SAE matches model, layer and side):

  | data | FVU on shard 83 |
  |---|---|
  | OLMoE L0 | 0.079 |
  | OLMoE L7 | 0.454 |
  | OLMoE L15 (both latent sets, one SAE) | 0.422 |
  | OLMoE L1 | 0.240 |
  | Qwen3-30B-A3B L24 | 0.473 (0.498 on shard 84) |

- Exam: `run_paper_protocol.py --arm S1 --scorers fuzz`, examiner DeepSeek-V4-Flash-reasoning, 100
  positives (ten per decile of the window peak) + 50 type-A + 50 type-B negatives per latent, one
  marked token per probe, seed 22 — the exam of `../paper_protocol_260920/`, on shard-83 windows.
  Latents with fewer than 140 activating windows get a smaller exam by the protocol's own rule
  (min(40, a third) of the windows reserved, the rest, at most 100, as positives; e.g. 57 windows
  give 38 positives), identically for both sides: OLMoE L1/85 (138 windows),
  L15/5165 (113), L15/28409 (137); Qwen3 L24/754 (84), 5492 (88), 8066 (125), 20369 (136), 23481
  (57), 8006 (49). L0/2565 fires in every window (52,800 of 52,800) and is not scoreable, on either
  shard.
- Explainers: Kimi-K3 (scored first), Qwen3.8-27B, DeepSeek-V4-Flash-reasoning; DeepSeek-V4-Flash
  has no explanation of ours and is left out.
- Pairing: a latent counts when both sides scored it. Qwen3 L24: ours explain 8066 (the
  collaborator's correction), the published run 8006; both are scored, neither is paired.
- Run: `run.sh` (extraction, then `run_exam.py`: one process per side x explainer x set x layer,
  Kimi-K3 first, at most 60 examiner calls in flight — about 150 in flight overloaded the shared
  gateway earlier the same day), `collect.sh` copies the per-latent records here. Raw calls stay in
  `outputs/exam_shard83/` (gitignored).

## Layout

`<side>/<model>/<set>/<explainer>/<layer>/feature<id>.json` + `run_config.json`, side = `own` |
`delphi`; the explanation text is in each record (`explanation`), its source in `run_config.json`.

## Results

Balanced accuracy (chance 0.5) of the single-mark exam and its parts (TPR on the firing tokens,
TNR on the type-A and type-B wrong marks), ours vs published, paired over the latents both sides
scored; difference with a bootstrap 95% interval, `*` = the interval excludes zero. 1,191 latent
records were scored (598 ours, 593 published: the published sets miss seven explanations); L0/2565 is
not scoreable on either side; 160 of the 233,536 probes were excluded as unparseable examiner replies,
as the protocol does.

| explainer | exam | n | BA ours vs published | TPR | TNR A | TNR B | BA ours - published [95% CI] |
|---|---|---|---|---|---|---|---|
| Kimi-K3 | shard 83 | 198 | 0.690 vs 0.707 | 0.439 vs 0.562 | 0.961 vs 0.873 | 0.922 vs 0.832 | -0.017 [-0.027, -0.007] * |
| Kimi-K3 | shard 84 | 198 | 0.692 vs 0.709 | 0.448 vs 0.570 | 0.962 vs 0.879 | 0.912 vs 0.818 | -0.017 [-0.027, -0.007] * |
| Qwen3.8-27B | shard 83 | 190 | 0.704 vs 0.719 | 0.501 vs 0.633 | 0.932 vs 0.831 | 0.884 vs 0.776 | -0.014 [-0.025, -0.004] * |
| DeepSeek-V4-Flash-reasoning | shard 83 | 197 | 0.693 vs 0.684 | 0.453 vs 0.701 | 0.956 vs 0.681 | 0.909 vs 0.653 | +0.009 [-0.005, +0.022] |

Shard-83 exam per latent set and layer, BA ours vs published (paired n) and the difference:

| set | Kimi-K3 | Qwen3.8-27B | DeepSeek-V4-Flash-reasoning |
|---|---|---|---|
| olmoe-1b-7b-0924 active-on-to L0 | 0.697 vs 0.685 (31), +0.012 [-0.010, +0.036] | 0.704 vs 0.665 (29), +0.039 [+0.012, +0.069] * | 0.694 vs 0.626 (31), +0.068 [+0.031, +0.108] * |
| olmoe-1b-7b-0924 active-on-to L7 | 0.656 vs 0.693 (32), -0.037 [-0.064, -0.011] * | 0.686 vs 0.708 (30), -0.023 [-0.058, +0.012] | 0.650 vs 0.675 (31), -0.025 [-0.067, +0.016] |
| olmoe-1b-7b-0924 active-on-to L15 | 0.610 vs 0.656 (32), -0.046 [-0.069, -0.021] * | 0.643 vs 0.686 (30), -0.042 [-0.068, -0.016] * | 0.636 vs 0.656 (32), -0.020 [-0.047, +0.011] |
| olmoe-1b-7b-0924 expert-linked L1 | 0.830 vs 0.842 (29), -0.013 [-0.040, +0.013] | 0.827 vs 0.835 (29), -0.007 [-0.034, +0.018] | 0.827 vs 0.810 (29), +0.017 [-0.010, +0.044] |
| olmoe-1b-7b-0924 expert-linked L15 | 0.664 vs 0.672 (55), -0.007 [-0.025, +0.012] | 0.668 vs 0.690 (54), -0.021 [-0.035, -0.007] * | 0.667 vs 0.653 (55), +0.014 [-0.006, +0.033] |
| qwen3-30b-a3b expert-linked L24 | 0.733 vs 0.748 (19), -0.016 [-0.054, +0.022] | 0.747 vs 0.777 (18), -0.029 [-0.059, -0.002] * | 0.727 vs 0.738 (19), -0.010 [-0.047, +0.027] |

The shard-84 row repeats the comparison of `../own_pipeline_260926/` on the exam of
`../paper_protocol_260920/`; `analyze.py` prints it when those records are present
(`../own_pipeline_260926/<model>/<set>/paper-exam-single-mark/`, Kimi-K3 at the time of writing).

**Reading.** Moving the exam to a shard the explanations never saw leaves the comparison where it
was: Kimi-K3's difference is the same to three decimals (-0.017 [-0.027, -0.007] on both shards).
(The shard-84 records of the other two explainers are not in this repository yet.) Our
explanations are narrower: they accept fewer firing tokens (TPR lower by 0.12-0.25) and reject far
more wrong marks, of both kinds (TNR higher by 0.09-0.28). The balance is slightly against ours for Kimi-K3 and Qwen3.8-27B and even for
DeepSeek-V4-Flash-reasoning, whose published-protocol explanations are the broadest (TPR 0.701, TNR
A 0.681). By layer, ours do better at OLMoE L0 and worse at L7 and L15 of the latents active on " to";
the expert-linked sets are within about 0.03.
