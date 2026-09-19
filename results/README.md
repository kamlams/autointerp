# Results

Only the latest result set is tracked.

## `feature_test32b_260918/` — 96 latents, three explainers, two examiners

32 latents each at layers 0, 7 and 15 of a TopK SAE (k = 32, 32768 latents, trained for 200M
tokens on the output of `post_attention_layernorm` of OLMoE-1B-7B-0924), selected by the SAE's
author. Activations were extracted with `extract_features.py --latents` on a held-out
1.69M-token corpus, then explained and scored with `run_autointerp.py` (24 explainer snippets,
20 + 20 detection probes, 20 fuzzing pairs, seed 0). 96 of 96 latents were explained.

| path | content |
|---|---|
| `features_L0.md`, `features_L7.md`, `features_L15.md` | one row per latent: density, firings, top fired-on tokens with P(fire \| token), label, detection, fuzzing; full explanations below each table |
| `moein_L{0,7,15}_test32b/` | per-latent JSON (explanation, scores, per-probe predictions, statistics card, explainer prompt), `sampling.json`, `run.log` — explainer and examiner `DeepSeek-V4-Flash-reasoning` |
| `by_explainer/{kimi,qwen}/L*/` | the same latents explained by `Kimi-K3` and `Qwen3.8-27B`, same examiner; `by_explainer/deepseek/` links to the run above |
| `by_explainer/*/check/L*/` | all three explanation sets re-scored by `DeepSeek-V4-Flash`, which wrote none of them |
| `by_explainer/compare_explainers.py` | reproduces every number of `by_explainer/README.md` from these files |
| `run_scoring.sh`, `run_explainers.sh`, `resume_explainers.sh` | how the runs were launched (four to six parallel processes, resumable) |

What the latents are (`README.md` in the directory has more): layer 0 — token identity and
next-token expectation, sparse ones firing on a single word with P(fire | token) = 100%; layer 7 —
lexical families (`give`, `drink`, `kill`) and syntactic roles, plus topic/register contexts that
detection recognises but fuzzing cannot place on a token; layer 15 — constructions and discourse
roles (prepositional-phrase onsets, appositives, reported speech, obituary lists). 17 of the 96
fire on more than 5% of tokens (one on 97.8%) and score near chance on fuzzing. Layer 0 holds
three copies of the same `" to"` detector (latents 36, 6584, 6758; firing-position Jaccard
0.95-0.99).

Which explainer (`by_explainer/README.md`): the three are indistinguishable — min(detection,
fuzzing) 0.582 / 0.569 / 0.566 under the external examiner, paired differences +0.003 to +0.016
with every 95% CI crossing zero — while explanation time is 159 s / 258 s / 236 s per latent.
Changing the examiner costs every arm 0.06-0.07, four times the largest gap between explainers.
