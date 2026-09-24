# CLAUDE.md — agent guide for this repository

This repository holds the **autointerp pipeline** — an LLM explains where an SAE feature fires
and the explanation is scored on held-out probes — together with the tool that extracts feature
activations from a TopK SAE checkpoint, a small example data set, and the latest result set.
Read this file before running or changing anything.

## Language policy

Everything committed here — code, comments, prompts, docs, commit messages, results — is
written in **English**. Discussion with a user may be in any language; files may not.

## What is in the repository

```
run_autointerp.py        pipeline entry point (CLI): activation dumps -> explanations + scores
run_paper_protocol.py    the published protocol (Paulo et al. 2025) and its single-mark exam; see results/paper_protocol_260920/
extract_features.py      TopK SAE checkpoint -> activation dumps in the feature-data layout (GPU)
summarize_features.py    one readable markdown table per run (density, top tokens, label, scores)
compare_runs.py          side-by-side detection/fuzzing table of several run directories
autointerp_pipeline/     the pipeline package (see Architecture)
feature-data/            example data: corpus + three latents of the latest result set
results/                 result sets (the paper's protocol on two latent sets, the own pipeline) + README index
outputs/                 scratch: generated feature-data/ and raw run directories (gitignored)
```

Data flow: `SAE checkpoint + corpus --extract_features.py--> outputs/feature-data/<name>/
--run_autointerp.py--> outputs/<run>/ --(copy the feature JSONs)--> results/<run>/`. Activation
dumps are reproducible in minutes and large, so they stay in `outputs/`; explanations cost hours
of LLM time and are not reproducible, so they are committed under `results/`.

`run_autointerp.py` reads `feature-data/` by default; point `--data-dir` at any directory with
the same layout (`token_list_*.pkl` + `feature{ID}__*.pt`), typically one written by
`extract_features.py` under `outputs/feature-data/`. A file name may carry a human label
(`feature{ID}__{label}__{tag}.pt`); it is then reported next to the LLM label.

## How to use

### 1. Environment

Any Python ≥ 3.10 with `torch` and `requests` (`pip install -r requirements.txt`). No GPU
is needed for `run_autointerp.py`: it only reads activation dumps and calls an LLM API.
`extract_features.py` needs a GPU plus `transformers`, `datasets`, `safetensors`, `numpy`, and
`eai-sparsify` once, to build the cached corpus.

### 2. Credentials

The LLM is reached through any OpenAI-compatible `/v1/chat/completions` endpoint.
Resolution order, first hit wins:

1. environment variables `RIVAULT_BASE_URL` and `RIVAULT_API_KEY`;
2. `~/.config/rivault.env` with shell-style `export RIVAULT_API_KEY="..."` lines;
3. `~/.config/opencode/opencode.json` (opencode provider config).

Never commit a key. The client reads it at runtime only.

### 3. Run

```bash
# the three example latents, recommended models (a few minutes)
python run_autointerp.py --model deepseek-ai/DeepSeek-V4-Flash-reasoning --scorer-model deepseek-ai/DeepSeek-V4-Flash

# cheap smoke test: one feature, fewer probes
python run_autointerp.py --features feature-data/feature25609__260918.pt --n-test 5 --n-fuzz 5

# change the examiner or a scoring protocol and re-score the SAME explanations (reasoning
# explainers are not deterministic, so re-running the explainer would confound the comparison)
python run_autointerp.py --reuse-explanations outputs/my_run --scorer-model <model> --out outputs/my_variant

# compare runs
python compare_runs.py outputs/my_run outputs/my_variant --labels

# a new SAE: extract, explain, report
python extract_features.py --sae <run>/layers.7.post_attention_layernorm/sae.safetensors --layer 7 --input h --name my_sae_L7 --n-features 100
# an SAE on another model (same layer layout): its own tokenizer gives its own cached corpus;
# the printed FVU (0.02-0.5 when model, layer and side match) guards against a mismatched SAE
python extract_features.py --model Qwen/Qwen3-30B-A3B --sae <run>/layers.24.post_attention_layernorm/sae.safetensors --layer 24 --input h --name qwen3_moein_L24 --latents "754 1284"
python run_autointerp.py --data-dir outputs/feature-data/my_sae_L7 --model <explainer> --scorer-model <examiner> --out outputs/my_sae_L7_run
python summarize_features.py outputs/feature-data/my_sae_L7 outputs/my_sae_L7_run report.md
```

Useful flags: `--data-dir DIR` (default `feature-data/`), `--out DIR` (default
`outputs/<timestamp>_<model>`), `--workers N`
(concurrent scorer calls; default 12, or 3 when Kimi-K3 is the examiner), `--seed`,
`--n-explainer 24`, `--n-test 20`, `--n-fuzz 20`, `--explain-max-tokens 32768`.

### 4. Read the output

`outputs/<run>/` (or `results/<run>/`) contains `report.md` (table + per-feature detail),
`feature{ID}.json` (label, explanation, scores, per-probe predictions, diagnostics,
statistics card, the explainer prompt), `calls.jsonl` (every LLM call: prompt, reply,
reasoning excerpt, token usage), `run_config.json`.

Scores are **balanced accuracy, chance = 0.50**, over held-out probes:

- **detection** — 20 firing probes stratified over activation strength + 20 random
  non-firing probes; the probed token is wrapped in `[[ ]]`; "does the feature fire on the
  probe token?" Rewards coverage of the whole activation range. `tpr_by_strength` shows
  which activation regime an explanation misses.
- **fuzzing** — 20 pairs of strong firings (≥ 30% of the feature max): one snippet with
  the anchor correctly marked `<< >>`, one twin with a single mark on a zero-activation
  token; "does the explanation predict a firing on the marked token?" Rewards token-level
  precision. Exactly one mark per probe.

A trustworthy explanation needs both high. Wrong explanations land at ~0.5 by
construction; that is the pipeline working. Dense features (≥ 10% density) are capped
low on fuzzing unless the explanation pins down where they stay silent.

Reference results: `results/README.md`, and the per-latent tables
`results/feature_test32b_260918/features_L{0,7,15}.md`.

## Architecture (autointerp_pipeline/)

- `data.py` — corpus and feature loading; firing index for O(1) negative sampling;
  corpus token counts for conditional rates.
- `stats.py` — the statistics card shown to the explainer: density, firing rate and mean
  activation by sequence position, top fired-on / preceding tokens among the strongest
  2000 and among all firings, each with its conditional rate P(fire | token). Zero LLM cost.
- `sampler.py` — 24 explainer snippets (8 from the top pool under proportional
  diversity quotas on the (preceding, fired-on) bigram and each token alone; 16 stratified
  half by rank bins, half by activation-value bins); context 31 tokens left / 8 right;
  `[START]` marker and position in headers; every firing marked by class (`<<strong>>`
  ≥ 30% of max, `<weak>` otherwise); dense features additionally use a P75 floor and get
  6 non-firing contrast snippets; disjoint explainer / detection / fuzzing sets; per-feature
  seed `seed*100003 + feature_id`.
- `prompts.py` — the three prompt templates (explainer, detection, fuzzing), verbatim.
- `explainer.py` — one call → `LABEL` + `EXPLANATION`.
- `scorer.py` — detection and fuzzing as above; batches of 5 probes per call; parse
  failures and failed batches are excluded from the metrics and counted.
- `sae_tools.py` — everything `extract_features.py` needs and nothing else: `TopKSAE.load`
  (sparsify's native `encoder.weight`/`encoder.bias` layout and the early `W_enc` export),
  `load_truncated_model` (OLMoE by default, or any model with `model.model.layers`, e.g.
  Qwen3-MoE, cut above the SAE's layer), `ActivationTap` on
  `layers.L.post_attention_layernorm` — side `h` = the module's output ("moein" SAEs), side
  `m` = its input, the pre-norm residual ("residmid" SAEs). The hookpoint name does not tell
  the two apart; feeding an SAE the wrong side gives FVU 10-170 instead of 0.02-0.5.
- `llm.py` — streaming (SSE) OpenAI-compatible client with a retry policy by failure
  class: thinking budget exhausted → retry with the budget doubled up to 128k;
  gateway/network failure (5xx/429, timeout, dropped or empty stream) → retry with backoff
  up to 30 times at the same budget; a model that finishes without text → `ModelGaveUp`,
  not retried; 4xx → not retried.

## Models seen so far

| model | role | notes |
|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-reasoning` | explainer, default | ~320 tok/s; ~160 s per feature including scoring |
| `deepseek-ai/DeepSeek-V4-Flash` | examiner, default | no reasoning, ~0.2 s/call, 24 features scored in 90 s; misreads the statistics card as an explainer |
| `moonshotai/Kimi-K3` | alternative explainer | same quality as the default on 96 features, 1.5x slower; use `--workers 3` when it is the examiner |
| `Qwen/Qwen3.8-27B` | alternative explainer | same quality as the default on 96 features, 1.6x slower |

Measured on 96 latents (`results/feature_test32b_260918/by_explainer/`): DeepSeek-reasoning,
Kimi-K3 and Qwen3.8-27B are indistinguishable as explainers, and the examiner moves scores four
times more than the explainer does. Recommended setup: explain with
`DeepSeek-V4-Flash-reasoning`, score with `DeepSeek-V4-Flash` via `--scorer-model` (a different
model from the explainer, and 24 features score in 90 s). Do not use
`ByteDance-Seed/Seed-OSS-36B-Instruct` as an examiner: it buries its verdict in a `<seed:think>`
block and every probe fails to parse. Before trusting a new model as an examiner, ask it the
same prompt twice at temperature 0: some served models are not reproducible.

## Conventions for changes

- Keep prompts in `prompts.py` only; keep every sampling knob in `SamplerConfig`.
- A scoring-protocol change is evaluated with `--reuse-explanations` against fixed
  explanations, then compared with `compare_runs.py`.
- `results/` holds the committed result sets (`results/README.md` says which protocol made each
  and which one the paper uses); raw run directories and `calls.jsonl` stay in `outputs/`
  (gitignored). Numbers quoted in a results README must be reproducible by a committed script
  (see `results/feature_test32b_260918/by_explainer/compare_explainers.py`).
- Nothing private is committed: no API key, no gateway address (it is read from
  `RIVAULT_BASE_URL`, and `llm.py` redacts the host from logs and errors), no cluster paths or
  user names. Before publishing, scan the added lines of every outgoing commit, not only the
  files at HEAD.
