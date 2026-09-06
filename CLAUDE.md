# CLAUDE.md — agent guide for this repository

This repository holds (1) sample activation data for five hand-labeled SAE features and
(2) the RouterInterp Stage-A **autointerp pipeline** that interprets such features with an
LLM and scores the interpretations. Read this file before running or changing anything.

## Language policy

Everything committed here — code, comments, prompts, docs, commit messages, results — is
written in **English**. Discussion with a user may be in any language; files may not.

## What is in the repository

```
token_list_260904.pkl              corpus: tokens[batch][prompt][pos] -> str  (825 x 16 x 128 = 1.69M tokens)
feature{ID}__{human_label}__260904.pt   [N,4] rows (batch, prompt, pos, activation): every firing of that feature
feature_reading_demo.ipynb         original demo of reading the data
run_autointerp.py                  pipeline entry point (CLI)
compare_runs.py                    side-by-side table of several runs
autointerp_pipeline/               the pipeline package (see Architecture)
results/                           committed full records of runs on the accepted protocol + README index + HTML report
outputs/                           scratch runs (gitignored)
```

The human labels in the filenames are ground truth for calibrating the pipeline. **Never
modify the data files.**

## How to use

### 1. Environment

Any Python ≥ 3.10 with `torch` and `requests` (`pip install -r requirements.txt`). No GPU
is needed: the pipeline only reads activation dumps and calls an LLM API.

### 2. Credentials

The LLM is reached through any OpenAI-compatible `/v1/chat/completions` endpoint.
Resolution order, first hit wins:

1. environment variables `RIVAULT_BASE_URL` and `RIVAULT_API_KEY`;
2. `~/.config/rivault.env` with shell-style `export RIVAULT_API_KEY="..."` lines;
3. `~/.config/opencode/opencode.json` (opencode provider config).

Never commit a key. The client reads it at runtime only.

### 3. Run

```bash
# all five features, the validated default model (~8 min, ~90 calls)
python run_autointerp.py --model deepseek-ai/DeepSeek-V4-Flash-reasoning

# cheap smoke test: one feature, fewer probes
python run_autointerp.py --features feature28617__to__260904.pt --n-test 5 --n-fuzz 5

# strong explainer, cheap examiner
python run_autointerp.py --model moonshotai/Kimi-K3 --scorer-model deepseek-ai/DeepSeek-V4-Flash

# change a scoring protocol and re-score the SAME explanations (reasoning explainers
# are not deterministic, so re-running the explainer would confound the comparison)
python run_autointerp.py --reuse-explanations results/v8_ds-reasoning --scorers fuzz --out outputs/my_variant

# compare runs
python compare_runs.py results/v8_ds-reasoning outputs/my_variant
```

Useful flags: `--out DIR` (default `outputs/<timestamp>_<model>`), `--workers N`
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

Reference results for the five features are in `results/README.md` and the HTML report
`results/autointerp_report.html`.

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
- `llm.py` — streaming (SSE) OpenAI-compatible client with a retry policy by failure
  class: thinking budget exhausted → retry with the budget doubled up to 128k;
  gateway/network failure (5xx/429, timeout, dropped or empty stream) → retry with backoff
  up to 30 times at the same budget; a model that finishes without text → `ModelGaveUp`,
  not retried; 4xx → not retried.

## Models seen so far (RIKEN RiVault gateway)

| model | role | notes |
|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-reasoning` | explainer + examiner, default | ~320 tok/s; 5 features in ~8 min |
| `deepseek-ai/DeepSeek-V4-Flash` | examiner only | no reasoning, ~0.2 s/call; misreads the statistics card as an explainer |
| `moonshotai/Kimi-K3` | explainer for hard features / spot checks | ~30 tok/s, 15–30k thinking tokens per explanation, 2–3 h for 5 features; use `--workers 3` |

## Conventions for changes

- Keep prompts in `prompts.py` only; keep every sampling knob in `SamplerConfig`.
- A scoring-protocol change is evaluated with `--reuse-explanations` against fixed
  explanations, then compared with `compare_runs.py`.
- Runs on the accepted protocol are copied into `results/<run>/` with their logs and
  added to `results/README.md`; everything else stays in `outputs/` (gitignored).
- Main-thread design and review: Fable with max thinking (`.claude/settings.json`);
  substantial implementation work: the `coder` agent (`.claude/agents/coder.md`, Opus).

## Planned Stage B

Router logits are linear in the residual stream, so with an SAE decomposition
x ≈ Σ aⱼdⱼ + b the contribution of feature j to expert e is aⱼ·(W_r[e]·dⱼ): closed-form
feature → expert attribution, to be validated with routing statistics and feature
ablation, then labeled with this pipeline.
