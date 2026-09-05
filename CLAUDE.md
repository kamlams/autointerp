# CLAUDE.md

## Project

RouterInterp: understanding superposed specialisation in Mixture-of-Experts routing by
mapping interpretable (SAE) features to experts.

- **Stage A (this repo, working)**: automated feature-interpretation ("autointerp")
  pipeline over open-weight LLM APIs — no OpenAI/Anthropic dependency.
- **Stage B (planned)**: feature → expert attribution. Router logits are linear in the
  residual stream, so with x ≈ Σ aⱼdⱼ + b the contribution of feature j to expert e is
  aⱼ·(W_r[e]·dⱼ); validate with routing-frequency statistics and ablation, then label the
  involved features with the Stage-A pipeline.

## Language policy

Everything in this repo — code, comments, docstrings, prompts, docs, commit messages,
reports — is written in **English** (the repo will be shared). Conversation with the user
may be in Chinese, but no Chinese in files.

## Data (external, do not modify)

`../autointerp/` (github.com/kamlams/autointerp) holds sample data from a collaborator:

- `token_list_*.pkl` — corpus: `tokens[batch][prompt][pos] -> str`
  (825 × 16 × 128 = 1.69M tokens)
- `feature{ID}__{human_label}__*.pt` — `[N,4]` rows `(batch, prompt, pos, activation)`,
  every position where that feature fired. The human labels in filenames are ground truth
  for calibrating the pipeline.

## Commands

```bash
# conda env with torch/requests (no torch in system python)
PY=~/miniconda3_x86/envs/vllm/bin/python

# full run over all features in the data dir
$PY run_autointerp.py --model moonshotai/Kimi-K3 --workers 12

# fast model, or split explainer/scorer models
$PY run_autointerp.py --model deepseek-ai/DeepSeek-V4-Flash
$PY run_autointerp.py --model moonshotai/Kimi-K3 --scorer-model deepseek-ai/DeepSeek-V4-Flash

# single feature, reduced probes (cheap smoke test)
$PY run_autointerp.py --features ../autointerp/feature28617__to__260904.pt --n-test 10
```

Outputs land in `outputs/<run>/`: `report.md`, `feature{ID}.json`, `calls.jsonl`
(full LLM traces), `run_config.json`.

## Architecture (autointerp_pipeline/)

- `data.py` — corpus + feature loading, firing index for O(1) negative sampling
- `stats.py` — the statistics card shown to the explainer: density, firing rate and mean
  activation by sequence position, top fired-on / preceding tokens among the strongest
  2000 and among all firings, each with its CONDITIONAL RATE P(fire | token) computed
  against corpus-wide token counts (`Corpus.token_counts()`, 0.3 s once). Shares say what
  is common among firings; conditional rates say how reliably a token triggers the
  feature (feature25337: ' cut'/' pick'/' click' → firing 100% of the time, ' up' fires
  on 21% of its occurrences — the mechanism is the preceding token's form). Zero LLM
  cost, 0.2 s per feature
- `sampler.py` — top pool + stratified snippets (half rank bins, half activation-value
  bins so heavy-tailed features get their strong regime probed); diversity quotas
  proportional to pool share (x0.5, floor 2, overflow when exhausted) on the
  (preceding, fired-on) bigram and on each token alone; disjoint explainer / detection /
  fuzzing sets; `[START]` marker and position in headers; every firing is marked by
  class — `<<strong>>` at ≥ 30% of the feature max (`strong_floor`; dense features
  ≥ 10% density also require ≥ P75) or `<weak>` spillover; dense features get non-firing
  contrast snippets; byte-fragment tokens merged. Fuzzing probes are strength-matched
  PAIRS from the same bin (one correct, one wrongly marked), anchors from strong
  firings only, ≤ 5 strong marks per probe, wrong marks on zero-activation tokens
- `prompts.py` — all three prompt templates (explainer, detection, fuzzing)
- `explainer.py` — one LLM call → `LABEL` + `EXPLANATION` (reasoning models spend
  15-30k thinking tokens here; `--explain-max-tokens` defaults to 32k)
- `scorer.py` — detection: the probed token is wrapped in `[[ ]]` with context on both
  sides and named in the header ("does the feature fire on the probe token?"; per-token
  protocol, well-defined for dense features, robust to examiners that would otherwise
  judge the wrong token) and fuzzing (are the `<<>>` marks placed correctly vs
  deliberately-wrong marks?); balanced accuracy, chance = 0.5, parse failures excluded
  from metrics and counted
- `llm.py` — OpenAI-compatible client: credential chain (env → ~/.config/rivault.env →
  ~/.config/opencode/opencode.json), retries with adaptive max_tokens doubling when a
  reasoning model burns the whole budget thinking (finish_reason=length, empty content)
- `compare_runs.py` (repo root) — side-by-side table of several output dirs

## Conventions

- API keys are never committed; the client resolves them at runtime (see llm.py).
- Reproducibility: per-feature RNG `seed*100003 + feature_id`, temperature 0.
- LLM endpoint is RiVault (RIKEN R-CCS gateway); model choice is a config string, local
  vLLM later is the same interface. Known models: moonshotai/Kimi-K3 (reasoning, ~30
  tok/s, strong; thinking cannot be disabled by any API parameter; requests over ~10 min
  get HTTP 504 from the gateway), deepseek-ai/DeepSeek-V4-Flash (no reasoning,
  ~0.2s/call, weak explainer), deepseek-ai/DeepSeek-V4-Flash-reasoning (reasoning at
  ~320 tok/s — the default for validation runs; its usage reports text_tokens=0 even
  when content is present), zai-org/GLM-5.2, Qwen/Qwen3.8-27B, and others listed by
  GET /v1/models.
- Scoring reads: balanced accuracy, chance = 0.5; a good explanation needs BOTH detection
  and fuzzing high. `tpr_by_strength` in detection results diagnoses which activation
  regime an explanation fails on.

## Claude Code model policy

- Main thread (design, review, discussion): Fable with max thinking —
  configured in `.claude/settings.json`.
- Complex implementation work: delegate to the `coder` agent (Opus,
  `.claude/agents/coder.md`).
- Background jobs edit this repo directly (worktree isolation disabled in settings).
