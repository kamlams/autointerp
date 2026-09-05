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
- `sampler.py` — top-k + activation-quantile-stratified snippets; disjoint
  explainer/detection/fuzzing sets; dense-feature branch (density ≥ 10%: highlight only
  > P75 activations and tell the explainer the density)
- `prompts.py` — all three prompt templates (explainer, detection, fuzzing)
- `explainer.py` — one LLM call → `LABEL` + `EXPLANATION`
- `scorer.py` — detection (does the feature fire on the FINAL token of the snippet? —
  final-token protocol, deviates from delphi's whole-window judgment on purpose) and
  fuzzing (are the `<<>>` marks placed correctly vs deliberately-wrong marks?);
  balanced accuracy, parse failures excluded from metrics and counted
- `llm.py` — OpenAI-compatible client: credential chain (env → ~/.config/rivault.env →
  ~/.config/opencode/opencode.json), retries with adaptive max_tokens doubling when a
  reasoning model burns the whole budget thinking (finish_reason=length, empty content)

## Conventions

- API keys are never committed; the client resolves them at runtime (see llm.py).
- Reproducibility: per-feature RNG `seed*100003 + feature_id`, temperature 0.
- LLM endpoint is RiVault (RIKEN R-CCS gateway); model choice is a config string, local
  vLLM later is the same interface. Known models: moonshotai/Kimi-K3 (reasoning, slow,
  strong), deepseek-ai/DeepSeek-V4-Flash (no reasoning, ~0.2s/call, weak explainer but
  adequate scorer), zai-org/GLM-5.2, qwen3-coder:30b.
- Scoring reads: balanced accuracy, chance = 0.5; a good explanation needs BOTH detection
  and fuzzing high. `tpr_by_strength` in detection results diagnoses which activation
  regime an explanation fails on.

## Claude Code model policy

- Main thread (design, review, discussion): Fable with max thinking —
  configured in `.claude/settings.json`.
- Complex implementation work: delegate to the `coder` agent (Opus,
  `.claude/agents/coder.md`).
- Background jobs edit this repo directly (worktree isolation disabled in settings).
