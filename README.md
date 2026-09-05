# RouterInterp

Goal: understand superposed specialisation in Mixture-of-Experts routing by mapping
interpretable (SAE) features to experts. Step 1 (this directory): an automated
feature-interpretation ("autointerp") pipeline that runs on open-weight LLM APIs or
local vLLM — no OpenAI/Anthropic dependency.

## Autointerp pipeline

Methodology follows Cunningham et al. (ICLR 2024, [F76bwRSLeK](https://openreview.net/forum?id=F76bwRSLeK))
and the cheaper scoring protocols of Paulo et al. 2024 ([2410.13928](https://arxiv.org/abs/2410.13928), delphi),
re-implemented against our own activation-dump format:

- corpus `token_list_*.pkl`: `tokens[batch][prompt][pos] -> str`
- feature `feature{ID}__{label}__*.pt`: `[N,4]` rows `(batch, prompt, pos, activation)` — all firings

Stages per feature (`autointerp_pipeline/`):

1. **sampler** — top-k + activation-quantile-stratified snippets (±32/8 token context,
   firing tokens marked `<<...>>`), held-out probe sets, uniform non-firing negatives.
   Dense features (density ≥ 15%, e.g. feature24792 at 92.7%) get strength-graded
   handling: only firings above the P75 activation are highlighted and the explainer is
   told the density.
2. **explainer** — LLM writes `LABEL` + `EXPLANATION` from the marked snippets.
3. **scorers** —
   - *detection*: does the feature fire on the final token of a left-context snippet?
     20 held-out firing probes (stratified over strength) + 20 negatives, balanced accuracy.
     (Final-token probing instead of delphi's whole-window judgment: stays well-defined
     for dense features and matches per-token routing decisions.)
   - *fuzzing*: are `<<>>` marks placed correctly? 10 correctly vs 10 wrongly marked.
4. outputs: `outputs/<run>/feature{ID}.json`, `report.md`, `calls.jsonl` (full LLM traces),
   `run_config.json`.

## Running

```bash
# credentials: env RIVAULT_API_KEY / RIVAULT_BASE_URL, or ~/.config/opencode/opencode.json
~/miniconda3_x86/envs/vllm/bin/python run_autointerp.py \
    --data-dir ../autointerp --model moonshotai/Kimi-K3 --workers 6
```

Any OpenAI-compatible endpoint works (`--model`, env `RIVAULT_BASE_URL`): RiVault-served
Kimi-K3 / GLM-5.2 / DeepSeek-V4-Flash today, local vLLM later.

Never commit API keys; the client reads them at runtime only.

## Sanity results (5 hand-labeled sample features)

See `outputs/*/report.md`. First smoke run (Kimi-K3): feature28617 (human "to") →
LLM label "'to' token detector", detection 1.00, fuzzing 1.00 balanced accuracy.

## Next: router → expert mapping (planned)

Router logits are linear in the residual stream, so with an SAE decomposition
x ≈ Σ aⱼdⱼ + b the contribution of feature j to expert e's logit is aⱼ·(W_r[e]·dⱼ):
closed-form feature→expert attribution, to be validated with routing-frequency
statistics and feature-ablation causal checks, then aggregated per expert into
labeled "feature portfolios" via this pipeline.
