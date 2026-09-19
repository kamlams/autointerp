# autointerp

Automated interpretation of sparse-autoencoder (SAE) features: an LLM reads where a feature
fires and writes an explanation, and the explanation is then scored on held-out probes. It runs on
any OpenAI-compatible endpoint serving open-weight models — no OpenAI/Anthropic dependency.

The features studied here come from TopK SAEs (k = 32, 32768 latents) trained on
OLMoE-1B-7B-0924, at the output of `layers.L.post_attention_layernorm`.

## How a feature is interpreted

Methodology follows Cunningham et al. (ICLR 2024, [F76bwRSLeK](https://openreview.net/forum?id=F76bwRSLeK))
and the cheaper scoring protocols of Paulo et al. 2024 ([2410.13928](https://arxiv.org/abs/2410.13928), delphi),
re-implemented against a plain activation-dump format:

- corpus `token_list_*.pkl`: `tokens[batch][prompt][pos] -> str`
- feature `feature{ID}__*.pt`: `[N,4]` rows `(batch, prompt, pos, activation)` — every firing

Per feature (`autointerp_pipeline/`, driven by `process_feature` in `run_autointerp.py`):

1. **statistics card** (`stats.py`, no LLM cost) — density, firing rate by sequence position,
   top fired-on / preceding tokens with their conditional rates P(fire | token). Shares say what
   is common among firings; conditional rates say how reliably a token triggers the feature.
2. **sampler** (`sampler.py`) — 24 snippets for the explainer: a diversity-quota'd top set plus a
   set stratified over activation rank and value bins; every firing marked by class
   (`<<strong>>` >= 30% of the feature max, `<weak>` otherwise). Dense features (density >= 10%)
   also get non-firing contrast snippets. Probe sets are disjoint from the explainer set.
3. **explainer** — one LLM call -> `LABEL` + `EXPLANATION`.
4. **detection** — 20 held-out firing probes (stratified over strength) + 20 non-firing probes;
   the probed token is wrapped in `[[ ]]`: "does the feature fire on the probe token?"
5. **fuzzing** — 20 strength-matched pairs of strong firings: one snippet with the anchor
   correctly marked `<<...>>`, its twin with a single mark on a zero-activation token; exactly one
   mark per probe, so every probe is a single judgment.

Both scores are balanced accuracy, chance = 0.5; an explanation is good only if **both** are high.

## Running

```bash
pip install -r requirements.txt     # torch + requests; no GPU needed for run_autointerp.py

# credentials of an OpenAI-compatible endpoint (never committed):
export RIVAULT_BASE_URL=https://<your-gateway-host>/v1/
export RIVAULT_API_KEY=...          # or put both lines in ~/.config/rivault.env

# the example data in feature-data/: three latents of the latest result set (a few minutes)
python run_autointerp.py --model deepseek-ai/DeepSeek-V4-Flash-reasoning \
    --scorer-model deepseek-ai/DeepSeek-V4-Flash

# re-score the SAME explanations with another examiner or protocol (explainers are not
# deterministic, so re-running them would confound the comparison), then compare
python run_autointerp.py --reuse-explanations outputs/<run> --scorer-model <model> --out outputs/<new>
python compare_runs.py outputs/<run> outputs/<new> --labels
```

## From an SAE checkpoint to explanations

`extract_features.py` turns any TopK SAE on OLMoE-1B-7B-0924 into activation dumps (GPU; one
truncated model forward over the corpus, a few minutes per layer whatever the number of latents):

```bash
# named latents of a layer-7 SAE trained on the norm output (--input h;
# an SAE trained on the pre-norm residual takes --input m)
python extract_features.py --sae <run>/layers.7.post_attention_layernorm/sae.safetensors \
    --layer 7 --input h --name moein_L7 --latents 971,15329,25609
# or a density-stratified random sample
python extract_features.py --sae <...>/sae.safetensors --layer 7 --input h --name moein_L7 --n-features 100

python run_autointerp.py --data-dir outputs/feature-data/moein_L7 --model <explainer> --scorer-model <examiner>
python summarize_features.py outputs/feature-data/moein_L7 outputs/<run> report.md
```

The corpus is a held-out SmolLM2-135M-10B shard cut into 825 x 16 x 128 tokens and cached, so
every SAE is read on the same tokens; `sampling.json` records the SAE, layer, input side and the
latents taken. Activation dumps are reproducible in minutes and large, so they live in the
gitignored `outputs/`; explanations cost hours of LLM time and are not reproducible, so they are
committed under `results/`.

## Results

`results/feature_test32b_260918/` — 96 latents chosen by the SAE's author (32 each at layers 0, 7
and 15), each explained by three models and scored by two examiners. 96 of 96 explained.

- Layer 0 latents are token-identity and next-token-expectation features, layer 7 lexical
  families and syntactic roles, layer 15 constructions and discourse roles.
- DeepSeek-V4-Flash-reasoning, Kimi-K3 and Qwen3.8-27B are indistinguishable as explainers
  (paired differences +0.003 to +0.016, every 95% CI crossing zero); the first is 1.5-1.6x faster.
- The examiner moves scores four times more than the explainer: always state it next to a score.
- Latents firing on more than 5% of tokens (17 of the 96, one at 97.8%) have no describable
  meaning and score near chance on fuzzing.

`results/README.md` has the details; `results/feature_test32b_260918/features_L{0,7,15}.md` are the
readable per-latent tables.
