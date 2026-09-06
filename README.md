# RouterInterp

Goal: understand superposed specialisation in Mixture-of-Experts routing by mapping
interpretable (SAE) features to experts. Stage A (this directory): an automated
feature-interpretation ("autointerp") pipeline that runs on open-weight LLM APIs or
local vLLM — no OpenAI/Anthropic dependency. Stage B (planned): feature → expert
attribution.

## Autointerp pipeline

Methodology follows Cunningham et al. (ICLR 2024, [F76bwRSLeK](https://openreview.net/forum?id=F76bwRSLeK))
and the cheaper scoring protocols of Paulo et al. 2024 ([2410.13928](https://arxiv.org/abs/2410.13928), delphi),
re-implemented against our own activation-dump format:

- corpus `token_list_*.pkl`: `tokens[batch][prompt][pos] -> str`
- feature `feature{ID}__{label}__*.pt`: `[N,4]` rows `(batch, prompt, pos, activation)` — all firings

Per feature (`autointerp_pipeline/`):

1. **statistics card** (`stats.py`, no LLM cost) — density, firing rate by sequence
   position, top fired-on / preceding tokens with their conditional rates
   P(fire | token) against corpus-wide counts. Shares say what is common among firings;
   conditional rates say how reliably a token triggers the feature.
2. **sampler** — 24 snippets for the explainer: a diversity-quota'd top set plus a set
   stratified over activation rank and value bins; every firing marked by class
   (`<<strong>>` ≥ 30% of the feature max, `<weak>` otherwise), `[START]` marker and
   position in headers. Dense features (density ≥ 10%) additionally get non-firing
   contrast snippets. Probe sets are disjoint from the explainer set.
3. **explainer** — one LLM call → `LABEL` + `EXPLANATION`.
4. **detection** — 20 held-out firing probes (stratified over strength) + 20 uniform
   non-firing probes; the probed token is wrapped in `[[ ]]` and named in the header;
   "does the feature fire on the probe token?" Balanced accuracy, chance = 0.5;
   `tpr_by_strength` shows which activation regime the explanation covers.
5. **fuzzing** — 20 strength-matched pairs of strong firings: one snippet with the
   anchor correctly marked `<<...>>`, its twin with a single mark on a zero-activation
   token. Exactly one mark per probe, so every probe is a single judgment (multi-mark
   probes with mixed evidence made reasoning examiners ruminate for 100k tokens).
   "Does the explanation predict that the feature fires on the marked token?"
6. outputs: `outputs/<run>/feature{ID}.json`, `report.md`, `calls.jsonl` (full LLM
   traces incl. reasoning excerpts), `run_config.json`. `compare_runs.py` tabulates
   several runs side by side.

## Running

```bash
pip install -r requirements.txt          # torch + requests, no GPU needed

# credentials for any OpenAI-compatible endpoint (never committed):
export RIVAULT_BASE_URL=https://api.class2.llm.ai.r-ccs.riken.jp/v1/
export RIVAULT_API_KEY=...               # or put these lines in ~/.config/rivault.env

# full run on the five sample features (data files sit next to the script)
python run_autointerp.py --model deepseek-ai/DeepSeek-V4-Flash-reasoning

# cheap smoke test
python run_autointerp.py --features feature28617__to__260904.pt --n-test 5 --n-fuzz 5

# split explainer / examiner models
python run_autointerp.py --model moonshotai/Kimi-K3 --scorer-model deepseek-ai/DeepSeek-V4-Flash

# change a scoring protocol and re-score the SAME explanations
python run_autointerp.py --reuse-explanations results/v8_ds-reasoning --scorers fuzz --out outputs/<new>
python compare_runs.py results/v8_ds-reasoning outputs/<new>
```

Responses are streamed (SSE) because long reasoning generations exceed the gateway's HTTP
timeout otherwise; retries are classed by cause (see `CLAUDE.md`). `CLAUDE.md` is the
full usage and architecture guide.

## Results on the 5 hand-labeled sample features

Explanations from DeepSeek-V4-Flash-reasoning, scored by the same model
(detection / fuzzing balanced accuracy; full three-model records in `results/`):

| feature | human label | density | det | fuzz |
|---|---|---|---|---|
| 28617 | to | 1.9% | 0.80 | 1.00 |
| 12298 | after_a | 3.5% | 0.77 | 0.93 |
| 14825 | a_ubiquitous_feature | 12.3% | 0.72 | 0.60 |
| 24792 | content_word_marker | 92.7% | 0.65 | 0.55 |
| 25337 | preceded_by_a_verb | 4.3% | 0.65 | 0.68 |

Notes: the scores are honest — wrong explanations land at chance level. Corpus
statistics show feature25337 keys on the FORM of the preceding token (short stems ending
in -ck/-t/-p: cut, pick, click → firing 100% of the time), of which the human label
"preceded by a verb" is a subset; the explainer has not yet recovered this, and fuzzing
TPR 0.35 quantifies the gap. Dense features remain hard to fuzz: a single wrong mark on a
93%-dense feature is only detectable if the explanation pins down the 7% of positions
where it does not fire.

## Next: router → expert mapping (planned)

Router logits are linear in the residual stream, so with an SAE decomposition
x ≈ Σ aⱼdⱼ + b the contribution of feature j to expert e's logit is aⱼ·(W_r[e]·dⱼ):
closed-form feature→expert attribution, to be validated with routing-frequency
statistics and feature-ablation causal checks, then aggregated per expert into
labeled "feature portfolios" via this pipeline.
