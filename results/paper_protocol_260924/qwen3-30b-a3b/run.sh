#!/bin/bash
# The protocol of ../../paper_protocol_260920/ (published explainer, single-mark fuzzing exam) on the
# 20 latents of the collaborator's Qwen3_30B_A3B_features.txt of 2026-09-24: Qwen3-30B-A3B, TopK SAE
# (k = 32, 32768 latents) on the output of layers.24.post_attention_layernorm, expert 9: 10 promotive
# + 10 suppressive latents.
#
#   extraction: extract_features.py --model Qwen/Qwen3-30B-A3B, the same held-out SmolLM2 shard 84
#               tokenized with the Qwen3 tokenizer (825 x 16 x 128 tokens)
#   explainers: Kimi-K3 (first), Qwen3.8-27B, DeepSeek-V4-Flash-reasoning; 32768-token budget for all
#   examiner:   DeepSeek-V4-Flash-reasoning
#
# Paths are relative to the repository root. Resumable: a latent whose JSON exists is skipped, so a
# second pass retries the explanations that came back empty.
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
OUT=outputs/paper260924_qwen3-30b-a3b
DATA=outputs/feature-data/qwen3_moein_L24_260924
SAE=${SAE:?set SAE to <collaborator run>/layers.24.post_attention_layernorm/sae.safetensors}
REASON=deepseek-ai/DeepSeek-V4-Flash-reasoning
declare -A MODEL=( [kimi]=moonshotai/Kimi-K3 [qwen]=Qwen/Qwen3.8-27B [reasoning]=$REASON )
L=24
mkdir -p $OUT/logs

echo "$(date '+%F %T') 0. activations (GPU)"
if [ ! -f $DATA/sampling.json ]; then
  $PY extract_features.py --model Qwen/Qwen3-30B-A3B --sae $SAE --layer $L --input h \
      --name $(basename $DATA) --tag 260924 --device ${DEVICE:-cuda:0} \
      --latents "10118 15048 6905 24647 18569 11567 8006 7119 3796 16990 29915 15816 20369 754 9624 1284 7801 23481 11934 5492" \
      > $OUT/logs/extract_L$L.txt 2>&1 || { echo "extraction failed, see $OUT/logs/extract_L$L.txt"; exit 1; }
fi
# stop before any LLM call if the SAE does not reconstruct this model's activations
$PY -c "import json,sys; f=json.load(open('$DATA/sampling.json'))['fvu_first_batches']; print('FVU', f); sys.exit(f > 1)" \
    || { echo "FVU above 1: wrong model, layer or side; stopping"; exit 1; }

explain() {  # explain <key>: two passes, so an empty reply is retried once
  for pass in 1 2; do
    $PY run_paper_protocol.py --data-dir $DATA --arm P0 --explainer-model ${MODEL[$1]} \
        --explain-max-tokens 32768 --scorers none --feature-workers 3 \
        --out $OUT/explain_$1/L$L >> $OUT/logs/explain_$1_L$L.txt 2>&1
  done
}
score() {    # score <key>: single-mark fuzzing exam on those explanations
  $PY run_paper_protocol.py --data-dir $DATA --arm S1 --explanations $OUT/explain_$1/L$L \
      --scorer-model $REASON --scorers fuzz --feature-workers 1 --workers 4 \
      --out $OUT/score_$1/L$L >> $OUT/logs/score_$1_L$L.txt 2>&1
}

echo "$(date '+%F %T') 1. Kimi-K3 explanations"
explain kimi
echo "$(date '+%F %T') 2. Kimi-K3 scored; Qwen3.8-27B and DeepSeek-V4-Flash-reasoning explanations"
score kimi &
for K in qwen reasoning; do explain $K & done
wait
echo "$(date '+%F %T') 3. Qwen3.8-27B and DeepSeek-V4-Flash-reasoning scored"
for K in qwen reasoning; do score $K & done
wait
echo "$(date '+%F %T') ALL DONE"
