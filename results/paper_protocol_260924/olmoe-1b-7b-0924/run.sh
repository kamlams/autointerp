#!/bin/bash
# The protocol of ../../paper_protocol_260920/ (published explainer, single-mark fuzzing exam) on the
# 84 latents of the collaborator's feature_to_test.txt of 2026-09-24: layer 1 (expert 9: 20 positive
# + 9 negative latents) and layer 15 (expert 39: 32 positive + 23 negative latents).
#
#   explainers: Kimi-K3 (first), Qwen3.8-27B, DeepSeek-V4-Flash-reasoning; 32768-token budget for all
#   examiner:   DeepSeek-V4-Flash-reasoning
#
# Paths are relative to the repository root. Resumable: a latent whose JSON exists
# is skipped, so a second pass retries the explanations that came back empty. The gateway is shared:
# at most about 20 calls are in flight at once, as in the 2026-09-20 run.
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
OUT=outputs/paper260924_olmoe-1b-7b-0924
REASON=deepseek-ai/DeepSeek-V4-Flash-reasoning
declare -A MODEL=( [kimi]=moonshotai/Kimi-K3 [qwen]=Qwen/Qwen3.8-27B [reasoning]=$REASON )
LAYERS="1 15"
mkdir -p $OUT/logs

explain() {  # explain <key> <layer>: two passes, so an empty reply is retried once
  for pass in 1 2; do
    $PY run_paper_protocol.py --data-dir outputs/feature-data/moein_L$2_260924 --arm P0 \
        --explainer-model ${MODEL[$1]} --explain-max-tokens 32768 --scorers none \
        --feature-workers 3 --out $OUT/explain_$1/L$2 >> $OUT/logs/explain_$1_L$2.txt 2>&1
  done
}
score() {    # score <key> <layer>: single-mark fuzzing exam on those explanations
  $PY run_paper_protocol.py --data-dir outputs/feature-data/moein_L$2_260924 --arm S1 \
      --explanations $OUT/explain_$1/L$2 --scorer-model $REASON --scorers fuzz \
      --feature-workers 1 --workers 4 --out $OUT/score_$1/L$2 >> $OUT/logs/score_$1_L$2.txt 2>&1
}

echo "$(date '+%F %T') 1. Kimi-K3 explanations"
for L in $LAYERS; do explain kimi $L & done; wait
echo "$(date '+%F %T') 2. Kimi-K3 scored; Qwen3.8-27B and DeepSeek-V4-Flash-reasoning explanations"
for L in $LAYERS; do score kimi $L & done
for K in qwen reasoning; do for L in $LAYERS; do explain $K $L & done; done
wait
echo "$(date '+%F %T') 3. Qwen3.8-27B and DeepSeek-V4-Flash-reasoning scored"
for K in qwen reasoning; do for L in $LAYERS; do score $K $L & done; done
wait
echo "$(date '+%F %T') ALL DONE"
