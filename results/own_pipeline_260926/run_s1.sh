#!/bin/bash
# The single-mark fuzzing exam of ../paper_protocol_260920/ (run_paper_protocol.py --arm S1: examiner
# DeepSeek-V4-Flash-reasoning, 100 positives, 50 type-A + 50 type-B negatives, five probes per query,
# the same seed and windows as 2026-09-20) applied to the 30-word explanations in this folder, so this
# pipeline's explanations can be compared with the published-protocol explanations on the same exam.
#
#   all three explainers at once, each unit (set x layer) in one process scoring several latents at a
#   time (about one per eight latents of the unit), two examiner calls per latent: about 150 calls in
#   flight. (First started one explainer after another; switched to this at the user's request.)
#
# Resumable: a latent whose JSON exists is skipped. Paths are relative to the repository root.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
OUT=outputs/own260926_s1
R=results/own_pipeline_260926
EXAMINER=deepseek-ai/DeepSeek-V4-Flash-reasoning
declare -A EXPL=( [kimi]=kimi-k3 [qwen]=qwen3.8-27b [reasoning]=deepseek-v4-flash-reasoning )
# set:layer:data dir:latents scored at once
UNITS="olmoe-1b-7b-0924/active-on-to:L0:moein_L0_test32b:4 olmoe-1b-7b-0924/active-on-to:L7:moein_L7_test32b:4
       olmoe-1b-7b-0924/active-on-to:L15:moein_L15_test32b:4 olmoe-1b-7b-0924/expert-linked:L1:moein_L1_260924:4
       olmoe-1b-7b-0924/expert-linked:L15:moein_L15_260924:7 qwen3-30b-a3b/expert-linked:L24:qwen3_moein_L24_260926:3"
mkdir -p $OUT/logs

score() {  # score <explainer> <set> <layer> <data dir> <feature workers> <examiner workers>
  local log=$OUT/logs/$(echo $2 | tr / _)_$1_$3.txt
  mkdir -p $OUT/$2/${EXPL[$1]}
  $PY run_paper_protocol.py --data-dir outputs/feature-data/$4 --arm S1 \
      --explanations $R/$2/explanations-and-scores/${EXPL[$1]}/$3 --scorer-model $EXAMINER --scorers fuzz \
      --feature-workers $5 --workers $6 --out $OUT/$2/${EXPL[$1]}/$3 >> $log 2>&1
}
each() {   # each <explainer> <examiner workers>
  for u in $UNITS; do IFS=: read S L D F <<< "$u"; score $1 $S $L $D $F $2 & done
}

echo "$(date '+%F %T') all three explainers' explanations on the single-mark exam"
for K in kimi qwen reasoning; do each $K 2; done; wait
echo "$(date '+%F %T') ALL DONE"
