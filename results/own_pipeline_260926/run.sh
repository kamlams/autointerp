#!/bin/bash
# This repository's own pipeline (run_autointerp.py: statistics card + 24 snippets, LABEL +
# EXPLANATION, detection + fuzzing) with the EXPLANATION capped at 30 words, on the latents of the
# two published-protocol runs:
#
#   to96          OLMoE-1B-7B-0924 L0/L7/L15, the 96 latents active on " to" (paper_protocol_260920)
#   olmoe-expert  OLMoE L1 expert 9 and L15 expert 39 latents (paper_protocol_260924)
#   qwen3-expert  Qwen3-30B-A3B L24 expert 9 latents, with the collaborator's 2026-09-26 fix 8006 -> 8066
#
#   explainers: Kimi-K3 (first), then Qwen3.8-27B and DeepSeek-V4-Flash-reasoning together
#   examiner:   DeepSeek-V4-Flash-reasoning; afterwards DeepSeek-V4-Flash re-scores every explanation
#
# Every unit (set x layer) is split into two halves run in parallel. A second pass (<part>_retry)
# retries the latents that failed. Paths are relative to the repository root.
# PHASES selects the phases to run (default "1 2 3"). On 2026-09-26 phase 1 was left finishing one
# slow half (to96 L0, Kimi-K3 on dense latents) while phases 2 and 3 were started with PHASES="2 3";
# phase 3 therefore first waits until no explainer run of this directory is left.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
OUT=outputs/own260926
EXAMINER=deepseek-ai/DeepSeek-V4-Flash-reasoning
CHECK=deepseek-ai/DeepSeek-V4-Flash
declare -A MODEL=( [kimi]=moonshotai/Kimi-K3 [qwen]=Qwen/Qwen3.8-27B [reasoning]=$EXAMINER )
UNITS="to96:L0:moein_L0_test32b to96:L7:moein_L7_test32b to96:L15:moein_L15_test32b
       olmoe-expert:L1:moein_L1_260924 olmoe-expert:L15:moein_L15_260924
       qwen3-expert:L24:qwen3_moein_L24_260926"
mkdir -p $OUT/logs

missing() {  # missing <dir> <feature files...>: the files with no feature JSON in <dir>
  local dir=$1; shift
  for f in "$@"; do id=$(basename $f | sed 's/^feature\([0-9]*\)__.*/\1/'); [ -f $dir/feature$id.json ] || echo $f; done
}

explain_unit() {  # explain_unit <explainer> <set> <layer> <data dir> <examiner workers>
  local K=$1 S=$2 L=$3 D=outputs/feature-data/$4 W=$5
  local files=($(ls $D/feature*.pt | sort)); local half=$(( (${#files[@]} + 1) / 2 ))
  for part in a b; do
    local sel; if [ $part = a ]; then sel=("${files[@]:0:$half}"); else sel=("${files[@]:$half}"); fi
    ( $PY run_autointerp.py --data-dir $D --features "${sel[@]}" --model ${MODEL[$K]} --scorer-model $EXAMINER \
          --max-explanation-words 30 --workers $W --out $OUT/$S/$K/${L}_$part >> $OUT/logs/${S}_${K}_${L}_$part.txt 2>&1
      todo=($(missing $OUT/$S/$K/${L}_$part "${sel[@]}"))
      [ ${#todo[@]} -gt 0 ] && $PY run_autointerp.py --data-dir $D --features "${todo[@]}" --model ${MODEL[$K]} \
          --scorer-model $EXAMINER --max-explanation-words 30 --workers $W --out $OUT/$S/$K/${L}_${part}_retry \
          >> $OUT/logs/${S}_${K}_${L}_${part}_retry.txt 2>&1 ) &
  done
}

check_unit() {  # check_unit <explainer> <set> <layer> <data dir>: re-score every stored explanation with $CHECK
  local K=$1 S=$2 L=$3 D=outputs/feature-data/$4
  for src in $OUT/$S/$K/${L}_*; do
    [ -d $src ] || continue
    local ids=($(ls $src/feature*.json 2>/dev/null | sed 's/.*feature\([0-9]*\)\.json/\1/'))
    [ ${#ids[@]} -gt 0 ] || continue
    local fs=(); for id in "${ids[@]}"; do fs+=($(ls $D/feature${id}__*.pt)); done
    $PY run_autointerp.py --data-dir $D --features "${fs[@]}" --reuse-explanations $src --scorers detect,fuzz \
        --model $CHECK --workers 8 --out $OUT/$S/check_$K/$(basename $src) >> $OUT/logs/${S}_check_${K}_$(basename $src).txt 2>&1 &
  done
}

each() {  # each <function> <explainer> [extra args]: run <function> for every unit
  local fn=$1 K=$2; shift 2
  for u in $UNITS; do IFS=: read S L D <<< "$u"; $fn $K $S $L $D "$@"; done
}

PHASES=${PHASES:-"1 2 3"}
phase() { [[ " $PHASES " == *" $1 "* ]]; }
explainers_running() { pgrep -f "run_autointerp.py .*--out $OUT/[^ ]*/(kimi|qwen|reasoning)/" > /dev/null; }

if phase 1; then
  echo "$(date '+%F %T') 1. Kimi-K3 explanations"
  each explain_unit kimi 6; wait
fi
if phase 2; then
  echo "$(date '+%F %T') 2. Qwen3.8-27B and DeepSeek-V4-Flash-reasoning explanations"
  each explain_unit qwen 4; each explain_unit reasoning 4; wait
fi
if phase 3; then
  # an earlier invocation may still be writing explanations (and may start a retry pass after it)
  while explainers_running || { sleep 120; explainers_running; }; do sleep 60; done
  echo "$(date '+%F %T') 3. DeepSeek-V4-Flash re-scores every explanation"
  for K in kimi qwen reasoning; do each check_unit $K; done; wait
fi
echo "$(date '+%F %T') ALL DONE"
