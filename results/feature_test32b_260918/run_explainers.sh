#!/bin/bash
# Re-explain the whole 2026-09-18 feature_to_test.txt batch (96 latents, layers 0/7/15) with two
# further explainers, keeping the examiner fixed to the one the existing DeepSeek arm used, so the
# three arms are directly comparable. Then a cheap cross-check of all three with a second examiner.
set -u
RI=$(cd "$(dirname "$0")/../.." && pwd)   # repository root
PY=${PY:-python}            # env with torch + transformers (extraction, reports)
PYA=${PYA:-python}          # env with torch + requests (run_autointerp.py)
EXAMINER=deepseek-ai/DeepSeek-V4-Flash-reasoning
CHECK=deepseek-ai/DeepSeek-V4-Flash
R=$RI/results/feature_test32b_260918/by_explainer
cd $RI
mkdir -p $R
log() { echo "$(date '+%F %T') $*"; }

explain() {  # tag model workers
  local tag=$1 model=$2 w=$3
  for L in 0 7 15; do
    local name=moein_L${L}_test32b
    files=($(ls outputs/feature-data/$name/feature*.pt | sort)); half=$(( ${#files[@]} / 2 ))
    mkdir -p outputs/${tag}_L${L}_a outputs/${tag}_L${L}_b
    $PYA run_autointerp.py --data-dir outputs/feature-data/$name --features "${files[@]:0:$half}" \
      --model "$model" --scorer-model $EXAMINER --workers $w --out outputs/${tag}_L${L}_a > outputs/${tag}_L${L}_a.log 2>&1 &
    $PYA run_autointerp.py --data-dir outputs/feature-data/$name --features "${files[@]:$half}" \
      --model "$model" --scorer-model $EXAMINER --workers $w --out outputs/${tag}_L${L}_b > outputs/${tag}_L${L}_b.log 2>&1 &
  done
  wait
  log "$tag: first pass done"
  for L in 0 7 15; do
    local name=moein_L${L}_test32b
    mkdir -p outputs/${tag}_L${L}
    cp outputs/${tag}_L${L}_a/feature*.json outputs/${tag}_L${L}_b/feature*.json outputs/${tag}_L${L}/ 2>/dev/null
    missing=$($PY -c "
import json, glob, os
want = {int(k) for k in json.load(open('outputs/feature-data/$name/sampling.json'))['features']}
have = {int(os.path.basename(f)[7:-5]) for f in glob.glob('outputs/${tag}_L${L}/feature*.json')}
print(' '.join(str(i) for i in sorted(want - have)))")
    if [ -n "$missing" ]; then
      retry=(); for i in $missing; do retry+=(outputs/feature-data/$name/feature${i}__*.pt); done
      mkdir -p outputs/${tag}_L${L}_retry
      $PYA run_autointerp.py --data-dir outputs/feature-data/$name --features "${retry[@]}" \
        --model "$model" --scorer-model $EXAMINER --workers $w --out outputs/${tag}_L${L}_retry > outputs/${tag}_L${L}_retry.log 2>&1 &
    fi
  done
  wait
  for L in 0 7 15; do
    cp outputs/${tag}_L${L}_retry/feature*.json outputs/${tag}_L${L}/ 2>/dev/null
    mkdir -p $R/$tag/L$L
    cp outputs/${tag}_L${L}/feature*.json $R/$tag/L$L/
    cat outputs/${tag}_L${L}_{a,b,retry}.log > $R/$tag/L$L/run.log 2>/dev/null
    $PY summarize_features.py outputs/feature-data/moein_L${L}_test32b outputs/${tag}_L${L} $R/$tag/features_L$L.md \
      --title "OLMoE layer $L, moein SAE — the 2026-09-18 latents explained by $model, scored by $EXAMINER"
    log "$tag L$L: $(ls outputs/${tag}_L${L}/feature*.json | wc -l) / 32 explained"
  done
}

explain kimi  "moonshotai/Kimi-K3"  6
explain qwen  "Qwen/Qwen3.8-27B"    6

# cross-check: the same three explanation sets scored by a second, non-reasoning examiner
for L in 0 7 15; do
  for pair in "deepseek:outputs/interp_moein_L${L}_test32b" "kimi:outputs/kimi_L${L}" "qwen:outputs/qwen_L${L}"; do
    tag=${pair%%:*}; src=${pair#*:}
    out=outputs/check_${tag}_L${L}; mkdir -p $out
    $PYA run_autointerp.py --data-dir outputs/feature-data/moein_L${L}_test32b \
      --features $(ls outputs/feature-data/moein_L${L}_test32b/feature*.pt | sort | tr '\n' ' ') \
      --reuse-explanations $src --scorers detect,fuzz --model $CHECK --workers 8 --out $out > $out.log 2>&1 &
  done
done
wait
log "cross-check with $CHECK done"
log "all done"
