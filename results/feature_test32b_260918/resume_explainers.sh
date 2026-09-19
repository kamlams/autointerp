#!/bin/bash
# Resume the Kimi / Qwen explanation runs over the 2026-09-18 batch after an allocation ended.
# Idempotent: merges whatever the previous attempt produced, explains only the features that are
# still missing, then builds the reports and the second-examiner cross-check. Safe to run repeatedly.
set -u
RI=$(cd "$(dirname "$0")/../.." && pwd)   # repository root
PY=${PY:-python}            # env with torch + transformers (extraction, reports)
PYA=${PYA:-python}          # env with torch + requests (run_autointerp.py)
EXAMINER=deepseek-ai/DeepSeek-V4-Flash-reasoning
CHECK=deepseek-ai/DeepSeek-V4-Flash
R=$RI/results/feature_test32b_260918/by_explainer
WORKERS=${WORKERS:-6}
cd $RI
mkdir -p $R
log() { echo "$(date '+%F %T') $*"; }

missing_of() {  # tag layer -> ids still to explain
  local tag=$1 L=$2
  $PY -c "
import json, glob, os
want = {int(k) for k in json.load(open('outputs/feature-data/moein_L${L}_test32b/sampling.json'))['features']}
have = set()
for d in ('outputs/${tag}_L${L}', 'outputs/${tag}_L${L}_a', 'outputs/${tag}_L${L}_b', 'outputs/${tag}_L${L}_retry'):
    have |= {int(os.path.basename(f)[7:-5]) for f in glob.glob(d + '/feature*.json')}
print(' '.join(str(i) for i in sorted(want - have)))"
}

for spec in "kimi:moonshotai/Kimi-K3" "qwen:Qwen/Qwen3.8-27B"; do
  tag=${spec%%:*}; model=${spec#*:}
  for L in 0 7 15; do
    mkdir -p outputs/${tag}_L${L}
    cp outputs/${tag}_L${L}_a/feature*.json outputs/${tag}_L${L}/ 2>/dev/null
    cp outputs/${tag}_L${L}_b/feature*.json outputs/${tag}_L${L}/ 2>/dev/null
    cp outputs/${tag}_L${L}_retry/feature*.json outputs/${tag}_L${L}/ 2>/dev/null
    ids=$(missing_of $tag $L)
    n=$(echo $ids | wc -w)
    log "$tag L$L: $(ls outputs/${tag}_L${L}/feature*.json 2>/dev/null | wc -l)/32 done, $n to go"
    [ "$n" -eq 0 ] && continue
    files=(); for i in $ids; do files+=(outputs/feature-data/moein_L${L}_test32b/feature${i}__*.pt); done
    half=$(( ${#files[@]} / 2 ))
    if [ $half -gt 0 ]; then
      mkdir -p outputs/${tag}_L${L}_r1; $PYA run_autointerp.py --data-dir outputs/feature-data/moein_L${L}_test32b \
        --features "${files[@]:0:$half}" --model "$model" --scorer-model $EXAMINER --workers $WORKERS \
        --out outputs/${tag}_L${L}_r1 > outputs/${tag}_L${L}_r1.log 2>&1 &
    fi
    mkdir -p outputs/${tag}_L${L}_r2; $PYA run_autointerp.py --data-dir outputs/feature-data/moein_L${L}_test32b \
      --features "${files[@]:$half}" --model "$model" --scorer-model $EXAMINER --workers $WORKERS \
      --out outputs/${tag}_L${L}_r2 > outputs/${tag}_L${L}_r2.log 2>&1 &
  done
  wait
  for L in 0 7 15; do
    cp outputs/${tag}_L${L}_r1/feature*.json outputs/${tag}_L${L}/ 2>/dev/null
    cp outputs/${tag}_L${L}_r2/feature*.json outputs/${tag}_L${L}/ 2>/dev/null
    mkdir -p $R/$tag/L$L; cp outputs/${tag}_L${L}/feature*.json $R/$tag/L$L/
    cat outputs/${tag}_L${L}_*.log > $R/$tag/L$L/run.log 2>/dev/null
    $PY summarize_features.py outputs/feature-data/moein_L${L}_test32b outputs/${tag}_L${L} $R/$tag/features_L$L.md \
      --title "OLMoE layer $L, moein SAE — the 2026-09-18 latents explained by $model, scored by $EXAMINER" || true
    log "$tag L$L: $(ls outputs/${tag}_L${L}/feature*.json | wc -l)/32 after this pass"
  done
done

# second-examiner cross-check of all three arms, only for layers that are complete
for L in 0 7 15; do
  for pair in "deepseek:outputs/interp_moein_L${L}_test32b" "kimi:outputs/kimi_L${L}" "qwen:outputs/qwen_L${L}"; do
    tag=${pair%%:*}; src=${pair#*:}
    [ "$(ls $src/feature*.json 2>/dev/null | wc -l)" -eq 32 ] || { log "cross-check skipped: $tag L$L incomplete"; continue; }
    out=outputs/check_${tag}_L${L}
    [ "$(ls $out/feature*.json 2>/dev/null | wc -l)" -eq 32 ] && continue
    mkdir -p $out
    $PYA run_autointerp.py --data-dir outputs/feature-data/moein_L${L}_test32b \
      --features $(ls outputs/feature-data/moein_L${L}_test32b/feature*.pt | sort | tr '\n' ' ') \
      --reuse-explanations $src --scorers detect,fuzz --model $CHECK --workers 8 --out $out > $out.log 2>&1 &
  done
done
wait
log "resume pass complete"
