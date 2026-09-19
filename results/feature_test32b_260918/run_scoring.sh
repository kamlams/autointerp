#!/bin/bash
# Explain and score the second batch of 96 latents (data260913/feature_to_test.txt, updated 2026-09-18;
# layers 0, 7, 15 of the moein SAE). Latents already explained in the 2026-09-16 batch are reused
# verbatim (same SAE, same corpus, same protocol); only the rest are sent to the model.
set -u
RI=$(cd "$(dirname "$0")/../.." && pwd)   # repository root
PY=${PY:-python}            # env with torch + transformers (extraction, reports)
PYA=${PYA:-python}          # env with torch + requests (run_autointerp.py)
MODEL=deepseek-ai/DeepSeek-V4-Flash-reasoning
R=$RI/results/feature_test32b_260918
cd $RI
log() { echo "$(date '+%F %T') $*"; }
score() { local name=$1 out=$2; shift 2; $PYA run_autointerp.py --data-dir outputs/feature-data/$name --features "$@" --model $MODEL --workers 8 --out $out > $out.log 2>&1; }

for L in 0 7 15; do
  new=moein_L${L}_test32b; old=moein_L${L}_test32
  mkdir -p outputs/interp_$new
  reused=0
  for f in outputs/feature-data/$new/feature*.pt; do
    fid=$(basename $f | sed 's/feature\([0-9]*\)__.*/\1/')
    if [ -f outputs/interp_$old/feature$fid.json ] && [ ! -f outputs/interp_$new/feature$fid.json ]; then
      cp outputs/interp_$old/feature$fid.json outputs/interp_$new/; reused=$((reused + 1))
    fi
  done
  pending=(); for f in $(ls outputs/feature-data/$new/feature*.pt | sort); do
    fid=$(basename $f | sed 's/feature\([0-9]*\)__.*/\1/')
    [ -f outputs/interp_$new/feature$fid.json ] || pending+=($f)
  done
  log "layer $L: reused $reused explanations, scoring ${#pending[@]}"
  half=$(( ${#pending[@]} / 2 ))
  mkdir -p outputs/interp_${new}_a outputs/interp_${new}_b
  [ $half -gt 0 ] && score $new outputs/interp_${new}_a "${pending[@]:0:$half}" &
  score $new outputs/interp_${new}_b "${pending[@]:$half}" &
done
log "scoring processes started"
wait
log "scoring finished"
for L in 0 7 15; do
  new=moein_L${L}_test32b
  cp outputs/interp_${new}_a/feature*.json outputs/interp_${new}_b/feature*.json outputs/interp_$new/ 2>/dev/null
  missing=$($PY -c "
import json, glob, os
want = {int(k) for k in json.load(open('outputs/feature-data/$new/sampling.json'))['features']}
have = {int(os.path.basename(f)[7:-5]) for f in glob.glob('outputs/interp_$new/feature*.json')}
print(' '.join(str(i) for i in sorted(want - have)))")
  log "$new: $(ls outputs/interp_$new/feature*.json | wc -l) of 32; give-ups to retry: [$missing]"
  if [ -n "$missing" ]; then
    retry=(); for i in $missing; do retry+=(outputs/feature-data/$new/feature${i}__*.pt); done
    mkdir -p outputs/interp_${new}_retry; score $new outputs/interp_${new}_retry "${retry[@]}" &
  fi
done
wait
for L in 0 7 15; do
  new=moein_L${L}_test32b
  cp outputs/interp_${new}_retry/feature*.json outputs/interp_$new/ 2>/dev/null
  mkdir -p $R/$new
  cp outputs/interp_$new/feature*.json outputs/feature-data/$new/sampling.json $R/$new/
  cat outputs/interp_${new}_{a,b,retry}.log > $R/$new/run.log 2>/dev/null
  $PY summarize_features.py outputs/feature-data/$new outputs/interp_$new $R/features_L$L.md \
    --title "OLMoE layer $L, moein SAE (data260913) — the 32 latents of the 2026-09-18 feature_to_test.txt"
  log "$new: $(ls outputs/interp_$new/feature*.json | wc -l) / 32 scored"
done
log "done"
