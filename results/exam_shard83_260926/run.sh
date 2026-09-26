#!/bin/bash
# The single-mark fuzzing exam of ../paper_protocol_260920/ (run_paper_protocol.py --arm S1, examiner
# DeepSeek-V4-Flash-reasoning, 100 positives + 50 type-A + 50 type-B negatives, seed 22) rebuilt on
# corpus shard 83 of SmolLM2-135M-10B, for explanations whose evidence came from shard 84:
#   own     the 30-word explanations of ../own_pipeline_260926/ (statistics card + 24 snippets, shard 84)
#   delphi  the published-protocol explanations of ../paper_protocol_260920/ and ../paper_protocol_260924/
#           (40 windows of shard 84)
# Explanation evidence and exam windows therefore come from different documents by construction.
#
#   PHASES="extract exam" results/exam_shard83_260926/run.sh     (default: both)
#
# extract  builds the shard-83 corpus (same 825 x 16 x 128 chunking; one per tokenizer) and extracts the
#          same latents with the same SAEs as the shard-84 data directories, whose sampling.json supplies
#          the SAE, layer, input side and latent list (so no checkpoint path is written here). GPU 0 only.
#          Qwen3 L24 also extracts 8006, the latent the published-protocol run explained in place of 8066.
# exam     run_exam.py: every (side, explainer, set, layer) unit on the shard-83 exam, Kimi-K3 first,
#          at most 60 examiner calls in flight (more overloaded the shared gateway on 2026-09-26).
# Resumable: extraction skips a directory with sampling.json, the exam skips a latent with a JSON.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
OUT=outputs/exam_shard83
PHASES=${PHASES:-extract exam}
mkdir -p $OUT/logs

field() {  # field <sampling.json> <key> [default]
  $PY -c "import json, sys; s = json.load(open(sys.argv[1])); print(s.get(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ''))" "$@"
}
extract() {  # extract <shard-84 data dir> [extra latent ids]
  local src=outputs/feature-data/$1 dst=outputs/feature-data/$1_shard83
  [ -f $dst/sampling.json ] && { echo "$(date '+%F %T') $dst exists"; return; }
  local ids
  ids=$($PY -c "import json, sys; print(' '.join(sorted(json.load(open(sys.argv[1]))['features'], key=int)))" $src/sampling.json)
  echo "$(date '+%F %T') extract $1 on shard 83"
  CUDA_VISIBLE_DEVICES=0 $PY extract_features.py --model "$(field $src/sampling.json model allenai/OLMoE-1B-7B-0924)" \
      --sae "$(field $src/sampling.json sae)" --layer "$(field $src/sampling.json layer)" \
      --input "$(field $src/sampling.json input)" --shard 83 --name $1_shard83 --latents "$ids $2" \
      --device cuda:0 >> $OUT/logs/extract_$1.txt 2>&1
}

if [[ " $PHASES " == *" extract "* ]]; then
  : "${HF_HOME:?export HF_HOME: the Hugging Face cache holding the SmolLM2-135M-10B shards}"
  for d in moein_L0_test32b moein_L7_test32b moein_L15_test32b moein_L1_260924 moein_L15_260924; do extract $d; done
  extract qwen3_moein_L24_260926 8006
  echo "$(date '+%F %T') extraction done"
fi
if [[ " $PHASES " == *" exam "* ]]; then
  $PY results/exam_shard83_260926/run_exam.py --budget ${BUDGET:-60} >> $OUT/logs/exam.txt 2>&1
  echo "$(date '+%F %T') exam done"
fi
