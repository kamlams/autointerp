#!/bin/bash
# Copy the per-latent JSONs and run_config.json of every exam unit from outputs/exam_shard83/ into this
# folder as <side>/<model>/<set>/<explainer>/<layer>/ (calls.jsonl stays in the gitignored outputs/).
cd "$(dirname "$0")/../.."
OUT=outputs/exam_shard83
R=results/exam_shard83_260926
for d in $OUT/own/*/*/*/L* $OUT/delphi/*/*/*/L*; do
  ls $d/feature*.json > /dev/null 2>&1 || continue   # a unit that has not scored a latent yet
  rel=${d#$OUT/}
  mkdir -p $R/$rel
  cp $d/feature*.json $R/$rel/
  [ -f $d/run_config.json ] && cp $d/run_config.json $R/$rel/
done
echo "$(date '+%F %T') collected $(find $R/own $R/delphi -name 'feature*.json' | wc -l) latent records into $R"
