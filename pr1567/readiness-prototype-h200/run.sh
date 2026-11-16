#!/usr/bin/env bash
# Experimental, unpublished readiness gate comparison for #1567.
set -euo pipefail
ROOT=/home/bench/praxis/issue-bench
PY=/home/bench/praxis/mstar-venv/bin/python
RESULTS=$ROOT/results1567-readiness
CFG=$ROOT/chatterbox-perf.yaml
export PATH=/home/bench/praxis/mstar-venv/bin:$PATH
export MSTAR_SERVER_BIN=$ROOT/mstar-server
export TORCHDYNAMO_DISABLE=1
export MSTAR_SHM_ARENA=1
export MSTAR_SPEC_PEEK_FOR_FAIRNESS=0
export MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=8
mkdir -p "$RESULTS"
printf 'pair,order,arm,burst,sweep_dir\n' > "$RESULTS/manifest.csv"
cleanup() {
  if [[ -n ${SERVER:-} ]]; then "$SERVER" stop issue1567ready >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
run_arm() {
  local pair=$1 order=$2 arm=$3
  local SRC=$ROOT/pr1567-1959
  if [[ $arm == proto ]]; then SRC=$ROOT/pr1567-proto-readiness; fi
  export PYTHONPATH=$SRC
  SERVER=$SRC/.claude/skills/benchmarking/scripts/server.sh
  cleanup
  "$SERVER" start issue1567ready 8488 --timeout 900 -- "$PY" -m mstar.cli.main serve chatterbox \
    --config "$CFG" --gpus 0 --port 8488
  local OUT=$RESULTS/$arm-$pair
  mkdir -p "$OUT/sweep"
  flock -n /home/bench/gpu.lock "$PY" "$ROOT/mstar-chatterbox-burst.py" \
    --url http://127.0.0.1:8488/v1/audio/speech --arm "$arm" --start "$order" \
    --out "$OUT/burst.json"
  flock -n /home/bench/gpu.lock "$PY" "$ROOT/mstar-chatterbox-sweep-original.py" \
    --url http://127.0.0.1:8488/v1/audio/speech --modes mixed --conc 8,16 \
    --n-per-lane 6 --min-n 48 --out "$OUT/sweep"
  printf '%s,%s,%s,%s,%s\n' "$pair" "$order" "$arm" \
    "$arm-$pair/burst.json" "$arm-$pair/sweep" >> "$RESULTS/manifest.csv"
  cp "${SERVER_STATE_DIR:-/tmp/servers_$USER}/issue1567ready.log" "$OUT/server.log"
  cleanup
}
run_arm 1 1 pr
run_arm 1 2 proto
run_arm 2 3 proto
run_arm 2 4 pr
