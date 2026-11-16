#!/usr/bin/env bash
# Current-main versus #631, fresh-server ABBA with matched inputs.
set -euo pipefail
ROOT=/home/bench/praxis/issue-bench
PY=/home/bench/praxis/mstar-venv/bin/python
RESULTS=$ROOT/results631
CFG=$ROOT/cosyvoice3-perf.yaml
mkdir -p "$RESULTS"
head -n 27 "$ROOT/t2s_x3.txt" > "$ROOT/t2s_27.txt"
printf 'pair,order,arm,c1,c8\n' > "$RESULTS/manifest.csv"
export PATH=/home/bench/praxis/mstar-venv/bin:$PATH
export TORCHDYNAMO_DISABLE=1
export MSTAR_SHM_ARENA=1
SERVER=""
cleanup() {
  if [[ -n "$SERVER" ]]; then "$SERVER" stop issue631 >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
run_arm() {
  local pair=$1 order=$2 arm=$3
  local SRC=$ROOT/main-2c8ff
  export MSTAR_SERVER_BIN=$ROOT/mstar-server
  if [[ $arm == pr ]]; then
    SRC=$ROOT/pr631-f6ed
    export MSTAR_SERVER_BIN=$ROOT/mstar-server-pr631
  fi
  export PYTHONPATH=$SRC
  SERVER=$SRC/.claude/skills/benchmarking/scripts/server.sh
  cleanup
  "$SERVER" start issue631 18800 --timeout 1200 -- "$PY" -m mstar.cli.main serve cosyvoice3 \
    --config "$CFG" --gpus 0 --port 18800
  local OUT=$RESULTS/$arm-$pair
  mkdir -p "$OUT"
  flock -n /home/bench/gpu.lock "$PY" -m benchmark.tts_speech_bench \
    --engine mstar --url http://127.0.0.1:18800 --model cosyvoice3 \
    --voice zero_shot --seed 7 --sentences "$ROOT/t2s_27.txt" \
    --concurrency 1 --repeats 1 --warmup 8 --label "$arm-c1" --out "$OUT/c1.json"
  flock -n /home/bench/gpu.lock "$PY" -m benchmark.tts_speech_bench \
    --engine mstar --url http://127.0.0.1:18800 --model cosyvoice3 \
    --voice zero_shot --seed 7 --sentences "$ROOT/t2s_x3.txt" \
    --concurrency 8 --repeats 1 --warmup 16 --label "$arm-c8" --out "$OUT/c8.json"
  printf '%s,%s,%s,%s,%s\n' "$pair" "$order" "$arm" \
    "$arm-$pair/c1.json" "$arm-$pair/c8.json" >> "$RESULTS/manifest.csv"
  cp "${SERVER_STATE_DIR:-/tmp/servers_$USER}/issue631.log" "$OUT/server.log"
  cleanup
  SERVER=""
}
run_arm 1 1 main
run_arm 1 2 pr
run_arm 2 3 pr
run_arm 2 4 main
run_arm 3 5 main
run_arm 3 6 pr
