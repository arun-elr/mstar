#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/bench/praxis/issue-bench
SRC=$ROOT/metrics-diag
PY=/home/bench/praxis/mstar-venv/bin/python
SERVER=$SRC/.claude/skills/benchmarking/scripts/server.sh
RESULTS=$ROOT/results735-rust
export PATH=/home/bench/praxis/mstar-venv/bin:$PATH
export PYTHONPATH=$SRC
export TORCHDYNAMO_DISABLE=1
export MSTAR_SHM_ARENA=1
export MSTAR_SERVER_BIN=$ROOT/mstar-server
mkdir -p "$RESULTS"
cd "$SRC"

scraper_pid=
cleanup() {
  if [[ -n "$scraper_pid" ]]; then
    kill "$scraper_pid" 2>/dev/null || true
    scraper_pid=
  fi
  "$SERVER" stop issues735 >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_arm() {
  local block=$1 arm=$2
  cleanup
  export MSTAR_METRICS_DIAG_STAGE=$arm
  local metrics_flag=()
  if [[ "$arm" != off ]]; then metrics_flag=(--enable-metrics); fi
  "$SERVER" start issues735 8000 --timeout 600 -- "$PY" -m mstar.api_server.entrypoint \
    --config "$ROOT/qwen35-metrics.yaml" --port 8000 \
    --tensor-comm-protocol SHM --timeout 60 "${metrics_flag[@]}"
  if [[ "$arm" != off ]]; then
    while :; do
      date +%s
      curl -fsS http://127.0.0.1:8000/metrics
      sleep 5
    done > "$RESULTS/$arm-$block-metrics.txt" 2>&1 &
    scraper_pid=$!
  fi
  flock -n /home/bench/gpu.lock "$PY" -m benchmark.runner \
    --url http://127.0.0.1:8000 --model qwen3.5 --model-id Qwen/Qwen3.5-0.8B \
    --request-type text_to_text --profiling-type closed_loop --max-concurrency 64 \
    --num-requests 256 --num-warmup 4 --ignore-eos --output-len-min 256 \
    --output-len-max 256 --output-dir "$RESULTS/$arm-$block-warm" \
    > "$RESULTS/$arm-$block-warm.log" 2>&1
  flock -n /home/bench/gpu.lock "$PY" -m benchmark.runner \
    --url http://127.0.0.1:8000 --model qwen3.5 --model-id Qwen/Qwen3.5-0.8B \
    --request-type text_to_text --profiling-type closed_loop --max-concurrency 64 \
    --num-requests 2600 --num-warmup 4 --ignore-eos --output-len-min 256 \
    --output-len-max 256 --output-dir "$RESULTS/$arm-$block" \
    > "$RESULTS/$arm-$block.log" 2>&1
  cp "/tmp/servers_bench/issues735.log" "$RESULTS/$arm-$block-server.log"
  "$PY" - "$RESULTS/$arm-$block/results.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print(sys.argv[1], report['completed'], report['failed'], report['aggregate']['text_token_throughput'], flush=True)
PY
}

run_arm 1 off
run_arm 1 full
run_arm 2 full
run_arm 2 off
run_arm 3 off
run_arm 3 full
run_arm 4 full
run_arm 4 off
