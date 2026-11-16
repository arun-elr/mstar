#!/usr/bin/env bash
# Chatterbox peek-off cap 16/32 rotation on the PR code, same checkpoint/clients as #1567 A/B.
set -euo pipefail
ROOT=/home/bench/praxis/issue-bench
PY=/home/bench/praxis/mstar-venv/bin/python
RESULTS=$ROOT/results1567-caps
CFG=$ROOT/chatterbox-perf.yaml
SRC=$ROOT/pr1567-1959
SERVER=$SRC/.claude/skills/benchmarking/scripts/server.sh
export PATH=/home/bench/praxis/mstar-venv/bin:$PATH
export PYTHONPATH=$SRC
export MSTAR_SERVER_BIN=$ROOT/mstar-server
export TORCHDYNAMO_DISABLE=1
export MSTAR_SHM_ARENA=1
export MSTAR_SPEC_PEEK_FOR_FAIRNESS=0
mkdir -p "$RESULTS"
printf 'order,cap,rep,burst,sweep_dir\n' > "$RESULTS/manifest.csv"
cleanup() { "$SERVER" stop issue1567cap >/dev/null 2>&1 || true; }
trap cleanup EXIT
run_cap() {
  local order=$1 cap=$2 rep=$3
  export MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=$cap
  cleanup
  "$SERVER" start issue1567cap 8487 --timeout 900 -- "$PY" -m mstar.cli.main serve chatterbox \
    --config "$CFG" --gpus 0 --port 8487
  local OUT=$RESULTS/cap${cap}-${rep}
  mkdir -p "$OUT/sweep"
  flock -n /home/bench/gpu.lock "$PY" "$ROOT/mstar-chatterbox-burst.py" \
    --url http://127.0.0.1:8487/v1/audio/speech --arm pr --start "$order" \
    --out "$OUT/burst.json"
  flock -n /home/bench/gpu.lock "$PY" "$ROOT/mstar-chatterbox-sweep-original.py" \
    --url http://127.0.0.1:8487/v1/audio/speech --modes mixed --conc 8,16 \
    --n-per-lane 6 --min-n 48 --out "$OUT/sweep"
  printf '%s,%s,%s,%s,%s\n' "$order" "$cap" "$rep" \
    "cap${cap}-${rep}/burst.json" "cap${cap}-${rep}/sweep" >> "$RESULTS/manifest.csv"
  cp "${SERVER_STATE_DIR:-/tmp/servers_$USER}/issue1567cap.log" "$OUT/server.log"
  cleanup
}
run_cap 1 16 1
run_cap 2 32 1
run_cap 3 32 2
run_cap 4 16 2
