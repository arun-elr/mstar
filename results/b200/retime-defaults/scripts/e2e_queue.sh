#!/bin/bash
while tmux has-session -t =retime 2>/dev/null || tmux has-session -t =retime2 2>/dev/null || tmux has-session -t =retime3 2>/dev/null; do sleep 20; done
export E2E_DIR=$HOME/e2e-cosmos3 RUNS=runs PHASES=c1 E2E_NO_FA3=1
for v in default before; do
  echo "[$(date +%T)] start $v"; bash $E2E_DIR/scripts/run_variant.sh $v; echo "[$(date +%T)] end $v rc=$?"
done
echo ALL_DONE
