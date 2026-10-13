#!/bin/bash
# The three variants back to back, one GPU job at a time. Before each variant,
# waits (re-checking every 5 min) until no other process or container is on the GPU.
set -u
E=${E2E_DIR:-$HOME/e2e-cosmos3}
gpu_busy() {
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || [ -n "$(docker ps -q)" ]
}
vs=("$@"); [ ${#vs[@]} -eq 0 ] && vs=(before after0 after1)
for v in "${vs[@]}"; do
  for w in $(seq 1000); do
    gpu_busy || break
    echo "[$(date +%T)] GPU busy ($(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';') docker: $(docker ps --format '{{.Image}}' | tr '\n' ' ')); waiting 5 min"
    sleep 300
  done
  echo "[$(date +%T)] start $v"
  bash "$E/scripts/run_variant.sh" "$v"
  rc=$?
  echo "[$(date +%T)] end $v rc=$rc"
  [ "$rc" -eq 0 ] || { echo "ABORT: $v failed"; exit 1; }
done
echo ALL_DONE
