#!/bin/bash
# One variant end to end on one GPU: serve configs/cosmos3_edge.yaml with the
# variant's decoder switches overlaid (as benchmark/wan_vae_decoder/e2e_stream.sh
# does), then run the phases against it and stop the server.
#
#   run_variant.sh before|after0|after1|default
#
# Layout under $E2E_DIR (default ~/e2e-cosmos3): venv/ (the M* environment),
# hf/ (HF_HOME with nvidia/Cosmos3-Edge), mstar-main/ and mstar-pr/ (checkouts,
# each with a REV file naming its commit), scripts/ (this directory).
#
# Phases (peak-memory counters reset between them):
#   kv_c1    kv rollout, 241 frames (8 windows), concurrency 1: 1 warm-up + 3 timed
#   kv_dump  the same request once more with the raw-frame dump on (not timed)
#   kv_c4    kv rollout, concurrency 4 (seeds 0..3): 1 warm-up round + 2 timed
#   ch_c1    chained rollout, 241 frames (10 windows), concurrency 1: 1 warm-up + 2 timed
#   ch_dump  the same chained request with the raw-frame dump on (not timed)
# PHASES=c1 runs only kv_c1 and ch_c1; PHASES=c8 and PHASES=dump as below.
set -u
V="$1"
E=${E2E_DIR:-$HOME/e2e-cosmos3}
R=$E/${RUNS:-runs}/$V
PHASES=${PHASES:-all}   # all | dump (only the two untimed dump requests)
PORT=8100
PY=$E/venv/bin/python
export HF_HOME=$E/hf HF_HUB_OFFLINE=1
# flashinfer JIT-builds its paged-attention kernels with ninja + nvcc (cached in ~/.cache/flashinfer)
export PATH="$E/venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
EDGE_REV=344d602b128d1bbdacb43b08d0a3626f46343e29
case "$V" in
  before) MSTAR=$E/mstar-main; OV='{}' ;;
  after0) MSTAR=$E/mstar-pr; OV='{"windowed_decode_warm_cache": true, "vae_decode_channels_last": true, "windowed_decode_latents_per_call": 0, "vae_decode_cudnn_benchmark": true}' ;;
  after1) MSTAR=$E/mstar-pr; OV='{"windowed_decode_warm_cache": true, "vae_decode_channels_last": true, "windowed_decode_latents_per_call": 1, "vae_decode_cudnn_benchmark": true}' ;;
  default) MSTAR=$E/mstar-pr; OV='{}' ;;  # the PR checkout's own defaults
  *) echo "unknown variant $V" >&2; exit 2 ;;
esac
rm -rf "$R"; mkdir -p "$R"/mem "$R"/dump
log() { echo "[$(date +%T)] $*" | tee -a "$R/run.log"; }
phase_mark() { echo "{\"phase\": \"$1\", \"event\": \"$2\", \"t\": $(date +%s.%N), \"stats_bytes\": $(stat -c %s "$R/stats.txt" 2>/dev/null || echo 0)}" >> "$R/phases.jsonl"; }
mem_snap() { "$PY" - "$R/mem" "$R/mem_$1.json" <<'EOF'
import glob, json, os, sys
recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], "[0-9]*.json")))]
json.dump(recs, open(sys.argv[2], "w"), indent=1)
for r in recs:
    print(f"  pid {r['pid']} resets={r['resets']} max_alloc={r['max_allocated_gib']:.2f} GiB "
          f"max_reserved={r['max_reserved_gib']:.2f} GiB  {r['argv'][:80]}")
EOF
}
reset_peaks() { touch "$R/mem/RESET"; sleep 2; }

log "variant $V MSTAR=$MSTAR rev=$(cat "$MSTAR/REV") overrides=$OV"
log "gpu apps at start: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';') docker: $(docker ps -q | tr '\n' ' ')"
"$PY" - "$MSTAR/configs/cosmos3_edge.yaml" "$R/config.yaml" "$OV" <<'EOF'
import json, sys, yaml
src, dst, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
cfg = yaml.safe_load(open(src))
cfg.setdefault("model_kwargs", {}).update(extra)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF
IMG=$("$PY" -c "from huggingface_hub import hf_hub_download as h; print(h('nvidia/Cosmos3-Edge','assets/example_i2v_input.jpg',revision='$EDGE_REV'))")
log "image $IMG md5 $(md5sum "$IMG" | cut -d' ' -f1)"

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$R/smi.csv" &
SMI=$!
sock=$(mktemp -d); upload=$(mktemp -d)
T_LAUNCH=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  COSMOS3_GEN_CAPTURE_RES=192x320,480x832,640x640 PYTHONPATH="$E/hook:$MSTAR" \
  E2E_MEM_DIR="$R/mem" E2E_DUMP_DIR="$R/dump" \
  setsid "$PY" "$MSTAR/mstar/api_server/entrypoint.py" --config "$R/config.yaml" \
  --socket-path-prefix "$sock/" --upload-dir "$upload/" --port "$PORT" --mooncake-port $((PORT + 1000)) \
  --tensor-comm-protocol SHM --log-stats-file "$R/stats.txt" > "$R/server.log" 2>&1 &
SPID=$!
echo "$SPID" > "$R/server.pid"
stop_server() {
  log "stopping server pgid $SPID"
  kill -INT -- -"$SPID" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- -"$SPID" 2>/dev/null
  sleep 3; kill "$SMI" 2>/dev/null
  log "server stopped; gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
}
trap stop_server EXIT
for i in $(seq 1800); do
  curl -sf "localhost:$PORT/health" > /dev/null && break
  kill -0 "$SPID" 2>/dev/null || { log "server died during startup"; tail -50 "$R/server.log"; exit 1; }
  sleep 1
done
T_READY=$(date +%s.%N)
log "server ready after $(echo "$T_READY - $T_LAUNCH" | bc) s"
echo "{\"launch\": $T_LAUNCH, \"ready\": $T_READY}" > "$R/startup.json"
mem_snap startup | tee -a "$R/run.log"

client() {  # phase, then client args
  local ph="$1"; shift
  reset_peaks
  phase_mark "$ph" start
  log "phase $ph: $*"
  "$PY" "$E/scripts/stream_client.py" --port "$PORT" --image "$IMG" --tag "$V/$ph" --out "$R/client_$ph.json" "$@" \
    2>&1 | tee -a "$R/client.log"
  phase_mark "$ph" end
  sleep 1.5
  mem_snap "$ph" | tee -a "$R/run.log"
  grep -i -c "out of memory" "$R/server.log" | xargs -I{} echo "  'out of memory' lines in server.log so far: {}" | tee -a "$R/run.log"
}
dump_on() { echo "$1" > "$R/dump/ENABLE"; }
dump_off() { rm -f "$R/dump/ENABLE"; }

if [ "$PHASES" = dump ]; then
  dump_on kv; client kv_dump --mode kv --frames 241 --concurrency 1 --warmup 0 --rounds 1 --save-dir "$R/mp4/kv_dump"; dump_off
  dump_on chained; client ch_dump --mode chained --frames 241 --concurrency 1 --warmup 0 --rounds 1 --save-dir "$R/mp4/ch_dump"; dump_off
  log "variant $V dumps done"; exit 0
fi
if [ "$PHASES" = c8 ]; then
  client kv_warm --mode kv --frames 241 --concurrency 1 --warmup 1 --rounds 0
  client kv_c8 --mode kv --frames 241 --concurrency 8 --warmup 0 --rounds 1 --save-dir "$R/mp4/kv_c8"
  log "variant $V c8 done"; exit 0
fi
client kv_c1 --mode kv --frames 241 --concurrency 1 --warmup 1 --rounds 3 --save-dir "$R/mp4/kv_c1"
if [ "$PHASES" = c1 ]; then
  client ch_c1 --mode chained --frames 241 --concurrency 1 --warmup 1 --rounds 2 --save-dir "$R/mp4/ch_c1"
  log "variant $V c1 done"; exit 0
fi
if grep -q "ERROR" <(sed -n '/'"$V"'\/kv_c1/,/DONE/p' "$R/client.log"); then log "kv_c1 had errors; aborting $V"; exit 1; fi
dump_on kv; client kv_dump --mode kv --frames 241 --concurrency 1 --warmup 0 --rounds 1 --save-dir "$R/mp4/kv_dump"; dump_off
client kv_c4 --mode kv --frames 241 --concurrency 4 --warmup 1 --rounds 2 --save-dir "$R/mp4/kv_c4"
client ch_c1 --mode chained --frames 241 --concurrency 1 --warmup 1 --rounds 2 --save-dir "$R/mp4/ch_c1"
dump_on chained; client ch_dump --mode chained --frames 241 --concurrency 1 --warmup 0 --rounds 1 --save-dir "$R/mp4/ch_dump"; dump_off
log "variant $V done"
