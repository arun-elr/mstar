#!/bin/bash
# End-to-end streaming rollout with each Wan VAE decode configuration: serve
# configs/cosmos3_edge.yaml with the variant's model_kwargs overlaid, then time
# the rollout with benchmark/cosmos3/bench_stream_video.py (time to first frame,
# chunk gap, frames/s; the decoder shares the GPU with the denoise loop).
#
#   e2e_stream.sh serve <variant> <gpu> <port>   # foreground; Ctrl-C to stop
#   e2e_stream.sh bench <port> [cond_image.jpg]  # kv and chained, 241 frames at 832x480
#
# Variants: default (the checkout's defaults; with MSTAR at upstream main this is
# the baseline), redecode, warm, warm_cl, warm_cl_chunk, warm_cl_chunk_bench (see
# bench_decode_nodes.py). MSTAR = the checkout to serve (default: this repo).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
MSTAR="${MSTAR:-$(cd "$here/../.." && pwd)}"

overrides() {
  case "$1" in
    default) echo '{}' ;;
    redecode) echo '{"windowed_decode_warm_cache": false, "vae_decode_channels_last": false}' ;;
    warm) echo '{"windowed_decode_warm_cache": true, "vae_decode_channels_last": false}' ;;
    warm_cl) echo '{"windowed_decode_warm_cache": true, "vae_decode_channels_last": true}' ;;
    warm_cl_chunk) echo '{"windowed_decode_warm_cache": true, "vae_decode_channels_last": true,
      "windowed_decode_latents_per_call": 0}' ;;
    warm_cl_chunk_bench) echo '{"windowed_decode_warm_cache": true, "vae_decode_channels_last": true,
      "windowed_decode_latents_per_call": 0, "vae_decode_cudnn_benchmark": true}' ;;
    *) echo "unknown variant $1" >&2; exit 2 ;;
  esac
}

serve() {
  local variant="$1" gpu="$2" port="$3" cfg sock upload
  cfg=$(mktemp --suffix=.yaml); sock=$(mktemp -d); upload=$(mktemp -d)
  python - "$MSTAR/configs/cosmos3_edge.yaml" "$cfg" "$(overrides "$variant")" <<'EOF'
import json, sys, yaml
src, dst, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
cfg = yaml.safe_load(open(src))
cfg.setdefault("model_kwargs", {}).update(extra)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"serving {src} with model_kwargs overrides {extra}", file=sys.stderr)
EOF
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    COSMOS3_GEN_CAPTURE_RES=192x320,480x832,640x640 PYTHONPATH="$MSTAR" \
    python "$MSTAR/mstar/api_server/entrypoint.py" --config "$cfg" \
    --socket-path-prefix "$sock/" --upload-dir "$upload/" \
    --port "$port" --mooncake-port "$((port + 1000))" --tensor-comm-protocol SHM
}

bench() {
  local port="$1" img="${2:-}" extra=()
  [ -n "$img" ] && extra=(--image "$img")
  for mode in kv chained; do
    python "$MSTAR/benchmark/cosmos3/bench_stream_video.py" --port "$port" --mode "$mode" --size 832x480 \
      --frames 241 --window-frames 29 --steps 20 --gs 6.0 --rounds 3 "${extra[@]}"
  done
}

cmd="${1:?serve|bench}"; shift
"$cmd" "$@"
