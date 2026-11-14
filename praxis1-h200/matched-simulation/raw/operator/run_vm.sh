#!/usr/bin/env bash
# Disposable matched simulation run; only under the host's bounded timer.
set -euo pipefail
ROOT=/home/bench/praxis/mstar-lift-matched-operator
REF=/home/bench/praxis/src-mstar-sim
OUT=/home/bench/praxis/logs/lift-matched-rerun
CKPT=/home/bench/praxis/checkpoints/praxis-1-droid
QWEN=/home/bench/praxis/weights/Qwen3.5-2B
VAE_REF=/home/bench/praxis/weights/Wan2.2_VAE.pth
VAE_MSTAR=/home/bench/praxis/weights/Wan2.2-TI2V-5B-Diffusers
MSTAR_PY=/home/bench/praxis/mstar-venv/bin/python
RUST_BIN=/home/bench/praxis/mstar-server
[[ ! -e "$OUT" ]] || { echo "output directory already exists, refusing resume: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
sha256sum -c frozen.sha256
[[ -f "$REF/.venv/bin/python" && -x "$MSTAR_PY" && -x "$RUST_BIN" ]] || { echo 'missing matched runtime' >&2; exit 1; }
[[ $(git -C "$REF" rev-parse HEAD) == 68f4d7a254b4bf41951d2e97f44b07f6cac2bf69 ]] || { echo 'reference commit mismatch' >&2; exit 1; }
git -C "$REF" --no-pager diff --no-ext-diff HEAD > "$OUT/reference-working-tree.actual.diff"
cmp "$ROOT/reference-working-tree.diff" "$OUT/reference-working-tree.actual.diff"
git -C "$REF" status --short --untracked-files=normal > "$OUT/reference-working-tree.actual.status"
python3 - "$ROOT/reference-working-tree.status" "$OUT/reference-working-tree.actual.status" <<'PY'
import pathlib, sys
expected, actual = (pathlib.Path(p).read_text().splitlines() for p in sys.argv[1:])
normalise = lambda rows: sorted(row.rstrip('/') for row in rows)
if normalise(expected) != normalise(actual):
    raise SystemExit(f'reference status differs from pinned status: {actual!r}')
PY
cmp "$ROOT/sim_client.py" "$REF/praxis/serve/sim_client.py"
printf '%s\n' 'pre-load source gates passed' | tee "$OUT/preflight.log"

# Hash every model artifact before loading. Both arms use the same checkpoint
# and text path; the reference and M* VAE packages have different layouts.
find -L "$CKPT" "$QWEN" "$VAE_MSTAR" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/model-files.sha256"
sha256sum "$VAE_REF" >> "$OUT/model-files.sha256"
sha256sum "$REF/configs/sim_droid.yaml" "$ROOT/mstar-source.tar.gz" "$RUST_BIN" > "$OUT/source-files.sha256"

WORK=$(mktemp -d /tmp/lift-matched.XXXXXX)
trap 'rm -rf -- "$WORK"' EXIT
mkdir -p "$WORK/source" "$WORK/runner"
tar -xzf "$ROOT/mstar-source.tar.gz" -C "$WORK/source"
sha256sum "$WORK/source/configs/praxis1_droid.yaml" "$CKPT/config.json" \
  >> "$OUT/source-files.sha256"
python3 "$ROOT/prepare.py" "$REF" "$ROOT/trace_lift_cube_corrected.py" "$WORK/runner"
sha256sum "$WORK/runner/"*.py > "$OUT/disposable-runner.sha256"
"$REF/.venv/bin/python" - "$REF/configs/sim_droid.yaml" "$WORK/sim_droid_absolute.yaml" "$CKPT" "$VAE_REF" "$QWEN" <<'PY'
import sys, yaml
src, dst, checkpoint, vae, qwen = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg['model']['checkpoint'] = checkpoint
cfg['model']['vae_weights'] = vae
cfg['model']['text_encoder'] = qwen
with open(dst, 'w') as f: yaml.safe_dump(cfg, f, sort_keys=False)
PY
cp "$WORK/sim_droid_absolute.yaml" "$OUT/reference-served.yaml"
sha256sum "$OUT/reference-served.yaml" >> "$OUT/source-files.sha256"
export PYTHONPATH="$REF:$WORK/source"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PRAXIS_ENCODER_ATTENTION_BACKEND=triton PRAXIS_DECODER_ATTENTION_BACKEND=triton
export MSTAR_FLEX_BACKEND=TRITON

run_ref_trace() {
  local mode=$1
  local trace_scene=0
  [[ "$mode" == original ]] || trace_scene=$mode
  export DIAG_SCENE_MODE=$mode
  cd "$REF"
  flock -w 10 /home/bench/gpu.lock \
    "$REF/.venv/bin/python" "$WORK/runner/trace_lift_verified.py" \
      --config "$WORK/sim_droid_absolute.yaml" --mode reference \
      --checkpoint "$CKPT" --scene-seed "$trace_scene" --sampler-seed 0 \
      --out "$OUT/reference-lift-$mode-scene$trace_scene" \
      2>&1 | tee "$OUT/reference-lift-$mode-scene$trace_scene.log"
}

run_ref_suite() {
  local mode=$1
  export DIAG_SCENE_MODE=$mode
  cd "$REF"
  flock -w 10 /home/bench/gpu.lock \
    "$REF/.venv/bin/python" "$WORK/runner/run_sim_eval_verified.py" \
      --suite droid_guard --config "$WORK/sim_droid_absolute.yaml" \
      --sampler-seeds 0 --no-video --out "$OUT/reference-$mode" \
      2>&1 | tee "$OUT/reference-$mode.log"
}

run_mstar() {
  local mode=$1
  export DIAG_SCENE_MODE=$mode
  export DIAG_REF="$REF" DIAG_OUT="$OUT" DIAG_RUNNER="$WORK/runner"
  export DIAG_CONFIG="$WORK/sim_droid_absolute.yaml"
  export MSTAR_PYTHON="$MSTAR_PY" SOURCE_ROOT="$WORK/source"
  export CLIENT_SCRIPT="$WORK/source/scripts/praxis1_openpi_smoke.py"
  export GPU_LOCK_FILE=/home/bench/gpu.lock REFERENCE_ROOT="$REF"
  export MSTAR_REVISION=5902054c0f PORT=8123 FRONTEND=rust
  export RUST_SERVER_BIN="$RUST_BIN" LOG_ROOT="$OUT"
  export POST_READY_SCRIPT="$ROOT/mstar_post_ready.sh"
  export POST_READY_SECONDS=900 MAX_SERVER_SECONDS=1500 SKIP_OPENPI_SMOKE=1
  "$ROOT/mstar_serve_wrapper.sh" 2>&1 | tee "$OUT/mstar-$mode.log"
}

run_ref_trace original
run_mstar original
run_ref_suite original
"$REF/.venv/bin/python" "$ROOT/check_results.py" "$OUT" original
if [[ ${RUN_SCENE0:-0} == 1 ]]; then
  remaining=$(( DIAG_DEADLINE_EPOCH - $(date -u +%s) ))
  if (( remaining >= 900 )); then
    run_ref_trace 0
    run_mstar 0
    run_ref_suite 0
    "$REF/.venv/bin/python" "$ROOT/check_results.py" "$OUT" 0
  else
    echo "scene0 sweep skipped; only $remaining seconds until hard stop" | tee "$OUT/scene0-skip.txt"
  fi
fi
if [[ ${RUN_SCENE1:-0} == 1 ]]; then
  remaining=$(( DIAG_DEADLINE_EPOCH - $(date -u +%s) ))
  if (( remaining >= 900 )); then
    run_ref_trace 1
    run_mstar 1
    run_ref_suite 1
    "$REF/.venv/bin/python" "$ROOT/check_results.py" "$OUT" 1
  else
    echo "scene1 skipped; only $remaining seconds until hard stop" | tee "$OUT/scene1-skip.txt"
  fi
fi
echo 'matched rerun complete; host must stop VM and verify STOPPED'
