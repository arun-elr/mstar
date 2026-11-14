#!/usr/bin/env bash
# Called by the disposable M* server wrapper while the Rust server is ready.
set -euo pipefail
: "${DIAG_REF:?}" "${DIAG_OUT:?}" "${DIAG_RUNNER:?}" "${DIAG_CONFIG:?}" "${DIAG_SCENE_MODE:?}" "${SERVER_PORT:?}"
: "${SERVER_CONFIG:?}"
mode=$DIAG_SCENE_MODE
trace_scene=0
[[ "$mode" == original ]] || trace_scene=$mode
cp "$SERVER_CONFIG" "$DIAG_OUT/mstar-served-$mode.yaml"
sha256sum "$DIAG_OUT/mstar-served-$mode.yaml" > "$DIAG_OUT/mstar-served-$mode.sha256"
export PYTHONPATH="$DIAG_REF:${PYTHONPATH:-}"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MSTAR_FLEX_BACKEND=TRITON
"$MSTAR_PYTHON" - <<'PY' | tee "$DIAG_OUT/mstar-flex-backend-$mode.log"
from mstar.engine.resources.attn.flex import _FLEX_BACKEND
print(f"M* effective flex backend: {_FLEX_BACKEND}", flush=True)
if _FLEX_BACKEND != 'TRITON':
    raise SystemExit('M* flex backend is not Triton')
PY
cd "$DIAG_REF"
"$DIAG_REF/.venv/bin/python" "$DIAG_RUNNER/trace_lift_verified.py" \
  --config "$DIAG_CONFIG" --mode mstar \
  --mstar-client "$DIAG_REF/praxis/serve/sim_client.py" \
  --mstar-url "ws://127.0.0.1:${SERVER_PORT}/api/v1/session" \
  --scene-seed "$trace_scene" --sampler-seed 0 \
  --out "$DIAG_OUT/mstar-lift-$mode-scene$trace_scene" \
  2>&1 | tee "$DIAG_OUT/mstar-lift-$mode-scene$trace_scene.log"
"$DIAG_REF/.venv/bin/python" "$DIAG_RUNNER/run_sim_eval_verified.py" \
  --suite droid_guard --config "$DIAG_CONFIG" --sampler-seeds 0 --no-video \
  --mstar-url "ws://127.0.0.1:${SERVER_PORT}/api/v1/session" \
  --mstar-client praxis.serve.sim_client \
  --out "$DIAG_OUT/mstar-$mode" \
  2>&1 | tee "$DIAG_OUT/mstar-eval-$mode.log"
