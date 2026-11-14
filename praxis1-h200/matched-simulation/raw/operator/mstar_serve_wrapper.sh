#!/usr/bin/env bash
# Run the committed DROID session config with only a disposable serving gate.
# Caller supplies the paths below and the same lock used for all H200 tests.
set -euo pipefail

: "${MSTAR_PYTHON:?set MSTAR_PYTHON to the source-matched M* venv python}"
: "${SOURCE_ROOT:?set SOURCE_ROOT to the read-only M* source archive}"
: "${CLIENT_SCRIPT:?set CLIENT_SCRIPT to praxis1_openpi_smoke.py on the H200}"
: "${GPU_LOCK_FILE:?set GPU_LOCK_FILE to the shared H200 lock path}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT to the pinned Praxis reference source}"
: "${MSTAR_REVISION:?set MSTAR_REVISION to the M* source archive revision}"

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/home/bench/praxis/checkpoints/praxis-1-droid}
QWEN_ROOT=${QWEN_ROOT:-/home/bench/praxis/weights/Qwen3.5-2B}
VAE_ROOT=${VAE_ROOT:-/home/bench/praxis/weights/Wan2.2-TI2V-5B-Diffusers}
LOG_ROOT=${LOG_ROOT:-/home/bench/praxis/logs}
PORT=${PORT:-8123}
MAX_SERVER_SECONDS=${MAX_SERVER_SECONDS:-1800}
FRONTEND=${FRONTEND:-python}
if [[ "$FRONTEND" == rust ]]; then
  RUST_SERVER_BIN=${RUST_SERVER_BIN:-/home/bench/praxis/mstar-server-e5}
  [[ -x "$RUST_SERVER_BIN" ]] || { echo "missing source-matched Rust binary: $RUST_SERVER_BIN" >&2; exit 1; }
  frontend_args=(--rust-frontend-bin "$RUST_SERVER_BIN")
elif [[ "$FRONTEND" == python ]]; then
  frontend_args=(--python-frontend)
else
  echo "FRONTEND must be python or rust" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
exec 9>"$GPU_LOCK_FILE"
flock -n 9 || { echo "H200 lock is held: $GPU_LOCK_FILE" >&2; exit 75; }

probe_dir=$(mktemp -d /tmp/mstar-e5-serve-probe.XXXXXX)
server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf -- "$probe_dir"
}
trap cleanup EXIT INT TERM

cp -a "$SOURCE_ROOT" "$probe_dir/src"
"$MSTAR_PYTHON" - "$probe_dir/src" "$CHECKPOINT_ROOT" "$QWEN_ROOT" "$VAE_ROOT" "$probe_dir" "$MSTAR_REVISION" <<'PYDIAG'
import pathlib
import sys
import yaml

source, checkpoint, qwen, vae, work = map(pathlib.Path, sys.argv[1:6])
revision = sys.argv[6]
for path in (checkpoint / "config.json", qwen / "config.json", vae / "vae" / "config.json"):
    if not path.is_file():
        raise SystemExit(f"missing required local file: {path}")
model_file = source / "mstar" / "model" / "praxis1" / "praxis1_model.py"
body = model_file.read_text()
guard = "_SERVABLE = False"
if body.count(guard) != 1:
    raise SystemExit("unexpected Praxis serving guard in diagnostic copy")
model_file.write_text(body.replace(guard, "_SERVABLE = True", 1))
config = yaml.safe_load((source / "configs" / "praxis1_droid.yaml").read_text())
assert config["model_kwargs"]["max_context_chunks"] == 1
assert config["openpi"]["metadata"]["max_context_chunks"] == 0
assert config["positronic"]["meta"]["max_context_chunks"] == 1
assert config["positronic"]["control_mode"] == "droid_impedance"
config["model_kwargs"].update(
    model_path_hf=str(checkpoint), text_encoder_hf=str(qwen), vae_weights_hf=str(vae),
)
config["openpi"].setdefault("metadata", {})["code_version"] = revision
config["positronic"]["meta"]["code_version"] = revision
(work / "praxis1_droid.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PYDIAG
server_log="$LOG_ROOT/e5-droid-openpi-${FRONTEND}-server-${PORT}.log"
client_log="$LOG_ROOT/e5-droid-openpi-${FRONTEND}-client-${PORT}.log"
actions_file="$LOG_ROOT/e5-droid-openpi-${FRONTEND}-actions-${PORT}.npy"
echo "server log: $server_log"
echo "client log: $client_log"
export PYTHONPATH="$probe_dir/src:$REFERENCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "$FRONTEND" == rust ]]; then
  "$MSTAR_PYTHON" - "$RUST_SERVER_BIN" "$probe_dir/src/rust" <<'PYCHECK'
import pathlib
import sys
from mstar.api_server.rust_frontend import _stale_server
problem = _stale_server(sys.argv[1], pathlib.Path(sys.argv[2]))
if problem is not None:
    raise SystemExit(f"Rust server source mismatch: {problem}")
PYCHECK
fi
setsid timeout --signal=TERM --kill-after=15s "${MAX_SERVER_SECONDS}s" \
  "$MSTAR_PYTHON" -m mstar.cli.main serve praxis1_droid \
  --config "$probe_dir/praxis1_droid.yaml" --gpus 0 --host 127.0.0.1 \
  --port "$PORT" -O0 "${frontend_args[@]}" --log-stats \
  >"$server_log" 2>&1 &
server_pid=$!

ready=0
for ((attempt=0; attempt<180; attempt++)); do
  if curl --max-time 2 -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited before readiness; inspect $server_log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  echo "server did not become ready within 15 minutes; inspect $server_log" >&2
  exit 1
fi

if [[ "${SKIP_OPENPI_SMOKE:-0}" != 1 ]]; then
  timeout --signal=TERM --kill-after=15s 650s \
    "$MSTAR_PYTHON" "$CLIENT_SCRIPT" --host 127.0.0.1 --port "$PORT" --timeout 600 \
    --output "$actions_file" \
    2>&1 | tee "$client_log"
fi

if [[ -n "${POST_READY_SCRIPT:-}" ]]; then
  [[ -x "$POST_READY_SCRIPT" ]] || { echo "POST_READY_SCRIPT is not executable" >&2; exit 1; }
  export SERVER_URL="http://127.0.0.1:${PORT}" SERVER_PORT="$PORT"
  export SERVER_CONFIG="$probe_dir/praxis1_droid.yaml" SERVER_ACTIONS="$actions_file"
  timeout --signal=TERM --kill-after=15s "${POST_READY_SECONDS:-900}s" "$POST_READY_SCRIPT"
fi
