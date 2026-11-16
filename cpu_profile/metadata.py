import argparse
import hashlib
import json
import pathlib
import subprocess
import time


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--source-arm", choices=["main", "pr"], required=True)
    p.add_argument("--server-name", default=None)
    a = p.parse_args()
    base = pathlib.Path(f"/dev/shm/mstar-pr582-{a.source_arm}-src")
    build = pathlib.Path(f"/dev/shm/mstar-pr582-build/{a.source_arm}")
    cmd = pathlib.Path(f"/dev/shm/pr582-cpu-serverstate/pr582-{a.server_name or a.source_arm}.cmd")
    wheel = next(build.glob("*.whl"))
    report = {
        "arm": a.arm,
        "source_head": (base / "MEASURED_HEAD").read_text().strip(),
        "source_digests": {name: sha(base / name) for name in (
            "mstar/api_server/entrypoint.py",
            "mstar/conductor/conductor.py",
            "mstar/graph/runtime/rust.py",
            "rust/src/graph/frames.rs",
        )},
        "wheel_sha256": sha(wheel),
        "config_sha256": sha("/dev/shm/pr582-qwen.yaml"),
        "prompts_sha256": sha("/dev/shm/mstar-pr582-prompts.txt"),
        "benchmark_script_sha256": sha("/dev/shm/mstar-pr582-bench.py"),
        "checkpoint_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "checkpoint_safetensors_sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
        "launch_cmd": cmd.read_text().strip(),
        "frontend": "rust" if a.server_name == "rust" else "python",
        "launch_env": {"PATH": "/home/bench/praxis/mstar-venv/bin:$PATH",
                       "PYTHONPATH": f"/dev/shm/mstar-pr582-{a.source_arm}-pylib:{base}",
                       "MSTAR_COMPILE_LEVEL": "0", "MSTAR_RUST_GRAPH": "1",
                       "MSTAR_RUST_ZMQ": "1", "MSTAR_PHASE_TIMING": "100",
                       "SERVER_STATE_DIR": "/dev/shm/pr582-cpu-serverstate"},
        "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if a.server_name == "rust":
        report["server_binary_sha256"] = sha(build / "mstar-server")
    out = pathlib.Path(f"/dev/shm/pr582-cpu-results/{a.arm}-metadata.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out, report["source_head"])


if __name__ == "__main__":
    main()
