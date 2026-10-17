"""One model's base and head arms on one GPU, back to back, then the base-vs-head facts.

python -m benchmark.eval.e2e.arm JOB.json      (started by ``compare`` on the GPU's host, local or over SSH)

Each arm's server is ``mstar serve`` launched by ``ManagedServer`` from the arm's own checkout (``PYTHONPATH`` and
working directory, plus that tree's ``rust/`` build), with the eval client of the tree running ``compare``. Server
sessions alternate the arm order (ABBA with two sessions). Layout under ``job["out"]``::

  status.json                     phase, arm, case (polled by compare)
  <arm>/s<k>/startup.json         healthy, startup_s, peak VRAM, which checkout the server imported
  <arm>/s<k>/server.log
  <arm>/s<k>/<case>/              mstar-eval/1 results per concurrency, saved outputs, client log
  compare.json                    checks.compare_model
  done.json                       written last: done | skipped | error | interrupted
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from benchmark.eval.e2e import checks
from benchmark.eval.server import ManagedServer, ServerError, is_healthy

CASE_TIMEOUT_S = 3600
IDLE_MB = 2000

job: dict = {}
out = Path(".")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def status(**kw) -> None:
    (out / "status.json").write_text(json.dumps({"t": now(), **kw}))


def done(state: str, **kw) -> None:
    (out / "done.json").write_text(json.dumps({"status": state, "t": now(), **kw}))


def gpu_mem_mb() -> int:
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if job.get("gpu") is not None:
        cmd += ["-i", str(job["gpu"])]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return max(int(x) for x in r.stdout.split())
    except (OSError, ValueError, subprocess.SubprocessError):
        return -1


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop_ev = threading.Event()

    def run(self):
        while not self.stop_ev.wait(2.0):
            self.peak = max(self.peak, gpu_mem_mb())


def wait_gpu_idle(limit_s: float = 180) -> int:
    t0, mem = time.time(), gpu_mem_mb()
    while mem > IDLE_MB and time.time() - t0 < limit_s:
        time.sleep(5)
        mem = gpu_mem_mb()
    return mem


def base_env() -> dict[str, str]:
    env = {"PYTHONUNBUFFERED": "1"}
    if job.get("gpu") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    return env


def server_env(arm: dict) -> dict[str, str]:
    path = [arm["src"]] + ([arm["rust_site"]] if arm.get("rust_site") else [])
    return {**base_env(), **job["model"].get("env", {}), "PYTHONPATH": ":".join(path)}


def client_env() -> dict[str, str]:
    return {**os.environ, **base_env(), "PYTHONPATH": job["evalfw"]}


@contextmanager
def host_lock(name: str):
    """Serializes builds that GPU slots on one host share (a ``rust/`` tree's extension or server binary)."""
    path = Path(job["host_dir"]) / "locks" / f"{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def build_rust(arm: dict) -> str | None:
    """``mstar_rust`` for the arm's ``rust/`` tree into a shared site dir; returns the problem, or None."""
    tree = arm.get("rust_tree")
    if not tree:
        return "no rust/ tree"
    site = Path(job["host_dir"]) / "rust" / tree / "site"
    with host_lock(f"rust-{tree}"):
        if not (site / ".complete").exists():
            shutil.rmtree(site, ignore_errors=True)
            r = subprocess.run([job["python"], "-m", "pip", "install", "-q", "--no-deps", "--target", str(site),
                                f"{arm['src']}/rust"], capture_output=True, text=True, check=False)
            if r.returncode != 0:
                return f"mstar_rust build failed: {(r.stdout + r.stderr).strip()[-300:]}"
            (site / ".complete").touch()
    arm["rust_site"] = str(site)
    return None


def build_rust_server(arm: dict) -> str | None:
    tree = arm.get("rust_tree")
    binary = Path(job["host_dir"]) / "rust" / str(tree) / "mstar-server"
    with host_lock(f"rust-server-{tree}"):
        if not binary.exists():
            if not shutil.which("cargo"):
                return "the Rust frontend needs cargo on the host"
            target = binary.parent / "target-server"
            r = subprocess.run(["cargo", "build", "-q", "--release"], cwd=f"{arm['src']}/rust/server",
                               env={**os.environ, "PYO3_PYTHON": job["python"], "CARGO_TARGET_DIR": str(target)},
                               capture_output=True, text=True, check=False)
            if r.returncode != 0:
                return f"Rust frontend build failed: {r.stderr.strip()[-300:]}"
            shutil.copy(target / "release" / "mstar-server", binary)
    arm["rust_server"] = str(binary)
    return None


def requirements_problem(arm: dict) -> str:
    mods = job["model"].get("requires", [])
    if not mods:
        return ""
    code = ("import importlib, sys\nbad = []\nfor m in sys.argv[1:]:\n    try:\n        importlib.import_module(m)\n"
            "    except Exception as e:\n        bad.append(f'{m}: {type(e).__name__}')\nprint('; '.join(bad))")
    r = subprocess.run([job["python"], "-c", code, *mods], env={**os.environ, **server_env(arm)}, cwd=arm["src"],
                       capture_output=True, text=True, timeout=600, check=False)
    return r.stdout.strip() or (r.stderr.strip()[-300:] if r.returncode else "")


def case_cmds(case: dict, url: str, label: str, d: Path) -> list[list[str]]:
    py, data = job["python"], Path(job["data"])
    if case["driver"] in ("rp", "invariants"):
        return [[py, "-m", "benchmark.eval.e2e.request_path", "--url", url, "--suite", case["kind"],
                 "--inputs", str(data / "rp"), "--out", str(d / "rp.json")]]
    suite = case["kind"] if case["driver"] == "suite" else "e2e"
    # e2e: one point per call, so warmup counts waves at that point's concurrency. Speech suites: one call for
    # every point, so their judges (Whisper, WavLM) load once, with the largest point's waves as warmup.
    points = [[c] for c in case["conc"]] if suite == "e2e" else [case["conc"]]
    cmds = []
    for conc in points:
        cmd = [py, "-m", "benchmark.eval", "run", suite, "--url", url, "--concurrency", ",".join(map(str, conc)),
               "--repeats", str(case["repeats"]), "--warmup", str(case["warmup"] * max(conc)), "--label", label,
               "--out", str(d), "--save-audio", "last"]
        if case.get("data"):
            cmd += ["--data", str(data / job["datasets"][case["data"]])]
        if case["driver"] == "client":
            cmd += ["--kind", case["kind"], "--media", str(data / "media")]
        if case.get("n"):
            cmd += ["--limit", str(case["n"])]
        cmds.append(cmd + case["args"])
    return cmds


def serve_args(arm: dict) -> list[str]:
    return [a.replace("{rust_server_bin}", arm.get("rust_server") or "") for a in job["model"].get("serve_args", [])]


def run_arm(arm_name: str, session: int) -> dict:
    model, arm = job["model"], job["arms"][arm_name]
    d = out / arm_name / f"s{session}"
    shutil.rmtree(d, ignore_errors=True)   # a rerun must not compare against a previous attempt's outputs
    d.mkdir(parents=True)
    status(phase="startup", arm=arm_name, session=session)
    info: dict = {"gpu_mem_before_mb": wait_gpu_idle()}
    sampler = VramSampler()
    sampler.start()
    config = str(Path(arm["src"]) / model["config"]) if model.get("config") else None
    server = ManagedServer(model["serve"], config, extra_args=serve_args(arm), env=server_env(arm),
                           log_path=d / "server.log", startup_timeout_s=model["startup_timeout_s"],
                           python=job["python"], cwd=arm["src"])
    cases: dict = {}
    try:
        with server:
            info.update(healthy=True, startup_s=round(server.startup_s, 1))
            prov = subprocess.run([job["python"], "-c", "import mstar; print(mstar.__file__)"],
                                  env={**os.environ, **server_env(arm)}, cwd=arm["src"], capture_output=True,
                                  text=True, timeout=120, check=False)
            imported = os.path.realpath(prov.stdout.strip() or "/nonexistent")
            info["mstar_from_arm_src"] = imported.startswith(os.path.realpath(arm["src"]) + os.sep)
            log(f"{arm_name} s{session}: healthy after {info['startup_s']} s")
            for case in model["cases"] if info["mstar_from_arm_src"] else []:
                cd = d / case["id"]
                cd.mkdir(parents=True, exist_ok=True)
                status(phase="case", arm=arm_name, session=session, case=case["id"])
                t0, rc = time.monotonic(), 0
                with open(cd / "client.log", "w") as f:
                    for cmd in case_cmds(case, server.url, f"{model['name']}-{case['id']}-{arm_name}", cd):
                        try:
                            r = subprocess.run(cmd, env=client_env(), cwd=job["evalfw"], stdout=f,
                                               stderr=subprocess.STDOUT, timeout=CASE_TIMEOUT_S, check=False)
                            rc = r.returncode or rc
                        except subprocess.TimeoutExpired:
                            rc = "timeout"
                alive = is_healthy(server.url, timeout=10)
                cases[case["id"]] = {"rc": rc, "elapsed_s": round(time.monotonic() - t0, 1),
                                     "server_alive_after": alive}
                log(f"{arm_name} s{session}: {case['id']} rc={rc} {cases[case['id']]['elapsed_s']} s alive={alive}")
                if not alive:
                    log(f"{arm_name} s{session}: server died during {case['id']}; skipping its remaining cases")
                    break
    except ServerError as e:
        info.update(healthy=False, error=str(e)[-500:])
        log(f"{arm_name} s{session}: did not start: {e}")
    finally:
        sampler.stop_ev.set()
    info["peak_vram_gb"] = round(sampler.peak / 1024, 1) if sampler.peak > 0 else None
    info["cases"] = cases
    (d / "startup.json").write_text(json.dumps(info, indent=1))
    return info


def _on_term(signum, frame) -> None:  # noqa: ARG001
    raise SystemExit(128 + signum)   # unwinds through `with server:`, so no GPU workers are left behind


def run() -> None:
    model = job["model"]
    arms = job["arms"]
    log(f"{model['name']}: order {job['order']}, {job['sessions']} session(s)")
    problems = {a: build_rust(arm) for a, arm in arms.items()}
    if any(problems.values()) and arms["base"].get("rust_tree") != arms["head"].get("rust_tree"):
        bad = [f"{a}: {p}" for a, p in problems.items() if p]
        return done("error", reason="base and head have different rust/ trees and one did not build: " + "; ".join(bad))
    if any("{rust_server_bin}" in a for a in model.get("serve_args", [])):
        for arm in arms.values():
            problem = build_rust_server(arm)
            if problem:
                return done("skipped", reason=problem)
    missing = {a: requirements_problem(arm) for a, arm in arms.items()}
    if all(missing.values()):
        return done("skipped", reason=f"environment lacks: {missing['base']}")
    for s in range(job["sessions"]):
        for arm_name in job["order"] if s % 2 == 0 else list(reversed(job["order"])):
            info = run_arm(arm_name, s)
            if info.get("healthy") and not info.get("mstar_from_arm_src"):
                return done("error", reason=f"the {arm_name} server did not import mstar from its own checkout")
    status(phase="compare")
    checks.compare_model(out, model["cases"], job["sessions"])
    done("done", rust={a: p for a, p in problems.items() if p})
    status(phase="done")


def main(argv: list[str] | None = None) -> None:
    global job, out
    job = json.loads(Path((argv or sys.argv[1:])[0]).read_text())
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "done.json").unlink(missing_ok=True)
    (out / "pid").write_text(str(os.getpid()))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_term)
    t0 = time.time()
    try:
        run()
    except SystemExit as e:
        done("interrupted", reason=f"signal {int(e.code or 0) - 128}")
        raise
    except Exception as e:  # noqa: BLE001 - always leave a done.json behind
        done("error", reason=f"{type(e).__name__}: {str(e)[:400]}")
        raise
    log(f"{model_name()} finished in {time.time() - t0:.0f} s")


def model_name() -> str:
    return job.get("model", {}).get("name", "?")


if __name__ == "__main__":
    main()
