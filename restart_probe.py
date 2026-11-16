"""Process-level old-worker/restarted-conductor control-mesh probe."""

import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time


def actor(pipe, role, token, prefix):
    os.environ["MSTAR_DEPLOYMENT_TOKEN"] = token
    from mstar.communication.rust_communicator import RustZMQCommunicator

    peer = "conductor" if role == "worker_0" else "worker_0"
    comm = RustZMQCommunicator(my_id=role, push_ids=[peer], ipc_socket_path_prefix=prefix)
    pipe.send(("ready", comm._endpoint(role), comm._endpoint(peer)))
    while True:
        op, arg = pipe.recv()
        if op == "send":
            comm.send(peer, {"source": arg})
            pipe.send(("sent", None))
        elif op == "receive":
            end = time.monotonic() + arg
            got = []
            while time.monotonic() < end:
                got.extend(comm.get_all_new_messages())
                time.sleep(0.01)
            pipe.send(("received", got))
        else:
            break


def start(ctx, role, token, prefix):
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=actor, args=(child, role, token, prefix))
    proc.daemon = True
    proc.start()
    try:
        assert parent.poll(10), f"{role} did not start"
        ready = parent.recv()
        assert ready[0] == "ready", ready
        return proc, parent, ready
    except BaseException:
        proc.terminate()
        proc.join(1)
        if proc.is_alive():
            proc.kill()
            proc.join(1)
        raise


def stop(proc):
    if proc.is_alive():
        proc.terminate()
        proc.join(1)
    if proc.is_alive():
        proc.kill()
        proc.join(1)


def main():
    import mstar
    import mstar_rust

    repo = os.path.realpath(os.getcwd())
    source = os.path.realpath(mstar.__file__)
    if not source.startswith(repo + os.sep):
        raise RuntimeError(f"M* loaded outside checkout: {source}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    ctx = mp.get_context("spawn")
    started = []
    with tempfile.TemporaryDirectory(prefix="mstar-774-") as prefix:
        try:
            first, first_pipe, first_ready = start(ctx, "conductor", "first-launch", prefix)
            started.append(first)
            worker, worker_pipe, worker_ready = start(ctx, "worker_0", "first-launch", prefix)
            started.append(worker)
            worker_pipe.send(("send", "before-restart"))
            assert worker_pipe.poll(2) and worker_pipe.recv()[0] == "sent"
            first_pipe.send(("receive", 1.0))
            assert first_pipe.poll(3)
            before = first_pipe.recv()
            assert before == ("received", [{"source": "before-restart"}]), before
            first.terminate()
            first.join(3)
            assert first.exitcode is not None
            assert worker.is_alive()
            second, second_pipe, second_ready = start(ctx, "conductor", "second-launch", prefix)
            started.append(second)
            current_worker, current_worker_pipe, current_worker_ready = start(
                ctx, "worker_0", "second-launch", prefix
            )
            started.append(current_worker)
            # The send interval lets a surviving PUSH socket reconnect if the
            # new conductor reused its path.
            for i in range(20):
                worker_pipe.send(("send", f"old-{i}"))
                assert worker_pipe.poll(2)
                assert worker_pipe.recv()[0] == "sent"
                time.sleep(0.025)
            current_worker_pipe.send(("send", "current"))
            assert current_worker_pipe.poll(2)
            assert current_worker_pipe.recv()[0] == "sent"
            second_pipe.send(("receive", 2.0))
            assert second_pipe.poll(4)
            got = second_pipe.recv()
            assert got[0] == "received"
            print(json.dumps({
                "source_head": head,
                "source_dirty": dirty,
                "mstar_source": source,
                "rust_extension": mstar_rust.__file__,
                "rust_version": getattr(mstar_rust, "__version__", None),
                "first_endpoint": first_ready[1],
                "old_worker_peer": worker_ready[2],
                "second_endpoint": second_ready[1],
                "current_worker_peer": current_worker_ready[2],
                "old_worker_reached_first_before_restart": before[1],
                "old_messages_at_second": sum(
                    message["source"].startswith("old-") for message in got[1]
                ),
                "current_messages_at_second": sum(
                    message["source"] == "current" for message in got[1]
                ),
                "sample": got[1][:2],
            }), flush=True)
        finally:
            for proc in reversed(started):
                stop(proc)


if __name__ == "__main__":
    main()
