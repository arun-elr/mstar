import argparse
import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi import FastAPI
from fastapi.testclient import TestClient

from test.modular.test_openai_router import _StubAPI


parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


def one_run():
    count = 0
    seen = threading.Condition()
    release = threading.Event()

    class Slow(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal count
            with seen:
                count += 1
                seen.notify_all()
            release.wait(timeout=5)
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *unused):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    upstream.daemon_threads = True
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    remote = {"model": "bagel", "messages": [{"role": "user", "content": [{
        "type": "image_url", "image_url": {"url": f"http://127.0.0.1:{upstream.server_address[1]}/slow.png"},
    }]}]}
    plain = {"model": "bagel", "messages": [{"role": "user", "content": "hello"}]}
    stub = _StubAPI("bagel")
    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    fake_ep.api_server = stub
    sys.modules["mstar.api_server.entrypoint"] = fake_ep
    import mstar.api_server
    mstar.api_server.entrypoint = fake_ep
    from mstar.api_server.openai.router import router
    app = FastAPI()
    app.include_router(router)
    start = threading.Barrier(49)
    try:
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=48) as pool:
            def submit():
                start.wait(timeout=10)
                return client.post("/v1/chat/completions", json=remote)

            futures = [pool.submit(submit) for _ in range(48)]
            start.wait(timeout=10)
            expected = 8 if args.arm == "pr" else 32
            with seen:
                if not seen.wait_for(lambda: count >= expected, timeout=10):
                    raise RuntimeError(f"only {count} upstream fetches started, wanted {expected}")
            time.sleep(0.1)
            timer = threading.Timer(2.0, release.set)
            timer.start()
            begin = time.perf_counter()
            plain_reply = client.post("/v1/chat/completions", json=plain)
            plain_latency_ms = (time.perf_counter() - begin) * 1000
            replies = [f.result(timeout=15) for f in futures]
            timer.join(timeout=5)
            if plain_reply.status_code != 200 or any(r.status_code != 400 for r in replies):
                raise RuntimeError("unexpected status")
            busy = sum("too many concurrent fetches" in r.json()["error"]["message"] for r in replies)
            return {"plain_chat_ms": round(plain_latency_ms, 1), "remote_upstream_hits": count,
                    "remote_busy_refusals": busy, "remote_400s": len(replies)}
    finally:
        release.set()
        upstream.shutdown()
        upstream.server_close()


print(json.dumps({"arm": args.arm, "runs": [one_run() for _ in range(args.runs)]}))
