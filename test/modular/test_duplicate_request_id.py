"""A client-supplied ``request_id`` that is still in flight is refused with a
409. The request that holds it keeps its state, is not aborted, and finishes
normally. Checked on ``submit_request`` and through ``/generate`` and
``/generate/ws``."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import APIServer
from mstar.api_server.request_types import ResultChunk

TEXT = dict(input_modalities=["text"], output_modalities=["text"])


def _server(tmp_path):
    s = APIServer.__new__(APIServer)
    s.request_lock = threading.Lock()
    s.fatal_error = None
    s.pending_requests = {}
    s.recently_completed = {}
    s.timeout_seconds = 5
    s.log_stats = False
    s.enable_nvtx = False
    s.upload_dir = tmp_path
    s.submitted = []
    s.aborted = []
    s.preprocess_worker = SimpleNamespace(
        new_request=lambda inp: s.submitted.append((inp.request_id, inp.text)),
        abort_request=s.aborted.append,
    )
    return s


def _finish(server, rid, data=b"answer"):
    with server.request_lock:
        req = server.pending_requests[rid]
        req.chunks.append(ResultChunk(request_id=rid, modality="text", data=data, metadata={}))
        req.event.set()


def _finish_when_submitted(server, rid, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with server.request_lock:
            if rid in server.pending_requests:
                break
        time.sleep(0.005)
    _finish(server, rid)


def test_in_flight_id_is_refused_and_the_original_finishes(tmp_path):
    s = _server(tmp_path)
    s.submit_request(text="first", request_id="dup", streaming=False, **TEXT)
    original = s.pending_requests["dup"]

    with pytest.raises(HTTPException) as e:
        s.submit_request(text="second", request_id="dup", streaming=False, **TEXT)

    assert e.value.status_code == 409
    assert s.pending_requests["dup"] is original
    assert s.submitted == [("dup", "first")]
    _finish(s, "dup")
    chunks = asyncio.run(s.collect_results("dup"))
    assert [c.data for c in chunks] == [b"answer"]
    assert s.aborted == []


def test_recently_completed_id_is_refused(tmp_path):
    s = _server(tmp_path)
    s.recently_completed["dup"] = 0.0
    with pytest.raises(HTTPException) as e:
        s.submit_request(text="again", request_id="dup", **TEXT)
    assert e.value.status_code == 409
    assert s.submitted == []


def test_id_is_free_again_once_released(tmp_path):
    s = _server(tmp_path)
    s.submit_request(text="first", request_id="dup", streaming=False, **TEXT)
    _finish(s, "dup")
    asyncio.run(s.collect_results("dup"))
    s.submit_request(text="second", request_id="dup", streaming=False, **TEXT)
    assert s.submitted == [("dup", "first"), ("dup", "second")]


@pytest.mark.parametrize("streaming", [False, True])
def test_id_is_free_as_soon_as_a_failed_request_answers(tmp_path, streaming):
    # A failed request is parked in recently_completed until the data worker
    # side lets go; its client already has the error and may retry at once.
    s = _server(tmp_path)
    s.preprocess_worker.finished_reading = (
        lambda rid, drained=True: s.reads_done.append((rid, drained))
    )
    s.reads_done = []
    s.submit_request(text="first", request_id="dup", streaming=streaming, **TEXT)
    with s.request_lock:
        req = s.pending_requests["dup"]
        req.error, req.error_status = "bad input", 400
        req.event.set()
        s.recently_completed["dup"] = time.time()

    if streaming:
        async def read_all():
            return [c async for c in s.iter_result_chunks("dup")]
        assert [c.modality for c in asyncio.run(read_all())] == ["error"]
    else:
        with pytest.raises(HTTPException) as e:
            asyncio.run(s.collect_results("dup"))
        assert e.value.status_code == 400

    s.submit_request(text="retry", request_id="dup", streaming=False, **TEXT)
    assert s.submitted == [("dup", "first"), ("dup", "retry")]
    # The data worker still gets its READS_DONE for the failed request.
    assert s.reads_done == [("dup", False)]


def test_a_stream_closed_at_its_error_does_not_abort_a_retry(tmp_path):
    # The id is free once the error is handed out; a consumer that closes the
    # stream there (or a client gone at that moment) must not abort a request
    # that already reuses it.
    s = _server(tmp_path)
    s.preprocess_worker.finished_reading = lambda rid, drained=True: None
    s.submit_request(text="first", request_id="dup", **TEXT)
    with s.request_lock:
        req = s.pending_requests["dup"]
        req.error, req.error_status = "engine failed", 500
        req.event.set()
        s.recently_completed["dup"] = time.time()

    async def scenario():
        batches = s.iter_result_chunk_batches("dup")
        assert [c.modality for c in await anext(batches)] == ["error"]
        s.submit_request(text="retry", request_id="dup", **TEXT)
        retry = s.pending_requests["dup"]
        await batches.aclose()
        return retry

    retry = asyncio.run(scenario())
    assert s.pending_requests.get("dup") is retry
    assert s.aborted == []


def _fail(server, rid, data_worker):
    """A failure as the message thread records it: an engine error sets
    ``error``; a data worker's also appends its error chunk."""
    with server.request_lock:
        req = server.pending_requests[rid]
        req.chunks.append(ResultChunk(request_id=rid, modality="text", data=b"a", metadata={}))
        if data_worker:
            req.chunks.append(ResultChunk(
                request_id=rid, modality="error", data=b"postprocess failed", metadata={"status": 500},
            ))
        req.error, req.error_status = "failed", 500
        req.event.set()
        server.recently_completed[rid] = time.time()


@pytest.mark.parametrize("failure", [None, "engine", "data_worker"])
def test_the_bridge_sends_nothing_after_the_first_error(tmp_path, failure):
    # The frontend ends a route on the first error chunk, and the id can be
    # live again before anything that follows; a second error or a done would
    # end the new route.
    from mstar.api_server.rust_frontend import RustFrontendBridge

    s = _server(tmp_path)
    s.preprocess_worker.finished_reading = lambda rid, drained=True: None
    s.submit_request(text="first", request_id="dup", **TEXT)
    if failure:
        _fail(s, "dup", data_worker=failure == "data_worker")
    else:
        _finish(s, "dup", data=b"a")

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = s
    sent = []
    bridge._send = sent.append
    asyncio.run(bridge._relay("dup"))
    kinds = [(m["t"], m.get("modality")) for m in sent]
    assert kinds == [("chunk", "text"), ("chunk", "error") if failure else ("done", None)]
    assert "dup" not in s.pending_requests and s.aborted == []


def _fail_when_submitted(server, rid, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with server.request_lock:
            if rid in server.pending_requests:
                break
        time.sleep(0.005)
    _fail(server, rid, data_worker=True)


def test_websocket_replies_once_to_a_data_worker_failure(monkeypatch, tmp_path):
    # The stream adds its own error after the data worker's; a second reply
    # under the id would read as the answer to a retry that reuses it.
    s = _server(tmp_path)
    s.preprocess_worker.finished_reading = lambda rid, drained=True: None
    monkeypatch.setattr(entrypoint, "api_server", s)

    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        failer = threading.Thread(target=_fail_when_submitted, args=(s, "dup"))
        failer.start()
        ws.send_text(json.dumps({"text": "first", "request_id": "dup"}))
        replies = [json.loads(ws.receive_text()) for _ in range(2)]
        failer.join()
        assert replies[0]["data"] and replies[1]["request_id"] == "dup"
        assert replies[1]["error"] == "postprocess failed" and replies[1]["status"] == 500

        finisher = threading.Thread(target=_finish_when_submitted, args=(s, "next"))
        finisher.start()
        ws.send_text(json.dumps({"text": "second", "request_id": "next"}))
        after = [json.loads(ws.receive_text()) for _ in range(2)]
        finisher.join()

    assert [r["request_id"] for r in after] == ["next", "next"]
    assert after[1] == {"request_id": "next", "finish": True}
    assert s.aborted == []


@pytest.mark.parametrize("streaming", ["true", "false"])
def test_generate_returns_409(monkeypatch, tmp_path, streaming):
    s = _server(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", s)
    s.submit_request(text="first", request_id="dup", **TEXT)
    original = s.pending_requests["dup"]

    r = TestClient(entrypoint.app).post(
        "/generate", data={"text": "second", "request_id": "dup", "streaming": streaming},
    )

    assert r.status_code == 409
    assert "already in use" in r.json()["detail"]
    assert s.pending_requests["dup"] is original
    assert s.submitted == [("dup", "first")] and s.aborted == []


def test_websocket_replies_409_and_keeps_the_socket(monkeypatch, tmp_path):
    s = _server(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", s)
    s.submit_request(text="first", request_id="dup", **TEXT)
    original = s.pending_requests["dup"]

    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "second", "request_id": "dup"}))
        reply = json.loads(ws.receive_text())
        assert reply["request_id"] == "dup" and reply["status"] == 409
        assert "already in use" in reply["error"]

        finisher = threading.Thread(target=_finish_when_submitted, args=(s, "other"))
        finisher.start()
        ws.send_text(json.dumps({"text": "third", "request_id": "other"}))
        replies = [json.loads(ws.receive_text()) for _ in range(2)]
        finisher.join()

    assert replies[0]["data"] and replies[1] == {"request_id": "other", "finish": True}
    assert s.pending_requests["dup"] is original
    assert s.aborted == []
