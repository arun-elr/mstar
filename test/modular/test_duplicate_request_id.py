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


@pytest.mark.parametrize("failed", [False, True])
def test_the_bridge_sends_no_done_after_an_error(failed):
    # The frontend ends a route on an error chunk, and the id can be live
    # again before a trailing done arrives; that done would end the new route.
    from mstar.api_server.rust_frontend import RustFrontendBridge

    async def chunks(rid):
        yield ResultChunk(request_id=rid, modality="text", data=b"a", metadata={})
        if failed:
            yield ResultChunk(request_id=rid, modality="error", data=b"boom", metadata={"status": 500})

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = SimpleNamespace(iter_result_chunks=chunks)
    sent = []
    bridge._send = sent.append
    asyncio.run(bridge._relay("dup"))
    kinds = [(m["t"], m.get("modality")) for m in sent]
    assert kinds == (
        [("chunk", "text"), ("chunk", "error")] if failed else [("chunk", "text"), ("done", None)]
    )


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
