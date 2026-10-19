"""Error text that reaches a client names no server path, user name or process
id (#592). A ValueError or TypeError raised on purpose keeps its message with
paths cut to base names; anything else becomes an error id, and the server log
carries the full error and traceback under that id."""

import asyncio
import getpass
import json
import os
import queue
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mstar.api_server import entrypoint
from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.rust_frontend import RustFrontendBridge
from mstar.engine.engine import ExecutingBatch
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.worker import Worker

USER = getpass.getuser()
ERROR_ID = re.compile(r"error id ([0-9a-f]{12})\)")


def _assert_clean(text: str) -> None:
    for leak in ("/tmp", "/dev/shm", "/home", "mstar_arena", "mstar_uploads", USER):
        assert leak not in text, f"{leak!r} reaches the client: {text!r}"


def _error_id(text: str) -> str:
    (error_id,) = ERROR_ID.findall(text)
    return error_id


def _logged(caplog, error_id: str):
    """The one log record for ``error_id``, with the traceback attached."""
    (record,) = [r for r in caplog.records if f"[error id {error_id}]" in r.getMessage()]
    assert record.exc_info is not None
    return record


def _upload(tmp_path, name: str) -> str:
    """An upload under a per-user upload dir, as a deployment may set it."""
    upload_dir = tmp_path / f"mstar_uploads_{USER}"
    upload_dir.mkdir()
    path = upload_dir / name
    path.write_bytes(b"not a picture")
    return str(path)


def _data_worker():
    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.out_queue = queue.Queue()
    wt.request_output_state = {}
    return wt


def test_a_decoder_failure_on_an_upload_reaches_the_client_as_an_error_id(tmp_path, caplog):
    path = _upload(tmp_path, "3f2a9c_photo.png")
    wt = _data_worker()
    with pytest.raises(OSError) as raised:
        Image.open(path)
    assert path in str(raised.value)  # the decoder names the upload
    wt._fail_request("r1", raised.value, "preprocessing")

    chunk = wt.out_queue.get_nowait()
    text = chunk.data.decode()
    _assert_clean(text)
    assert text.startswith("preprocessing failed: internal error (error id ")
    assert chunk.metadata["status"] == 500
    assert path in _logged(caplog, _error_id(text)).getMessage()


def test_a_deliberate_decoder_refusal_keeps_its_message_without_the_path(tmp_path, caplog):
    path = _upload(tmp_path, "3f2a9c_clip.mp4")
    wt = _data_worker()
    wt._fail_request("r1", ValueError(f"no video frames decoded from {path}"), "preprocessing")

    chunk = wt.out_queue.get_nowait()
    text = chunk.data.decode()
    _assert_clean(text)
    assert text == "preprocessing failed: ValueError: no video frames decoded from 3f2a9c_clip.mp4"
    assert chunk.metadata["status"] == 400
    assert any(path in r.getMessage() for r in caplog.records)


def test_an_arena_io_error_reaches_the_client_as_an_error_id(caplog):
    segment = f"/dev/shm/mstar_arena_worker_0_{os.getpid()}_9c1e.seg3"
    wt = _data_worker()
    wt._fail_request("r1", OSError(f"io on {segment}: Bad address"), "audio output transfer", count=2)

    texts = [wt.out_queue.get_nowait().data.decode() for _ in range(2)]
    for text in texts:
        _assert_clean(text)
    assert texts[0] == texts[1]
    assert segment in _logged(caplog, _error_id(texts[0])).getMessage()


def _worker(rids=("r1",)):
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.sent = []
    w.communicator = SimpleNamespace(send=lambda entity, msg: w.sent.append(msg))
    w.request_state = SimpleNamespace(per_request_info={rid: object() for rid in rids})
    w._graph_runtime = SimpleNamespace(
        get_rid_string=lambda h: h, set_in_flight=lambda *args: None,
    )
    w.scheduler = MicroScheduler.__new__(MicroScheduler)
    w.scheduler.failed_rids = set()
    w.scheduler.held_until = {}
    w.scheduler.admit_errors = {}
    w.scheduler.backlog = {}
    w._in_flight_rids = set(rids)
    return w


def test_a_worker_exception_reaches_the_client_as_an_error_id(tmp_path, caplog):
    tensor_file = f"{tmp_path}/mstar_tensors_{USER}/r1_audio.pt"
    w = _worker()
    w._handle_main_loop_error(FileNotFoundError(f"cannot open {tensor_file}"), (None, None), None)

    (message,) = w.sent
    text = message.body.errors["r1"]
    _assert_clean(text)
    assert text.startswith("Error in worker: internal error (error id ")
    assert tensor_file in _logged(caplog, _error_id(text)).getMessage()


def test_a_per_request_stage_failure_reaches_the_client_as_an_error_id(tmp_path, caplog):
    batch = SimpleNamespace(failed_requests={}, node_name="talker", graph_walk="decode")
    ExecutingBatch.register_failure(batch, 7, RuntimeError(f"bad weights at {tmp_path}"))
    ExecutingBatch.register_failure(batch, 8, ValueError(f"no frames in {tmp_path}/a_b.wav"))

    _assert_clean(batch.failed_requests[7])
    assert str(tmp_path) in _logged(caplog, _error_id(batch.failed_requests[7])).getMessage()
    assert batch.failed_requests[8] == "ValueError: no frames in a_b.wav"


class _FailingServer:
    def __init__(self, exc, upload_dir=None):
        self.exc = exc
        self.upload_dir = upload_dir

    async def wait_for_request_id(self, request_id):
        return None

    def submit_request(self, **kwargs):
        raise self.exc

    async def iter_result_chunks(self, rid):
        raise self.exc
        yield


def test_the_generate_catch_all_answers_with_an_error_id(monkeypatch, caplog):
    segment = f"/dev/shm/mstar_arena_api_{os.getpid()}_77.seg0"
    monkeypatch.setattr(entrypoint, "api_server", _FailingServer(RuntimeError(f"io on {segment}")))

    response = TestClient(entrypoint.app).post("/generate", data={"text": "hi"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    _assert_clean(detail)
    assert detail.startswith("internal error (error id ")
    assert segment in _logged(caplog, _error_id(detail)).getMessage()


def test_the_generate_catch_all_keeps_a_deliberate_refusal(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", _FailingServer(ValueError("Unsupported modality: 'smell'")))
    response = TestClient(entrypoint.app).post("/generate", data={"text": "hi"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Unsupported modality: 'smell'"}


def test_generate_ws_answers_with_an_error_id(monkeypatch, tmp_path, caplog):
    segment = f"/dev/shm/mstar_arena_api_{os.getpid()}_77.seg0"
    server = _FailingServer(RuntimeError(f"io on {segment}"), upload_dir=tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", server)

    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "hi", "request_id": "r1"}))
        reply = json.loads(ws.receive_text())

    assert reply["request_id"] == "r1"
    _assert_clean(reply["error"])
    assert segment in _logged(caplog, _error_id(reply["error"])).getMessage()


def test_the_rust_frontend_bridge_answers_with_an_error_id(tmp_path, caplog):
    tensor_file = f"{tmp_path}/mstar_tensors_{USER}/r1.pt"
    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.sent = []
    bridge._send = bridge.sent.append
    bridge._keys = {}
    bridge.server = _FailingServer(FileNotFoundError(f"cannot open {tensor_file}"))

    bridge._submit({"rid": "r1", "text": "hi"})
    asyncio.run(bridge._relay("r2", None, bridge.server.iter_result_chunks("r2")))

    assert [m["rid"] for m in bridge.sent] == ["r1", "r2"]
    for message in bridge.sent:
        assert message["t"] == "err"
        _assert_clean(message["msg"])
        assert tensor_file in _logged(caplog, _error_id(message["msg"])).getMessage()


@pytest.mark.parametrize("raw,scrubbed", [
    ("Error opening '/tmp/up_alice/9f_a.wav': Format not recognised.",
     "Error opening '9f_a.wav': Format not recognised."),
    ("io on /dev/shm/mstar_arena_w_41_t.seg2: Bad address", "io on mstar_arena_w_41_t.seg2: Bad address"),
    ("cannot read file:///home/alice/x.png", "cannot read x.png"),
    ("not a file: /home/alice/", "not a file: <path>"),
    ("Error opening '/tmp/uploads (alice)/7_bad.wav': Format not recognised.",
     "Error opening '7_bad.wav': Format not recognised."),
    ('[Errno 2] No such file or directory: "/srv/my uploads/[alice]/x.pt"',
     '[Errno 2] No such file or directory: "x.pt"'),
    ("fetch https://example.com/a/b.png failed", "fetch https://example.com/a/b.png failed"),
    ("expected 3/4 of shape (2, 3)/2 and/or ./rel/path", "expected 3/4 of shape (2, 3)/2 and/or ./rel/path"),
])
def test_scrub_paths_cuts_absolute_paths_to_base_names(raw, scrubbed):
    from mstar.utils.client_errors import scrub_paths

    assert scrub_paths(raw) == scrubbed


def test_a_failed_kv_retrieve_reaches_the_client_without_its_path(caplog):
    from concurrent.futures import Future

    from mstar.engine.resources.kv.manager import CacheStream, KVManager

    snapshot = "/srv/private/alice/kv-worker-12345.pt"
    future = Future()
    future.set_exception(FileNotFoundError(2, "No such file or directory", snapshot))
    manager = SimpleNamespace(_streams={"r1": {"main": CacheStream(read_pending=True, read_future=future)}})
    with caplog.at_level("ERROR"):
        reasons = [KVManager._check_ready(manager, "r1", "main") for _ in range(2)]
    assert [ready for ready, _ in reasons] == [False, False]
    messages = {reason.message for _, reason in reasons}
    (message,) = messages
    _assert_clean(message)
    assert "/srv" not in message and "alice" not in message
    assert snapshot in _logged(caplog, _error_id(message)).getMessage()
