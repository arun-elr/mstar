"""A request_id reused while the engine still tears the older request down (#485).

A client can reuse its ``request_id`` as soon as the API server frees it (its
answer arrived, or it aborted), while the conductor and the data worker are
still tearing the older request down. Every message below the API server names
a request by its id, so the API server runs each submission under a key of its
own and the older request's teardown can never name the newer one.

The data worker is the real ``PreprocessWorkerThread`` loop on its own thread,
the conductor is the hand-built one of ``test_teardown_barrier``, and the API
server is real down to its preprocess-worker facade. Messages are delivered in
the order the conductor and the data worker send them.
"""

from __future__ import annotations

import asyncio
import itertools
import queue
import threading
import time
from types import SimpleNamespace

import pytest
from test_teardown_barrier import PREPROCESS, _conductor, _remove_targets, _request_data

from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
from mstar.api_server.entrypoint import APIServer
from mstar.api_server.request_types import ResultChunk
from mstar.utils.ipc_format import (
    ConductorMessageType,
    NewRequestConductor,
    ReadsDone,
    RemoveRequest,
    WorkerMessage,
    WorkerMessageType,
)

TEXT = dict(input_modalities=["text"], output_modalities=["text"])


class _Model:
    def process_prompt(self, *args, **kwargs):
        return None

    def postprocess(self, output, modality, request_kwargs=None):
        return f"<{int(output[0])}>".encode()


class _Socket:
    """The data worker's socket to the conductor; each loop pass polls it once."""

    def __init__(self):
        self.inbox = queue.Queue()
        self.sent = []
        self.polls = 0
        self.cond = threading.Condition()

    def get_all_new_messages(self):
        msgs = []
        while not self.inbox.empty():
            msgs.append(self.inbox.get())
        with self.cond:
            self.polls += 1
            self.cond.notify_all()
        return msgs

    def send(self, entity, msg):
        self.sent.append((entity, msg))

    def deliver(self, message_type, body):
        self.inbox.put(WorkerMessage(message_type=message_type, body=body))

    def sent_of(self, message_type):
        return [m.body for e, m in self.sent if e == "conductor" and m.message_type == message_type]


class _TensorManager:
    def __init__(self):
        self.forced = []

    def store_and_return_tensor_info(self, request_id, tensors):
        return {}

    def register_for_send(self, request_id, tensor_infos):
        pass

    def get_ready_tensors(self):
        return {}

    def has_inflight_reads(self, request_id):
        return False

    def force_cleanup_request(self, request_id):
        self.forced.append(request_id)

    def ack_unread_tensors(self, request_id, graph_edges):
        pass


def _api_server(preprocess_worker):
    s = APIServer.__new__(APIServer)
    s.request_lock = threading.Lock()
    s.fatal_error = None
    s.pending_requests = {}
    s.recently_completed = {}
    s._recently_completed_ttl = 15.0
    s._live = {}
    s._client_ids = {}
    s._generation = itertools.count(1)
    s.timeout_seconds = 5
    s.log_stats = False
    s.enable_nvtx = False
    s.preprocess_worker = preprocess_worker
    return s


class _Rig:
    def __init__(self):
        self.socket = _Socket()
        self.tm = _TensorManager()
        self.stop = threading.Event()
        self.worker = PreprocessWorkerThread(
            in_queue=queue.Queue(),
            result_tensor_queue=queue.Queue(),
            out_queue=queue.Queue(),
            profile_queue=queue.Queue(),
            cleanup_request_queue=queue.Queue(),
            abort_request_queue=queue.Queue(),
            reads_done_queue=queue.Queue(),
            discard_tensor_queue=queue.Queue(),
            stop_event=self.stop,
            communicator=self.socket,
            tensor_manager=self.tm,
            model=_Model(),
        )
        facade = PreprocessWorker.__new__(PreprocessWorker)
        facade.request_input_queue = self.worker.in_queue
        facade.result_tensor_input_queue = self.worker.result_tensor_queue
        facade.abort_request_queue = self.worker.abort_request_queue
        facade.reads_done_queue = self.worker.reads_done_queue
        facade.cleanup_request_queue = self.worker.cleanup_request_queue
        facade.discard_tensor_queue = self.worker.discard_tensor_queue
        facade.output_loop_idxs = {}
        facade.per_request_reading_tensors = {}
        self.data_worker = facade
        self.server = _api_server(facade)
        self.thread = threading.Thread(target=self.worker.run, daemon=True)
        self.thread.start()

    def settle(self):
        """Return once a whole loop pass has run after everything queued so far."""
        with self.socket.cond:
            target = self.socket.polls + 2
            assert self.socket.cond.wait_for(lambda: self.socket.polls >= target, timeout=10)

    def submit(self, text):
        """Submit under the client id ``X``; returns the NEW_REQUEST the
        conductor gets for it."""
        self.server.submit_request(text=text, request_id="X", streaming=False, **TEXT)
        self.settle()
        return self.socket.sent_of(ConductorMessageType.NEW_REQUEST)[-1]

    def output(self, request_id, value=7):
        """A worker's output for ``request_id``, as the message thread hands it
        over; returns the chunk the data worker emits, or None."""
        self.data_worker.new_result_token(request_id, value, None, "text_inputs", "text")
        try:
            return self.worker.out_queue.get(timeout=2)
        except queue.Empty:
            return None

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)


@pytest.fixture
def rig():
    r = _Rig()
    yield r
    r.close()


def test_the_older_request_s_drain_does_not_free_the_reuse_on_the_conductor(rig):
    """The client aborts X and resubmits X at once. The conductor admits the
    newer request while the older one still drains; the older one's last
    READS_DONE must free only the older one."""
    c = _conductor({})
    admitted = []
    c._admit_request = lambda body: admitted.append(body.request_id)

    first = rig.submit("first")
    c._ingest_request(first)
    c.requests[first.request_id] = _request_data(workers=("w0",))  # its dispatch

    rig.server.abort_request("X")
    rig.settle()
    (abort,) = rig.socket.sent_of(ConductorMessageType.ABORT_REQUEST)
    c._abort_request(abort.request_id)
    for reads_done in rig.socket.sent_of(ConductorMessageType.READS_DONE):
        c._handle_reads_done(reads_done)
    assert first.request_id in c.draining

    second = rig.submit("second")
    c._ingest_request(second)
    assert admitted == [first.request_id, second.request_id]
    newer = _request_data(workers=("w1",))
    c.requests[second.request_id] = newer

    c._handle_reads_done(ReadsDone(request_id=first.request_id, entity_id="w0"))

    assert c.requests.get(second.request_id) is newer, (
        "the older request's teardown freed the newer request's state"
    )
    assert _remove_targets(c, second.request_id) == set()
    assert _remove_targets(c, first.request_id) == {"w0", PREPROCESS}


def test_the_older_request_s_drain_does_not_drop_the_reuse_s_outputs(rig):
    """The data worker is still draining the aborted request when the reuse's
    first output arrives."""
    rig.submit("first")
    rig.server.abort_request("X")
    rig.settle()
    second = rig.submit("second")

    chunk = rig.output(second.request_id)

    assert chunk is not None, "the newer request's output was dropped as draining"
    assert chunk.data == b"<7>"
    assert chunk.request_id in rig.server.pending_requests


def test_the_older_request_s_remove_does_not_unlink_the_reuse_s_inputs(rig):
    """The older request finished and its client read the answer; the reuse is
    preprocessed before the older request's REMOVE_REQUEST lands."""
    first = rig.submit("first")
    s = rig.server
    with s.request_lock:
        s.recently_completed[first.request_id] = time.time()
        s.pending_requests[first.request_id].event.set()
    assert asyncio.run(s.collect_results("X")) == []
    with s.request_lock:
        s._prune_recently_completed()
    rig.settle()

    second = rig.submit("second")
    rig.socket.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest(first.request_id))
    rig.settle()

    assert second.request_id not in rig.tm.forced, (
        "the older request's REMOVE_REQUEST dropped the inputs the newer one is using"
    )
    assert rig.tm.forced == [first.request_id]
    chunk = rig.output(second.request_id)
    assert chunk is not None and chunk.data == b"<7>"


def test_the_seed_still_derives_from_the_client_request_id(rig):
    from mstar.conductor.conductor import _req_id_to_seed, _request_seed, _seed_id

    first = rig.submit("first")
    rig.server.abort_request("X")
    second = rig.submit("second")

    assert first.request_id != second.request_id
    assert first.client_request_id == second.client_request_id == "X"
    assert _request_seed({}, _seed_id(second)) == _req_id_to_seed("X")


def test_an_empty_pinned_request_id_keeps_its_seed():
    from mstar.conductor.conductor import _req_id_to_seed, _seed_id

    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="", **TEXT)
    s.abort_request("")
    s.submit_request(text="second", request_id="", **TEXT)

    seeds = {
        _req_id_to_seed(_seed_id(NewRequestConductor(
            request_id=inp.request_id,
            initial_signals={},
            initial_input_modalities=["text"],
            initial_output_modalities=["text"],
            input_metadata={},
            model_kwargs={},
            client_request_id=inp.client_request_id,
        )))
        for inp in pw.inputs
    }
    assert seeds == {_req_id_to_seed("")}


def test_chunks_reach_the_client_under_its_own_request_id():
    """The engine's chunks name the request by its key; what the stream and
    collect hand back names it by the client's id."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", streaming=False, **TEXT)
    key = pw.inputs[0].request_id
    engine_chunks = [ResultChunk(request_id=key, modality="text", data=b"hi")]
    s.preprocess_worker.get_result_chunks = lambda: [engine_chunks.pop()] if engine_chunks else []
    s.preprocess_worker.get_profile_updates = lambda: []
    s.communicator = SimpleNamespace(get_all_new_messages=lambda: [])
    s.conductor_proc = None
    s._liveness_interval_s = 0.5
    s.running = True
    thread = threading.Thread(target=s._process_messages, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not s.pending_requests[key].chunks and time.time() < deadline:
            time.sleep(0.005)
    finally:
        s.running = False
        thread.join(timeout=10)
    s.pending_requests[key].event.set()

    chunks = asyncio.run(s.collect_results("X"))
    assert [(c.request_id, c.data) for c in chunks] == [("X", b"hi")]


# ── the API server alone ─────────────────────────────────────────────────────

class _PreprocessWorker:
    def __init__(self):
        self.inputs = []
        self.aborted = []

    def new_request(self, inp):
        self.inputs.append(inp)

    def abort_request(self, request_id):
        self.aborted.append(request_id)


def test_a_closing_stream_does_not_abort_a_retry_under_its_id():
    """The request fails; the stream frees it and yields the error, the client
    retries under the same id, then closes the stream at the error. The
    stream's cleanup must abort only its own request."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", **TEXT)
    first = pw.inputs[0].request_id

    async def run():
        stream = s.iter_result_chunks("X")
        with s.request_lock:
            req = s.pending_requests[first]
            req.error, req.error_status = "boom", 500
            req.event.set()
        assert (await stream.__anext__()).modality == "error"
        s.submit_request(text="retry", request_id="X", **TEXT)
        await stream.aclose()

    asyncio.run(run())

    assert pw.aborted == [], "the old stream aborted the retry"
    assert pw.inputs[1].request_id in s.pending_requests


def test_a_second_submission_under_a_pending_id_replaces_the_first():
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", streaming=False, **TEXT)
    s.submit_request(text="second", request_id="X", streaming=False, **TEXT)
    first, second = (inp.request_id for inp in pw.inputs)

    assert pw.aborted == [first]
    assert list(s.pending_requests) == [second]

    s.abort_request("X")
    assert pw.aborted == [first, second]


def test_nothing_is_kept_once_the_request_is_done():
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.preprocess_worker.finished_reading = lambda rid, drained=True: None
    s.submit_request(text="first", request_id="X", streaming=False, **TEXT)
    key = pw.inputs[0].request_id
    with s.request_lock:
        s.recently_completed[key] = time.time()
        s.pending_requests[key].event.set()

    asyncio.run(s.collect_results("X"))
    assert s._live == {"X": key}  # its outputs are still being released
    with s.request_lock:
        s._prune_recently_completed()

    assert s._live == {} and s._client_ids == {}
    assert not s.pending_requests and not s.recently_completed
