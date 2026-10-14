"""A failed async tensor read (a Mooncake/RDMA transfer error) fails only its
own request. Before, ``get_ready_tensors`` re-raised the same failed future on
every poll, so no other request on that worker or data worker received its
inputs until the failed request was torn down (the repeated "Mooncake read
failed" of #332)."""

import ctypes
import gc
import threading
import weakref
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    READ_ASYNC_DRAIN_ENV,
    READ_DRAIN_CAP_ENV,
    READ_QUARANTINE_BUDGET_ENV,
    AsyncMooncakeReader,
    FutureAndPointers,
    MooncakeCommunicationManager,
    SharedMemoryCommunicationManager,
    TensorCommunicationManager,
    read_async_drain_enabled,
    read_drain_cap_seconds,
    read_quarantine_budget_bytes,
)
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.profile.format import ReadQuarantine
from mstar.utils.ipc_format import WorkerMessageType


class _Communicator(BaseCommunicator):
    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return []


def _manager(tmp_path):
    comm = _Communicator()
    mgr = SharedMemoryCommunicationManager(
        "worker_1", "host", "cpu", comm, shm_dir=str(tmp_path),
    )
    return mgr, comm


def _edge(*uuids):
    edge = GraphEdge(next_node="node", name="x")
    edge.tensor_info = [SimpleNamespace(uuid=u, source_entity="worker_0") for u in uuids]
    return edge


def _failed_future(message="Mooncake read failed. Status: -1"):
    f = Future()
    f.set_exception(RuntimeError(message))
    return f


def _read(rid, edge, future=None):
    return FutureAndPointers(future=future, graph_edges=[edge], request_id=rid)


def _kept_buffers(mgr):
    return [buffer for kept in mgr._kept_reads for buffer in kept.buffers]


def _acked(comm):
    return [
        (msg.body.request_id, dict(msg.body.successful_tensors))
        for _, msg in comm.sent
        if msg.message_type == WorkerMessageType.TENSOR_RECEIVED
    ]


def test_a_failed_read_does_not_block_other_requests(tmp_path):
    mgr, comm = _manager(tmp_path)
    bad, healthy = _edge(11), _edge()
    mgr.pending = [_read(7, bad, _failed_future()), _read(8, healthy)]

    assert mgr.get_ready_tensors() == {8: [healthy]}
    assert mgr.pending == []
    (failed,) = mgr.take_failed_reads()
    assert failed.request_id == 7 and failed.graph_edges == [bad]
    assert "Mooncake read failed" in str(failed.error)
    # the transfer may not have stopped, so the producer keeps its copy
    assert _acked(comm) == []

    assert mgr.get_ready_tensors() == {}
    assert mgr.take_failed_reads() == []


def test_a_cancelled_read_counts_as_failed(tmp_path):
    mgr, _ = _manager(tmp_path)
    cancelled = Future()
    cancelled.cancel()
    mgr.pending = [_read(7, _edge(11), cancelled)]

    assert mgr.get_ready_tensors() == {}
    (failed,) = mgr.take_failed_reads()
    assert failed.request_id == 7 and mgr.pending == []


def test_later_reads_of_a_failed_request_are_dropped_until_cleanup(tmp_path):
    mgr, comm = _manager(tmp_path)
    in_flight = Future()
    mgr.pending = [
        _read(7, _edge(11), _failed_future()),
        _read(7, _edge(12), in_flight),
        _read(7, _edge(13)),
    ]

    assert mgr.get_ready_tensors() == {}
    first, later = mgr.take_failed_reads()
    assert first.error is later.error
    # a read still touching the producer keeps the drain waiting
    assert [ep.future for ep in mgr.pending] == [in_flight]
    assert mgr.has_inflight_reads(7)

    in_flight.set_result(None)
    assert mgr.get_ready_tensors() == {}
    (dropped,) = mgr.take_failed_reads()
    assert dropped.error is first.error
    assert mgr.pending == [] and not mgr.has_inflight_reads(7)
    # the reads that completed are acked; the failed one is not
    assert _acked(comm) == [("7", {13: 1}), ("7", {12: 1})]

    mgr.cleanup_request(7)
    fresh = _edge()
    mgr.pending = [_read(7, fresh)]
    assert mgr.get_ready_tensors() == {7: [fresh]}
    assert mgr.take_failed_reads() == []


class _Engine:
    """Copies like a transfer engine; session "timed-out" returns -1 the way
    Mooncake's sync read does when it gives up on a batch it could not free."""

    def __init__(self, refuse_registrations=()):
        self.unregistered = []
        self.registered = []
        self.reads = []
        # indices of register_memory calls that fail, the way Mooncake
        # returns nonzero for a buffer it cannot pin
        self.refuse_registrations = set(refuse_registrations)
        self.started = threading.Event()
        self.release = threading.Event()

    def register_memory(self, ptr, nbytes):
        call = len(self.registered)
        self.registered.append(ptr)
        return -1 if call in self.refuse_registrations else 0

    def unregister_memory(self, ptr):
        self.unregistered.append(ptr)
        return 0

    def batch_transfer_sync_read(self, session, local, remote, lengths):
        self.reads.append(list(local))
        if session == "timed-out":
            return -1
        if session == "slow-timeout":
            self.started.set()
            assert self.release.wait(5)
            return -1
        for dst, src, n in zip(local, remote, lengths, strict=True):
            ctypes.memmove(dst, src, n)
        return 0


def _mooncake_manager(engine):
    mgr = MooncakeCommunicationManager.__new__(MooncakeCommunicationManager)
    TensorCommunicationManager.__init__(mgr, "receiver", "receiver", "cpu", _Communicator(), engine)
    mgr.protocol = CommProtocol.RDMA
    mgr._async_reader = AsyncMooncakeReader(engine, "cpu")
    return mgr


def _pointer(uuid, src, session):
    return TensorPointerInfo(
        list(src.shape), src.dtype, src.nbytes, src.data_ptr(),
        list(src.stride()), uuid, session, "producer",
    )


def _remote_edge(uuid, src, session):
    return GraphEdge(next_node="node", name="x", tensor_info=[_pointer(uuid, src, session)])


def test_a_failed_transfer_keeps_its_buffers_through_teardown():
    engine = _Engine()
    mgr = _mooncake_manager(engine)
    src = torch.tensor([42, 43])
    try:
        (bad,) = mgr.start_read_tensors("bad", [_remote_edge(31, src, "timed-out")])
        good_edge = _remote_edge(32, src, "peer")
        (good,) = mgr.start_read_tensors("good", [good_edge])
        buffer = mgr.get_tensor(31)
        for f in (bad, good):
            f.exception(timeout=5)

        assert mgr.get_ready_tensors() == {"good": [good_edge]}
        assert torch.equal(mgr.get_tensor(32), src)
        assert [f.request_id for f in mgr.take_failed_reads()] == ["bad"]

        mgr.force_cleanup_request("bad")
        # still registered and alive: the transfer may write into it
        assert buffer.data_ptr() not in engine.unregistered
        assert any(t is buffer for t in _kept_buffers(mgr))
        assert _acked(mgr.communicator) == [("good", {32: 1})]

        good_ptr = mgr.get_tensor(32).data_ptr()
        mgr.dereference(32)
        mgr.force_cleanup_request("good")
        assert engine.unregistered == [good_ptr]
        assert not mgr.tensor_store.get_all_uuids("good")
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_teardown_before_the_next_poll_still_keeps_a_failed_transfers_buffers():
    """The read fails after the last poll, so only teardown sees it."""
    engine = _Engine()
    mgr = _mooncake_manager(engine)
    try:
        (future,) = mgr.start_read_tensors("bad", [_remote_edge(41, torch.tensor([1]), "slow-timeout")])
        buffer = mgr.get_tensor(41)
        assert engine.started.wait(5)
        assert mgr.get_ready_tensors() == {}
        engine.release.set()
        assert future.exception(timeout=5) is not None
        assert not mgr.has_inflight_reads("bad")

        mgr.force_cleanup_request("bad")

        assert buffer.data_ptr() not in engine.unregistered
        assert any(t is buffer for t in _kept_buffers(mgr))
        assert _acked(mgr.communicator) == [] and mgr.pending == []
    finally:
        engine.release.set()
        mgr._async_reader._executor.shutdown(wait=True)


def test_kept_buffers_are_counted_and_flagged_past_the_budget(monkeypatch, caplog):
    """Without the drain check kept buffers are never freed, so they are
    counted (reads and bytes) and the process reports itself over budget once
    the bytes pass it, logging that once."""
    monkeypatch.setenv(READ_QUARANTINE_BUDGET_ENV, str(20 / (1 << 20)))  # 20 bytes
    engine = _Engine()
    mgr = _mooncake_manager(engine)
    try:
        assert mgr.read_quarantine() == ReadQuarantine("receiver", 0, 0, 20)
        seen = []
        for i, rid in enumerate(("a", "b", "c")):
            uuid = 50 + i
            mgr.uuid_to_edge_name[uuid] = "x"
            (future,) = mgr.start_read_tensors(rid, [_remote_edge(uuid, torch.tensor([1, 2]), "timed-out")])
            future.exception(timeout=5)
            with caplog.at_level("WARNING", logger="mstar.communication.tensors"):
                mgr.get_ready_tensors()
            assert uuid not in mgr.uuid_to_edge_name and uuid not in mgr.uuid_to_shard_dim
            seen.append((mgr.read_quarantine(), mgr.quarantine_over_budget()))

        assert seen == [
            (ReadQuarantine("receiver", 1, 16, 20), False),
            (ReadQuarantine("receiver", 2, 32, 20), True),
            (ReadQuarantine("receiver", 3, 48, 20), True),
        ]
        assert "(3 reads, 48 bytes kept so far; budget 20 bytes)" in caplog.text
        over = [r for r in caplog.records if "reporting unhealthy" in r.getMessage()]
        assert len(over) == 1 and over[0].levelname == "ERROR"
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_read_quarantine_budget_setting(monkeypatch):
    monkeypatch.delenv(READ_QUARANTINE_BUDGET_ENV, raising=False)
    assert read_quarantine_budget_bytes() == 1024 << 20
    monkeypatch.setenv(READ_QUARANTINE_BUDGET_ENV, "0.5")
    assert read_quarantine_budget_bytes() == 512 << 10
    for bad in ("lots", "-1", "nan", "inf", "-inf"):
        monkeypatch.setenv(READ_QUARANTINE_BUDGET_ENV, bad)
        with pytest.raises(ValueError, match=READ_QUARANTINE_BUDGET_ENV):
            read_quarantine_budget_bytes()


class _CountingFuture:
    """A read future that counts the event queries made on it."""

    def __init__(self):
        self.finished = False
        self.done_calls = 0
        self.result_calls = 0

    def done(self):
        self.done_calls += 1
        return self.finished

    def result(self):
        self.result_calls += 1


def test_a_finished_read_is_queried_once_per_poll_and_its_result_checked_once(tmp_path):
    """A read that finished behind an unfinished read of its request waits
    for it; while it waits, polling it costs nothing more."""
    mgr, _ = _manager(tmp_path)
    earlier, later = _CountingFuture(), _CountingFuture()
    later.finished = True
    first, second = _edge(), _edge()
    mgr.pending = [_read(7, first, earlier), _read(7, second, later)]

    for _ in range(3):
        assert mgr.get_ready_tensors() == {}
    assert earlier.done_calls == 3 and earlier.result_calls == 0
    assert later.done_calls == 1 and later.result_calls == 1

    earlier.finished = True
    assert mgr.get_ready_tensors() == {7: [first, second]}
    assert earlier.done_calls == 4 and earlier.result_calls == 1
    assert later.done_calls == 1 and later.result_calls == 1


def test_log_stats_report_shows_kept_read_buffers(tmp_path):
    from mstar.profile.display import pretty_print_profile
    from mstar.profile.format import RequestProfile

    out = tmp_path / "stats.txt"
    pretty_print_profile(RequestProfile(rid="r"), str(out))
    assert "Read quarantine" not in out.read_text()

    pretty_print_profile(RequestProfile(rid="r", read_quarantine=[
        ReadQuarantine("worker_0", 1229, 117 << 20, 1 << 30),
        ReadQuarantine("api_server_preprocess_worker", 2, 2048, 1024),
    ]), str(out))
    text = out.read_text()
    assert "Read quarantine (buffers kept from failed reads, process totals):" in text
    assert "worker_0" in text and "117.0 MiB  (x1229)  budget 1.0 GiB\n" in text
    assert "2.0 KiB  (x2)  budget 1.0 KiB  OVER BUDGET" in text


def test_teardown_during_a_running_transfer_keeps_its_buffers():
    engine = _Engine()
    mgr = _mooncake_manager(engine)
    try:
        mgr.start_read_tensors("r", [_remote_edge(42, torch.tensor([1]), "slow-timeout")])
        buffer = mgr.get_tensor(42)
        assert engine.started.wait(5)

        mgr.force_cleanup_request("r")

        assert buffer.data_ptr() not in engine.unregistered
        assert any(t is buffer for t in _kept_buffers(mgr))
        assert _acked(mgr.communicator) == []
    finally:
        engine.release.set()
        mgr._async_reader._executor.shutdown(wait=True)


def test_a_refused_registration_fails_only_its_read_and_keeps_nothing():
    """A receive buffer Mooncake will not register is never read into, so its
    read fails at once, nothing is kept, and cleanup does not unregister it."""
    engine = _Engine(refuse_registrations={1})
    mgr = _mooncake_manager(engine)
    src = torch.tensor([42, 43])
    try:
        good_edge = _remote_edge(61, src, "peer")
        (good,) = mgr.start_read_tensors("good", [good_edge])
        (bad,) = mgr.start_read_tensors("bad", [_remote_edge(62, src, "peer")])
        refused = mgr.get_tensor(62).data_ptr()
        assert bad.done() and "registration failed" in str(bad.exception())
        good.result(timeout=5)

        assert mgr.get_ready_tensors() == {"good": [good_edge]}
        (failed,) = mgr.take_failed_reads()
        assert failed.request_id == "bad" and failed.error is bad.exception()
        assert engine.reads == [[mgr.get_tensor(61).data_ptr()]]
        assert mgr._kept_reads == []
        assert mgr.read_quarantine() == ReadQuarantine(
            "receiver", 0, 0, mgr.quarantine_budget_bytes, registration_failures=1,
        )
        assert not mgr.quarantine_over_budget()

        mgr.force_cleanup_request("bad")
        assert refused not in engine.unregistered
        assert not mgr.tensor_store.get_all_uuids("bad")
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_a_registration_refused_partway_through_a_read_unregisters_only_what_registered():
    engine = _Engine(refuse_registrations={1})
    mgr = _mooncake_manager(engine)
    srcs = [torch.tensor([i, i]) for i in range(3)]
    edge = GraphEdge(next_node="node", name="x", tensor_info=[
        _pointer(70 + i, src, "peer") for i, src in enumerate(srcs)
    ])
    try:
        (future,) = mgr.start_read_tensors("r", [edge])
        ptrs = [mgr.get_tensor(70 + i).data_ptr() for i in range(3)]
        assert "registration failed" in str(future.exception())
        # the third buffer is not offered once the second is refused, and no read is posted
        assert engine.registered == ptrs[:2] and engine.reads == []
        assert [mgr.tensor_store.is_registered(70 + i) for i in range(3)] == [True, False, False]

        assert mgr.get_ready_tensors() == {}
        assert [f.request_id for f in mgr.take_failed_reads()] == ["r"]
        assert mgr.read_quarantine().reads == 0 and mgr.registration_failures == 1

        mgr.force_cleanup_request("r")
        assert engine.unregistered == ptrs[:1]
        assert not mgr.tensor_store.get_all_uuids("r")
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_log_stats_report_shows_registration_failures(tmp_path):
    from mstar.profile.display import pretty_print_profile
    from mstar.profile.format import RequestProfile

    out = tmp_path / "stats.txt"
    pretty_print_profile(RequestProfile(rid="r", read_quarantine=[
        ReadQuarantine("worker_0", 0, 0, 1 << 30, registration_failures=3),
        ReadQuarantine("worker_1", 1, 2048, 1 << 30),
    ]), str(out))
    text = out.read_text()
    assert "worker_0" in text and "0 B  (x0)  budget 1.0 GiB  registration failures 3\n" in text
    assert "2.0 KiB  (x1)  budget 1.0 GiB\n" in text


def test_the_api_server_reports_its_own_registration_failures():
    from mstar.api_server.entrypoint import APIServer

    server = APIServer.__new__(APIServer)
    server.worker_read_quarantine = {}
    own = ReadQuarantine("api_server_preprocess_worker", 0, 0, 1 << 30)
    server.preprocess_worker = SimpleNamespace(
        tensor_manager=SimpleNamespace(read_quarantine=lambda: own),
    )
    assert server.read_quarantines() == []
    own.registration_failures = 2
    assert server.read_quarantines() == [own]
    # they keep nothing, so they never fail /health
    server.fatal_error = None
    assert server.health_problem() is None


# ── draining kept reads (MSTAR_READ_ASYNC_DRAIN) ───────────────────────────


class _AsyncEngine(_Engine):
    """Mooncake's async read API. Each batch reports a state per poll by the
    session it reads from; 1 (completed) copies and, like -1, frees the batch.
    A poll of a freed batch is recorded as a violation, as is a poll by a
    thread that has already handed the batch to another."""

    def __init__(self, hcas=()):
        super().__init__()
        self.hcas = list(hcas)
        self.lock = threading.Lock()
        self.gate = threading.Event()
        self.next_id = 100
        self.batches = {}
        self.freed = set()
        self.polls = []  # (thread name, batch id)
        self.submits = []  # (session, local ptr)
        self.violations = []

    def get_local_topology(self):
        import json
        return json.dumps({"cpu:0": [self.hcas, []]})

    def batch_transfer_sync_read(self, *args):
        self.violations.append("sync read with the async drain on")
        return -1

    def batch_transfer_async_read(self, session, local, remote, lengths):
        with self.lock:
            attempt = sum(1 for s, p in self.submits if (s, p) == (session, local[0]))
            self.submits.append((session, local[0]))
            if session == "no-submit":
                return 0
            batch_id = self.next_id
            self.next_id += 1
            self.batches[batch_id] = (session, attempt, local[0], remote[0], lengths[0])
            return batch_id

    def _state(self, session, attempt):
        if session == "peer" or (session == "flaky" and attempt > 0):
            return 1
        if session in ("fails", "flaky"):
            return -1
        if session == "gated":
            return 1 if self.gate.is_set() else 0
        return 0  # "stuck": never reports

    def transfer_check_status(self, batch_id):
        with self.lock:
            thread = threading.current_thread().name
            if batch_id in self.freed:
                self.violations.append(f"batch {batch_id} polled after it was freed")
                return 0
            owners = [t for t, i in self.polls if i == batch_id]
            if owners and owners[-1] != thread and thread in owners:
                self.violations.append(f"batch {batch_id} polled again after it was handed over")
            self.polls.append((thread, batch_id))
            session, attempt, dst, src, n = self.batches[batch_id]
            state = self._state(session, attempt)
            if state == 1:
                ctypes.memmove(dst, src, n)
            if state in (1, -1):
                self.freed.add(batch_id)
            return state


def _draining_manager(engine, timeout_s=0.05):
    mgr = _mooncake_manager(engine)
    mgr._async_reader._executor.shutdown(wait=True)
    mgr._async_reader = AsyncMooncakeReader(engine, "cpu", async_drain=True)
    mgr._async_reader._transfer_timeout_s = timeout_s
    mgr._transfer_status = mgr._async_reader.transfer_status
    return mgr


def _polls_from(engine, thread):
    return [i for t, i in engine.polls if t == thread]


def test_the_async_read_takes_one_batch_per_tensor():
    engine = _AsyncEngine()
    mgr = _draining_manager(engine)
    srcs = [torch.tensor([5, 6]), torch.tensor([7, 8, 9])]
    edge = GraphEdge(next_node="node", name="x", tensor_info=[
        _pointer(80 + i, src, "peer") for i, src in enumerate(srcs)
    ])
    try:
        (future,) = mgr.start_read_tensors("r", [edge])
        future.result(timeout=5)
        assert mgr.get_ready_tensors() == {"r": [edge]}
        assert [torch.equal(mgr.get_tensor(80 + i), src) for i, src in enumerate(srcs)] == [True, True]
        assert len(engine.submits) == 2 and engine.freed == set(engine.batches)
        assert mgr._kept_reads == [] and engine.violations == []
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_a_failed_tensor_is_retried_as_often_as_the_sync_read_retries_its_batch():
    engine = _AsyncEngine(hcas=["mlx5_0"])
    mgr = _draining_manager(engine)
    src = torch.tensor([3])
    try:
        (future,) = mgr.start_read_tensors("r", [_remote_edge(85, src, "flaky")])
        future.result(timeout=5)
        assert len(engine.submits) == 2 and torch.equal(mgr.get_tensor(85), src)

        engine_no_hca = _AsyncEngine()
        mgr2 = _draining_manager(engine_no_hca)
        (failed,) = mgr2.start_read_tensors("r", [_remote_edge(86, src, "flaky")])
        assert "Status: -1" in str(failed.exception(timeout=5))
        assert len(engine_no_hca.submits) == 1
        mgr2._async_reader._executor.shutdown(wait=True)
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_a_kept_read_is_freed_once_its_transfer_stops(caplog):
    """One tensor failed, the other was still running when the read gave up.
    The kept read waits for the running one, then is unregistered, freed and
    uncounted, and the process is back within its budget."""
    engine = _AsyncEngine()
    mgr = _draining_manager(engine)
    mgr.quarantine_budget_bytes = 10
    # like the worker's, which only knows live requests
    live = {"r": "chat-r"}
    mgr.rid_to_str = lambda rid: str(live.get(rid))
    edge = GraphEdge(next_node="node", name="x", tensor_info=[
        _pointer(90, torch.tensor([1]), "fails"), _pointer(91, torch.tensor([2]), "gated"),
    ])
    try:
        (future,) = mgr.start_read_tensors("r", [edge])
        ptrs = [mgr.get_tensor(90).data_ptr(), mgr.get_tensor(91).data_ptr()]
        error = future.exception(timeout=5)
        assert "Status: -1" in str(error) and set(error.in_flight) == {ptrs[1]}
        mgr._next_drain_check = float("inf")
        assert mgr.get_ready_tensors() == {}
        assert [f.request_id for f in mgr.take_failed_reads()] == ["r"]
        assert mgr.read_quarantine() == ReadQuarantine("receiver", 1, 16, 10)
        assert mgr.quarantine_over_budget()
        live.clear()

        mgr.drain_kept_reads()
        assert len(mgr._kept_reads) == 1 and engine.unregistered == []
        changes = mgr.quarantine_changes

        engine.gate.set()
        with caplog.at_level("INFO", logger="mstar.communication.tensors"):
            mgr._next_drain_check = 0.0
            mgr.get_ready_tensors()
        assert mgr._kept_reads == [] and sorted(engine.unregistered) == sorted(ptrs)
        assert mgr.read_quarantine() == ReadQuarantine("receiver", 0, 0, 10, drained_reads=1)
        assert mgr.quarantine_changes > changes and not mgr.quarantine_over_budget()
        assert "kept for request chat-r: the transfer has stopped" in caplog.text
        assert "back within its read buffer budget" in caplog.text
        assert engine.violations == []

        mgr.force_cleanup_request("r")
        assert sorted(engine.unregistered) == sorted(ptrs)
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_the_drain_check_runs_about_once_a_second():
    engine = _AsyncEngine()
    mgr = _draining_manager(engine)
    try:
        (future,) = mgr.start_read_tensors("r", [_remote_edge(95, torch.tensor([1]), "stuck")])
        future.exception(timeout=5)
        mgr.get_ready_tensors()
        (batch_id,) = future.exception().in_flight.values()
        main = threading.current_thread().name
        for _ in range(50):
            mgr.get_ready_tensors()
        assert _polls_from(engine, main) == [batch_id]
        mgr._next_drain_check -= 1.0
        mgr.get_ready_tensors()
        assert _polls_from(engine, main) == [batch_id, batch_id]
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_a_transfer_that_never_reports_is_parked_at_the_cap_and_logged_once(caplog):
    engine = _AsyncEngine()
    mgr = _draining_manager(engine)
    mgr.drain_cap_s = 600.0
    try:
        (future,) = mgr.start_read_tensors("r", [_remote_edge(96, torch.tensor([1, 2]), "stuck")])
        assert "timed out" in str(future.exception(timeout=5))
        mgr._next_drain_check = float("inf")
        mgr.get_ready_tensors()
        (kept,) = mgr._kept_reads
        main = threading.current_thread().name

        mgr.drain_kept_reads(kept.kept_at + 599)
        assert not kept.parked and len(_polls_from(engine, main)) == 1
        with caplog.at_level("WARNING", logger="mstar.communication.tensors"):
            for dt in (600, 601, 5000):
                mgr.drain_kept_reads(kept.kept_at + dt)
        assert kept.parked and len(_polls_from(engine, main)) == 2
        parked = [r for r in caplog.records if "Parking" in r.getMessage()]
        assert len(parked) == 1 and "has not reported after 600 s" in parked[0].getMessage()
        assert mgr.read_quarantine() == ReadQuarantine(
            "receiver", 1, 16, mgr.quarantine_budget_bytes, parked_reads=1,
        )
        assert engine.unregistered == [] and engine.violations == []
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_a_read_whose_submit_failed_is_parked_at_once(caplog):
    """A failed submit may have posted part of its transfer and leaves no id
    to ask, so nothing can say it stopped."""
    engine = _AsyncEngine()
    mgr = _draining_manager(engine)
    try:
        (future,) = mgr.start_read_tensors("r", [_remote_edge(97, torch.tensor([1]), "no-submit")])
        assert "submit failed" in str(future.exception(timeout=5))
        with caplog.at_level("WARNING", logger="mstar.communication.tensors"):
            mgr.get_ready_tensors()
            mgr.drain_kept_reads()
        (kept,) = mgr._kept_reads
        assert kept.parked and mgr.parked_reads == 1
        assert sum("Parking" in r.getMessage() for r in caplog.records) == 1
        assert engine.polls == [] and engine.unregistered == []
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


@pytest.mark.parametrize("ending", ["completes", "times_out"])
def test_a_drain_racing_teardown_waits_for_the_reader(ending):
    """Torn down while its read is still running: the drain check leaves the
    batch to the reader until the read ends, then frees the buffers once the
    transfer has stopped, and never polls an id the reader still polls. The
    freed buffer is released, not only uncounted, although the reader still
    holds the read's future."""
    engine = _AsyncEngine()
    mgr = _draining_manager(engine, timeout_s=0.3 if ending == "times_out" else 30)
    try:
        (future,) = mgr.start_read_tensors("r", [_remote_edge(98, torch.tensor([4]), "gated")])
        buffer = mgr.get_tensor(98)
        ptr, buffer_ref = buffer.data_ptr(), weakref.ref(buffer)
        del buffer
        mgr.force_cleanup_request("r")
        (kept,) = mgr._kept_reads
        assert kept.future is future and ptr not in engine.unregistered

        main = threading.current_thread().name
        for _ in range(3):
            mgr.drain_kept_reads()
        assert _polls_from(engine, main) == [] and len(mgr._kept_reads) == 1

        if ending == "completes":
            engine.gate.set()
            future.result(timeout=5)
        else:
            assert "timed out" in str(future.exception(timeout=5))
            mgr.drain_kept_reads()
            # the reader handed its running batch over; it is polled here now
            assert len(mgr._kept_reads) == 1 and kept.in_flight
            assert len(_polls_from(engine, main)) == 1
            engine.gate.set()
        mgr.drain_kept_reads()
        assert mgr._kept_reads == [] and engine.unregistered == [ptr]
        assert mgr.read_quarantine().drained_reads == 1
        assert engine.violations == []
        del kept
        gc.collect()
        assert buffer_ref() is None
    finally:
        engine.gate.set()
        mgr._async_reader._executor.shutdown(wait=True)


class _AsyncCallsRecorded(_Engine):
    def __init__(self):
        super().__init__()
        self.async_calls = []

    def batch_transfer_async_read(self, *args):
        self.async_calls.append("submit")
        return 0

    def transfer_check_status(self, batch_id):
        self.async_calls.append("status")
        return 0


def _real_manager(monkeypatch, engine):
    from mstar.communication import tensors

    class _Transport:
        def __init__(self, **_):
            self._engine = engine

        def register_memory(self, ptr, nbytes):
            return engine.register_memory(ptr, nbytes)

        def unregister_memory(self, ptr):
            return engine.unregister_memory(ptr)

        def get_session_id(self):
            return "receiver"

    monkeypatch.setattr(tensors, "MooncakeTransferEngine", _Transport)
    return MooncakeCommunicationManager(
        "receiver", "host", "cpu", _Communicator(), protocol=CommProtocol.TCP,
    )


def test_with_the_drain_off_reads_and_kept_buffers_are_as_before(monkeypatch, caplog):
    monkeypatch.delenv(READ_ASYNC_DRAIN_ENV, raising=False)
    engine = _AsyncCallsRecorded()
    mgr = _real_manager(monkeypatch, engine)
    try:
        assert not mgr._async_reader.async_drain and mgr._transfer_status is None
        (future,) = mgr.start_read_tensors("r", [_remote_edge(99, torch.tensor([1, 2]), "timed-out")])
        buffer = mgr.get_tensor(99)
        future.exception(timeout=5)
        with caplog.at_level("INFO", logger="mstar.communication.tensors"):
            for _ in range(3):
                mgr._next_drain_check = 0.0
                mgr.get_ready_tensors()
            mgr.force_cleanup_request("r")
            mgr.drain_kept_reads()
        assert engine.reads == [[buffer.data_ptr()]] and engine.async_calls == []
        assert any(t is buffer for t in _kept_buffers(mgr)) and engine.unregistered == []
        assert mgr.read_quarantine() == ReadQuarantine("receiver", 1, 16, mgr.quarantine_budget_bytes)
        assert [r.getMessage().split(":")[0] for r in caplog.records] == [
            "Tensor read for request r failed", "Keeping 16 bytes of read buffers for request r",
        ]
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_with_the_drain_on_the_manager_polls_through_the_reader(monkeypatch):
    monkeypatch.setenv(READ_ASYNC_DRAIN_ENV, "1")
    monkeypatch.setenv(READ_DRAIN_CAP_ENV, "30")
    engine = _AsyncEngine()
    mgr = _real_manager(monkeypatch, engine)
    try:
        assert mgr._async_reader.async_drain and mgr.drain_cap_s == 30.0
        assert mgr._transfer_status == mgr._async_reader.transfer_status
        (future,) = mgr.start_read_tensors("r", [_remote_edge(100, torch.tensor([1]), "peer")])
        future.result(timeout=5)
        assert len(engine.submits) == 1 and engine.reads == []
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_drain_settings(monkeypatch):
    monkeypatch.delenv(READ_ASYNC_DRAIN_ENV, raising=False)
    monkeypatch.delenv(READ_DRAIN_CAP_ENV, raising=False)
    assert read_async_drain_enabled() is False and read_drain_cap_seconds() == 600.0
    for on in ("1", "true", "ON"):
        monkeypatch.setenv(READ_ASYNC_DRAIN_ENV, on)
        assert read_async_drain_enabled() is True
    for off in ("0", "", "off"):
        monkeypatch.setenv(READ_ASYNC_DRAIN_ENV, off)
        assert read_async_drain_enabled() is False
    monkeypatch.setenv(READ_ASYNC_DRAIN_ENV, "maybe")
    with pytest.raises(ValueError, match=READ_ASYNC_DRAIN_ENV):
        read_async_drain_enabled()
    monkeypatch.setenv(READ_DRAIN_CAP_ENV, "2.5")
    assert read_drain_cap_seconds() == 2.5
    for bad in ("soon", "-1", "nan", "inf"):
        monkeypatch.setenv(READ_DRAIN_CAP_ENV, bad)
        with pytest.raises(ValueError, match=READ_DRAIN_CAP_ENV):
            read_drain_cap_seconds()


def test_log_stats_report_shows_drained_and_parked_reads(tmp_path):
    from mstar.profile.display import pretty_print_profile
    from mstar.profile.format import RequestProfile

    out = tmp_path / "stats.txt"
    pretty_print_profile(RequestProfile(rid="r", read_quarantine=[
        ReadQuarantine("worker_0", 1, 1024, 1 << 30, drained_reads=7, parked_reads=1),
    ]), str(out))
    assert "1.0 KiB  (x1)  budget 1.0 GiB  drained 7  parked 1\n" in out.read_text()
