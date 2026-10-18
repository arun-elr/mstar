"""A tensor the transfer engine refuses to register for send fails only its
own request, and nothing of that request is sent. Before, the refusal raised
out of ``_register_outputs`` and the worker's main-loop handler failed every
request in the batch (#488)."""

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    AsyncMooncakeReader,
    MooncakeCommunicationManager,
    TensorCommunicationManager,
)
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.runtime.base import SendInput
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.utils.containers import ParallelList
from mstar.worker.worker import Worker
from test.modular.test_graph_runtime import (
    _build_on_real_manager,
    _complete,
    _fwd_info,
    _ingest,
    _RecordingCommunicator,
)


class _Communicator(BaseCommunicator):
    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []


class _Engine:
    """Registers like a transfer engine, but refuses the given pointers the
    way Mooncake refuses an overlapping region (-7)."""

    def __init__(self, refuse=()):
        self.refuse = set(refuse)
        self.registered = []
        self.unregistered = []

    def register_memory(self, ptr, nbytes):
        if ptr in self.refuse:
            return -7
        self.registered.append(ptr)
        return 0

    def unregister_memory(self, ptr):
        self.unregistered.append(ptr)
        return 0


def _manager(engine):
    mgr = MooncakeCommunicationManager.__new__(MooncakeCommunicationManager)
    TensorCommunicationManager.__init__(mgr, "w0", "w0", "cpu", _Communicator(), engine)
    mgr.protocol = CommProtocol.TCP
    mgr._async_reader = AsyncMooncakeReader(engine, "cpu")
    return mgr


def _store(mgr, rid, *tensors):
    infos = mgr.store_and_return_tensor_info(rid, {"x": list(tensors)})
    return [info.uuid for info in infos["x"]]


def test_a_refused_registration_fails_only_its_request():
    a, bad, c, other = (torch.randn(4) for _ in range(4))
    engine = _Engine(refuse={bad.data_ptr()})
    mgr = _manager(engine)
    try:
        r0 = _store(mgr, "r0", a, bad, c)
        r1 = _store(mgr, "r1", other)

        failed = mgr.register_for_send_uuids(ParallelList(["r0", "r1"], [r0, r1]))

        assert list(failed) == ["r0"]
        assert "request id r0" in failed["r0"] and f"uuid {r0[1]}" in failed["r0"]
        assert [mgr.tensor_store.is_registered(u) for u in r0] == [True, False, False]
        assert mgr.tensor_store.is_registered(r1[0])
        assert engine.registered == [a.data_ptr(), other.data_ptr()]

        # cleanup unregisters only what did register
        mgr.force_cleanup_request("r0")
        assert engine.unregistered == [a.data_ptr()]
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_no_refusal_reports_nothing():
    mgr = _manager(_Engine())
    try:
        uuids = _store(mgr, "r0", torch.randn(4))
        assert mgr.register_for_send_uuids(ParallelList(["r0"], [uuids])) is None
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_the_worker_fails_only_the_refused_request():
    mine, bad, theirs = (torch.randn(4) for _ in range(3))
    mgr = _manager(_Engine(refuse={bad.data_ptr()}))
    try:
        r7 = _store(mgr, 7, mine, bad)
        r8 = _store(mgr, 8, theirs)
        worker = Worker.__new__(Worker)
        worker.tensor_manager = mgr
        reported = []
        worker._fail_requests = reported.append
        route_output = type("RouteOutput", (), {
            "register_uuids": r7 + r8, "register_rids": [7, 7, 8],
        })()

        assert worker._register_outputs(route_output) == {7}

        (errors,) = reported
        assert list(errors) == [7]
        assert errors[7].startswith("Error in worker: Mooncake memory registration failed")
        assert mgr.tensor_store.is_registered(r8[0])
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


@pytest.mark.parametrize("failed", [False, True])
def test_the_python_runtime_sends_nothing_for_a_failed_rid(tmp_path, failed):
    section = GraphNode(
        name="gen", input_names={"tok"},
        outputs=[GraphEdge(name="out", next_node=EMIT_TO_CLIENT)],
    )
    tm, runtime, rid = _build_on_real_manager(section, {"gen"}, tmp_path)
    comm = _RecordingCommunicator()
    runtime._communicator = comm
    _ingest(runtime, rid, [GraphEdge(name="tok", next_node="gen")])
    _, out = _complete(tm, runtime, rid, "gen", {"out": [torch.ones(1)]}, ["out"])

    sent_rids = runtime.send_outputs(SendInput(
        completion_id=out.completion_id,
        per_request_info=ParallelList([rid], [_fwd_info("w")]),
        new_token_counts=ParallelList([], []),
        failed_rids=frozenset({rid}) if failed else frozenset(),
    ))

    sent = sorted(entity for entity, _ in comm.sent)
    if failed:
        assert sent == [] and sent_rids == []
    else:
        assert sent == ["api_server", "conductor"] and sent_rids == [rid]


def test_an_overlapping_view_fails_only_its_request_on_real_mooncake():
    """The issue's repro on Mooncake over TCP: a tensor and a view inside it
    overlap, which Mooncake refuses (-7)."""
    pytest.importorskip("mooncake.engine")
    from mstar.communication.tensors import create_tensor_communication_manager

    mgr = create_tensor_communication_manager(
        protocol=CommProtocol.TCP, my_entity_id="w0", hostname="localhost",
        device="cpu", communicator=_Communicator(), tcp_transfer_device="lo",
    )
    h = torch.randn(4, 64)
    r0 = _store(mgr, "r0", h, h[-1])
    r1 = _store(mgr, "r1", torch.randn(4, 64))

    failed = mgr.register_for_send_uuids(ParallelList(["r0", "r1"], [r0, r1]))

    assert list(failed) == ["r0"]
    assert mgr.tensor_store.is_registered(r1[0])
    mgr.force_cleanup_request("r0")
    mgr.force_cleanup_request("r1")
