"""A tensor the transfer engine refuses to register for send fails only its
own request, and nothing of that request is sent. Before, the refusal raised
out of ``_register_outputs`` and the worker's main-loop handler failed every
request in the batch (#488)."""

import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

from test_graph_runtime import (
    _build_on_real_manager,
    _complete,
    _fwd_info,
    _ingest,
    _RecordingCommunicator,
)

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    AsyncMooncakeReader,
    MooncakeCommunicationManager,
    StoredOutputs,
    TensorCommunicationManager,
)
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.runtime.base import FreedTensors, RouteOutput, SendInput
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import BatchedModelOutput
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import ConductorMessageType
from mstar.worker.worker import Worker


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
        worker.scheduler = SimpleNamespace(failed_rids=set())
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


def _postprocess_worker(refuse, follower=False):
    """A worker whose ``_postprocess_batch`` runs for real, with the engine,
    graph runtime and transfer engine faked around it. The transfer engine
    refuses every tensor of the rids in ``refuse``, step after step."""
    w = Worker.__new__(Worker)
    w._phase_period = 0
    w.enable_nvtx = w.enable_prof = False
    w.device = torch.device("cpu")
    w.worker_id = "w0"
    w.parallel_nodes = {"llm"} if follower else set()
    w.parallel_leader_nodes = set()
    w._last_active = {}
    w._innermost_loop_by_node = {}
    w.log = SimpleNamespace(stop_checked=[], routed=[], registered=[], sent=[], conductor=[])
    w.scheduler = SimpleNamespace(failed_rids=set())
    w.scheduler.fail_rids = w.scheduler.failed_rids.update
    w.communicator = SimpleNamespace(send=lambda e, m: w.log.conductor.append(m))
    w.request_state = SimpleNamespace(
        per_request_info={0: object(), 1: object()},
        get_fwd_info=lambda rid, p: None,
        get_pending_publish_info=lambda rid, p: None,
        flush_publish_info=lambda rid, p: None,
    )
    w._prematerialize_for_check_stop = (
        lambda outputs, ev, request_ids=None: (outputs.per_rid_outputs, None)
    )
    w._inline_emit_values = lambda *a: (None, None)
    w._count_new_tokens = lambda *a: {}
    w._stream_consumption_batch = lambda rids: []

    def check_stop(node_batch, cpu_outputs, host_rows=None):
        w.log.stop_checked.append(list(node_batch.request_ids))
        return {}

    engine = SimpleNamespace(
        check_stop_for_batch=check_stop,
        extend_prefix_chains=lambda *a, **k: None,
        materialize_publish=lambda *a: None,
        # like the engine's, reads each stopped rid's info
        finalize_stopped_requests=lambda nb, rids: [nb.per_request_info[r] for r in rids],
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda node: engine)

    def store(rids, outputs, signals, **kwargs):
        return StoredOutputs(
            flat_uuids=[100 + r for r in rids], flat_rids=list(rids),
            signal_idxs=[0] * len(rids), num_tensors=[1] * len(rids),
        )

    def register(per_request, skip_cuda_sync):
        w.log.registered.append(list(per_request.keys))
        return {r: "refused" for r in per_request.keys if r in refuse} or None

    w.tensor_manager = SimpleNamespace(
        cleanup_collectable=lambda *a: None,
        store_and_return_tensor_info_batch=store,
        increment_ref_batch_uniform=lambda *a: None,
        dereference_batch_uniform=lambda *a: None,
        register_for_send_uuids=register,
    )

    def route(route_input):
        rids = list(route_input.wg_ids.keys)
        w.log.routed.append(rids)
        return RouteOutput(
            completion_id=len(w.log.routed),
            register_uuids=list(route_input.tensors), register_rids=rids,
            new_token_output_idxs=[], local_streaming_by_signal={},
            freed_inputs=FreedTensors.none(),
        )

    def send(send_input):
        w.log.sent.append(
            [r for r in send_input.per_request_info.keys
             if r not in send_input.failed_rids]
        )
        return []

    w._graph_runtime = SimpleNamespace(
        cleanup_consumed_inputs=lambda *a: FreedTensors.none(),
        clear_pending_loop_stops=lambda: None,
        stop_loops_batched=lambda **kw: list(kw["loop_names"].keys),
        complete_and_route_batch=route,
        update_dynamic_loop_iters=lambda *a: None,
        send_outputs=send,
        get_rid_string=lambda rid: f"r{rid}",
        supports_inline_emit=False,
    )
    return w


def _decode_step(rids, final_stream=()):
    node_batch = SimpleNamespace(
        request_ids=list(rids), per_request_info=dict.fromkeys(rids),
        failed_requests={}, completion_event=None,
        final_stream_rids=set(final_stream), resource_publish_info={},
        stream_partition_done_rids=set(), cached_rids=set(),
    )
    batch = SimpleNamespace(
        node_name="llm", request_to_worker_graph=dict.fromkeys(rids, 0),
        output_signals=["tok"],
    )
    outputs = BatchedModelOutput(
        per_rid_outputs={r: {"tok": [torch.tensor([r])]} for r in rids},
    )
    pending = SimpleNamespace(
        batch=batch, node_batch=node_batch, node_name="llm",
        partition="p", graph_walk="decode", loop_name="decode",
        speculative_new_iter=False,
    )
    return pending, outputs


def _fail_requests_sent(w):
    return [
        m for m in w.log.conductor
        if m.message_type == ConductorMessageType.FAIL_REQUESTS
    ]


def test_the_speculated_next_step_sends_nothing_of_a_refused_request():
    """Async scheduling submits step N+1 before step N is post-processed, so
    a rid whose registration is refused in N is already in N+1. N+1 must
    neither route, register nor send it, nor report it a second time."""
    w = _postprocess_worker(refuse={0})
    step_n = _decode_step([0, 1])
    step_n1 = _decode_step([0, 1])  # speculated from N while N ran

    Worker._postprocess_batch(w, *step_n)
    assert w.log.sent == [[1]]
    assert w.scheduler.failed_rids == {0}

    Worker._postprocess_batch(w, *step_n1)

    assert w.log.stop_checked == [[0, 1], [1]]
    assert w.log.routed == [[0, 1], [1]]
    assert w.log.registered == [[0, 1], [1]]
    assert w.log.sent == [[1], [1]]
    assert len(_fail_requests_sent(w)) == 1


def test_a_step_of_only_failed_rids_routes_nothing():
    w = _postprocess_worker(refuse=set())
    w.scheduler.failed_rids.add(0)
    pending, outputs = _decode_step([0])

    Worker._postprocess_batch(w, pending, outputs)

    assert w.log.routed == [] and w.log.sent == []
    assert pending.batch.request_to_worker_graph == {}
    assert pending.node_batch.request_ids == []


def test_a_failed_rid_that_took_its_last_stream_chunk_does_not_stop_its_loop():
    """N+1 consumed the failed rid's terminal stream chunk; ending its loop
    would finalize a rid already taken out of the batch, and the raise would
    fail the healthy rows too."""
    w = _postprocess_worker(refuse={0})
    w._innermost_loop_by_node = {("decode", "llm"): "decode"}
    Worker._postprocess_batch(w, *_decode_step([0, 1]))

    pending, outputs = _decode_step([0, 1], final_stream={0, 1})
    Worker._postprocess_batch(w, pending, outputs)

    assert w.log.routed == [[0, 1], [1]]
    assert w.log.sent == [[1], [1]]
    assert pending.node_batch.final_stream_rids == {1}


def test_a_follower_still_routes_a_failed_rid_and_reports_it_once():
    """Rank 0 decides what a follower runs and may still send a step for the
    rid, which needs this step's loop-back; the refused sends stay skipped."""
    w = _postprocess_worker(refuse={0}, follower=True)

    Worker._postprocess_batch(w, *_decode_step([0, 1]))
    Worker._postprocess_batch(w, *_decode_step([0, 1]))

    assert w.log.routed == [[0, 1], [0, 1]]
    assert w.log.sent == [[1], [1]]
    assert len(_fail_requests_sent(w)) == 1


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
