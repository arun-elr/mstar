"""A failed input read on one follower of a lockstep (TP/SP) instance fails
only its own request (#564).

The leader schedules from its own copy of a request's inputs, so it can
broadcast a step carrying a request whose read then fails on a follower. The
follower runs that step anyway, on zeros of the advertised shape, and tells the
leader directly, which stops scheduling the request. The step's collective
completes, both ranks drain and ACK, and the other rows of the batch carry on.

Two ranks of a TP=2 instance, each with the real ``MicroScheduler`` and the
real ``Worker`` readiness, failure and drain code; the follower reads through
a real ``MooncakeCommunicationManager`` over a fake transfer engine. Messages
between the ranks are delivered by hand, through the wire codec.
"""

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "test/modular")
from test_failed_tensor_read import _Engine as _TransferEngine  # noqa: E402
from test_failed_tensor_read import _mooncake_manager  # noqa: E402
from test_micro_scheduler import NODE, WALK, _Engine, _Manager  # noqa: E402

import mstar.communication.wire_types  # noqa: E402, F401  (registers the tags)
from mstar.communication.tensors import StoredOutputs  # noqa: E402
from mstar.communication.wire import decode, encode  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.graph.base import GraphEdge, TensorPointerInfo  # noqa: E402
from mstar.graph.runtime.base import FreedTensors, RouteOutput  # noqa: E402
from mstar.model.submodule_base import BatchedModelOutput  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    ConductorMessageType,
    DrainRequest,
    FailRequests,
    OffloadDelta,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

WORKERS = ["w0", "w1"]



def _node_batch(rids):
    """The executing batch the leader broadcasts: its rows and step context."""
    return SimpleNamespace(
        node_name=NODE, graph_walk=WALK, request_ids=rids,
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
    )

def _rank(worker_id, manager, tensor_manager):
    """Instance rank ``WORKERS.index(worker_id)`` of a TP=2 instance of NODE."""
    leader = worker_id == WORKERS[0]
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda n: _Engine()),
        parallel_leader_nodes={NODE} if leader else set(),
    )
    sched.runtime = manager.runtime
    w = Worker.__new__(Worker)
    w.worker_id, w.scheduler = worker_id, sched
    w.parallel_nodes = {NODE}
    w.parallel_leader_nodes = {NODE} if leader else set()
    w.is_tp_follower = not leader
    w.tp_async_sched, w.tp_async_nodes = False, None
    w._tp_broadcast_seq = 0
    w.sent = []
    w.communicator = SimpleNamespace(
        send=lambda dest, msg=None: w.sent.append((dest, decode(encode(msg)))),
    )
    group = SimpleNamespace(
        tp_size=2, _tp_rank=WORKERS.index(worker_id), _workers=WORKERS,
    )
    sharding = SimpleNamespace(groups=[group], get_sharding_group=lambda n, g: group)

    def ingest_inputs_batch(block, can_buffer):
        for rid in block.rids:
            manager.ingest(rid)

    w._graph_runtime = SimpleNamespace(
        get_rid_handle=lambda r: r if r in manager.per_request_info else None,
        get_rid_string=lambda h: h,
        get_sharding_config=lambda r: sharding,
        ingest_inputs_batch=ingest_inputs_batch,
    )
    w.request_state = manager
    w.tensor_manager = tensor_manager
    w.engine_manager = SimpleNamespace(get_engine=lambda n: SimpleNamespace(
        take_resident_delta=lambda n: OffloadDelta.new(),
    ))
    w.wakeup_event = SimpleNamespace(drain=lambda: None)
    w.enable_nvtx = False
    w._reported_quarantine_reads = 0
    w._unprocessed_messages = {}
    w._in_flight_rids, w._pending_drains = set(), set()
    w._draining_rids, w._reads_done_sent = set(), set()
    return w


def _pair():
    """Leader w0 holds both inputs; follower w1 holds r0's and is reading r1's,
    a read that will fail."""
    leader_mgr = _Manager(["r0", "r1"])
    leader = _rank("w0", leader_mgr, SimpleNamespace(has_inflight_reads=lambda r: False))

    follower_mgr = _Manager(["r0", "r1"])
    follower_mgr.queues["wg0"]._ready = {"r0": {NODE}}
    transfer = _TransferEngine()
    tm = _mooncake_manager(transfer)
    follower = _rank("w1", follower_mgr, tm)
    tm.placeholder_nodes = frozenset(follower.parallel_nodes - follower.parallel_leader_nodes)
    src = torch.ones(3, 4, dtype=torch.bfloat16)
    info = TensorPointerInfo(
        list(src.shape), src.dtype, src.nbytes, src.data_ptr(),
        list(src.stride()), 91, "timed-out", "producer",
    )
    (read,) = tm.start_read_tensors(
        "r1", [GraphEdge(next_node=NODE, name="x", tensor_info=[info])],
    )
    return leader, leader_mgr, follower, follower_mgr, read


def _taken(w, dest, message_type):
    """Messages ``w`` sent to ``dest`` of ``message_type``, removed from its outbox."""
    hits = [m for d, m in w.sent if d == dest and m.message_type == message_type]
    w.sent = [(d, m) for d, m in w.sent if m not in hits]
    return hits


def _deliver(src, dest, message_type):
    msgs = _taken(src, dest.worker_id, message_type)
    dest._process_message_list(msgs)
    return msgs


def _shutdown(follower):
    follower.tensor_manager._async_reader._executor.shutdown(wait=True)


def test_a_follower_runs_the_leaders_step_on_a_placeholder_and_only_that_request_fails():
    leader, leader_mgr, follower, follower_mgr, read = _pair()
    try:
        # The leader's read for r1 landed: it schedules [r0, r1], broadcasts,
        # and waits in the step's collective.
        batch = leader.scheduler.get_next_batch(leader_mgr)
        assert list(batch.request_to_worker_graph) == ["r0", "r1"]
        leader._in_flight_rids.update(batch.request_to_worker_graph)
        leader.maybe_send_zmq_to_tp_followers(_node_batch(list(batch.request_to_worker_graph)))
        _deliver(leader, follower, WorkerMessageType.SCHEDULE_TP)
        assert follower.scheduler.get_next_batch(follower_mgr) is None  # r1 not here yet

        # The follower's read for r1 fails.
        read.exception(timeout=5)
        follower._check_ready_tensors()

        # It runs the leader's step, both rows, r1 on zeros of the advertised shape.
        step = follower.scheduler.get_next_batch(follower_mgr)
        assert step is not None and step.tp_seq == 0
        assert list(step.request_to_worker_graph) == ["r0", "r1"]
        placeholder = follower.tensor_manager.get_tensor(91)
        assert placeholder.shape == (3, 4) and placeholder.dtype == torch.bfloat16
        assert not placeholder.any()
        assert dict(follower.scheduler.pending_tp_follow_count) == {}

        # It reported r1 to the conductor and, directly, to the leader.
        (fail,) = _taken(follower, "conductor", ConductorMessageType.FAIL_REQUESTS)
        assert list(fail.body.errors) == ["r1"]
        (told,) = _deliver(follower, leader, WorkerMessageType.TP_FOLLOWER_FAILED)
        assert list(told.body.errors) == ["r1"]
        assert leader.scheduler.failed_rids == {"r1"}
        (leader_fail,) = _taken(leader, "conductor", ConductorMessageType.FAIL_REQUESTS)
        assert list(leader_fail.body.errors) == ["r1"]

        # The conductor drains r1 on both ranks while the step is in flight;
        # the follower waits for the leader's forward, the leader for its step.
        follower._drain_request(DrainRequest("r1"))
        leader._drain_request(DrainRequest("r1"))
        assert leader._pending_drains == {"r1"}

        # The step's collective completes on both ranks.
        leader._in_flight_rids.clear()
        leader._apply_pending_drains(set())
        _deliver(leader, follower, WorkerMessageType.DRAIN_REQUEST)
        acks = [
            m.body.entity_id
            for w in (leader, follower)
            for m in _taken(w, "conductor", ConductorMessageType.READS_DONE)
        ]
        assert acks == ["w0", "w1"]
        assert dict(follower.scheduler.pending_tp_follow_count) == {}

        # r0 was never failed, and the leader schedules it alone from here on.
        assert "r0" not in leader.scheduler.failed_rids
        assert "r0" not in follower.scheduler.failed_rids
        leader_mgr.ingest("r0")
        leader_mgr.ingest("r1")
        nxt = leader.scheduler.get_next_batch(leader_mgr)
        assert list(nxt.request_to_worker_graph) == ["r0"]
        assert not [
            m for w in (leader, follower) for _, m in w.sent
            if isinstance(m.body, FailRequests)
        ]
    finally:
        _shutdown(follower)


def test_a_leader_told_before_it_schedules_never_broadcasts_the_request():
    leader, leader_mgr, follower, follower_mgr, read = _pair()
    try:
        read.exception(timeout=5)
        follower._check_ready_tensors()
        _deliver(follower, leader, WorkerMessageType.TP_FOLLOWER_FAILED)

        batch = leader.scheduler.get_next_batch(leader_mgr)
        assert list(batch.request_to_worker_graph) == ["r0"]
        leader.maybe_send_zmq_to_tp_followers(_node_batch(["r0"]))
        _deliver(leader, follower, WorkerMessageType.SCHEDULE_TP)
        step = follower.scheduler.get_next_batch(follower_mgr)
        assert list(step.request_to_worker_graph) == ["r0"]

        leader._drain_request(DrainRequest("r1"))
        _deliver(leader, follower, WorkerMessageType.DRAIN_REQUEST)
        acks = [
            m.body.entity_id
            for w in (leader, follower)
            for m in _taken(w, "conductor", ConductorMessageType.READS_DONE)
        ]
        assert acks == ["w0", "w1"]
    finally:
        _shutdown(follower)


def _postprocess_fakes(w):
    """Fake the engine, routing and sends around ``w._postprocess_batch``;
    returns the log of the rids it routed and sent."""
    log = SimpleNamespace(routed=[], sent=[])
    w._phase_period = 0
    w.enable_prof = False
    w.device = torch.device("cpu")
    w._last_active, w._innermost_loop_by_node = {}, {}
    w._prematerialize_for_check_stop = (
        lambda outputs, ev, request_ids=None: (outputs.per_rid_outputs, None)
    )
    w._inline_emit_values = lambda *a: (None, None)
    w._count_new_tokens = lambda *a: {}
    w._stream_consumption_batch = lambda rids: []
    w.request_state.get_fwd_info = lambda rid, p: None
    w.request_state.get_pending_publish_info = lambda rid, p: None
    engine = SimpleNamespace(
        check_stop_for_batch=lambda *a, **k: {},
        extend_prefix_chains=lambda *a, **k: None,
        materialize_publish=lambda *a: None,
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda n: engine)
    w.tensor_manager.cleanup_collectable = lambda *a: None
    w.tensor_manager.store_and_return_tensor_info_batch = (
        lambda rids, *a, **k: StoredOutputs(
            list(range(len(rids))), list(rids), [0] * len(rids), [1] * len(rids),
        )
    )
    w.tensor_manager.increment_ref_batch_uniform = lambda *a: None
    w.tensor_manager.dereference_batch_uniform = lambda *a: None
    w.tensor_manager.register_for_send_uuids = lambda *a, **k: None

    def route(route_input):
        log.routed.append(list(route_input.wg_ids.keys))
        return RouteOutput(
            completion_id=0, register_uuids=[], register_rids=[],
            new_token_output_idxs=[], local_streaming_by_signal={},
        )

    def send(send_input):
        log.sent.append(list(send_input.per_request_info.keys))
        return []

    w._graph_runtime.cleanup_consumed_inputs = lambda *a: FreedTensors.none()
    w._graph_runtime.clear_pending_loop_stops = lambda: None
    w._graph_runtime.complete_and_route_batch = route
    w._graph_runtime.update_dynamic_loop_iters = lambda *a: None
    w._graph_runtime.send_outputs = send
    w._graph_runtime.supports_inline_emit = False
    return log


def _finished_step(rids):
    """The step [rids] of NODE as it comes back from the forward."""
    pending = SimpleNamespace(
        batch=ScheduledBatch(
            node_name=NODE, graph_walk=WALK,
            request_to_worker_graph=dict.fromkeys(rids, "wg0"), output_signals=["y"],
        ),
        node_batch=SimpleNamespace(
            request_ids=list(rids), per_request_info=dict.fromkeys(rids),
            failed_requests={}, completion_event=None, final_stream_rids=set(),
            resource_publish_info={}, stream_partition_done_rids=set(),
            cached_rids=set(),
        ),
        node_name=NODE, partition="p", graph_walk=WALK, loop_name=None,
        speculative_new_iter=False,
    )
    outputs = BatchedModelOutput(per_rid_outputs={r: {"y": [torch.zeros(1)]} for r in rids})
    return pending, outputs


def test_the_leader_drops_the_placeholder_row_once_told():
    """The zero row's outputs are meaningless. Once the follower's report has
    failed the request on the leader, the leader routes and sends nothing of
    it, for a step already in flight too; the follower still routes its row,
    since rank 0 decides what runs there."""
    leader, leader_mgr, follower, follower_mgr, read = _pair()
    try:
        batch = leader.scheduler.get_next_batch(leader_mgr)
        leader.maybe_send_zmq_to_tp_followers(_node_batch(list(batch.request_to_worker_graph)))
        _deliver(leader, follower, WorkerMessageType.SCHEDULE_TP)
        read.exception(timeout=5)
        follower._check_ready_tensors()
        step = follower.scheduler.get_next_batch(follower_mgr)
        _deliver(follower, leader, WorkerMessageType.TP_FOLLOWER_FAILED)
        assert leader.scheduler.failed_rids == {"r1"}

        leader_log, follower_log = _postprocess_fakes(leader), _postprocess_fakes(follower)
        Worker._postprocess_batch(leader, *_finished_step(batch.request_to_worker_graph))
        Worker._postprocess_batch(follower, *_finished_step(step.request_to_worker_graph))

        assert leader_log.routed == [["r0"]] and leader_log.sent == [["r0"]]
        assert follower_log.routed == [["r0", "r1"]]
    finally:
        _shutdown(follower)


def test_the_leader_ignores_a_report_for_a_request_it_no_longer_serves():
    leader, _, follower, _, _ = _pair()
    try:
        leader.scheduler.fail_rids({"r1"})
        leader._process_message_list([WorkerMessage(
            message_type=WorkerMessageType.TP_FOLLOWER_FAILED,
            body=FailRequests(errors={"r1": "failed", "gone": "failed"}),
        )])
        assert leader.sent == []
    finally:
        _shutdown(follower)
