"""Small tensors sent inline over the Mooncake transports
(``MSTAR_INLINE_TENSOR_MAX_BYTES``): the producer attaches the bytes to the
descriptor instead of registering the tensor, and the consumer rebuilds it
without a read, keeping the references and the ack of a read.

Two real ``MooncakeCommunicationManager``s on CPU over TCP loopback, with the
descriptor going through the wire codec between them. Skips without Mooncake.
"""
from concurrent.futures import Future

import msgpack
import pytest
import torch

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensor_store import (
    PythonTensorBookkeeping,
    TensorStore,
)
from mstar.communication.tensors import (
    INLINE_STEP_MAX_BYTES_ENV,
    INLINE_TENSOR_MAX_BYTES_ENV,
    MOONCAKE_IMPORT_ERROR,
    TRANSPORT_STATS_ENV,
    create_tensor_communication_manager,
    inline_step_max_bytes,
    inline_tensor_max_bytes,
)
from mstar.communication.wire import decode, encode
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge, NodeAndGraphWalk, TensorPointerInfo
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import (
    InputSignals,
    TensorReceived,
    WorkerMessage,
    WorkerMessageType,
)

pytestmark = pytest.mark.skipif(
    MOONCAKE_IMPORT_ERROR is not None, reason="mooncake not installed",
)

try:
    from mstar.communication.tensor_store import RustTensorBookkeeping
    RustTensorBookkeeping()
    _BOOKKEEPERS = ["python", "rust"]
except ImportError:
    _BOOKKEEPERS = ["python"]

# TensorPointerInfo's wire keys on main, in order: a frame without inline
# bytes must be exactly today's.
_MAIN_DESCRIPTOR_KEYS = [
    "dims", "dtype", "nbytes", "address", "stride", "uuid", "source_session_id",
    "source_entity", "offset", "source_tp_size", "source_tp_rank", "shm_offset",
]


class _Comm(BaseCommunicator):
    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return []

    def acks(self):
        return [m.body for _, m in self.sent
                if getattr(m, "message_type", None) == WorkerMessageType.TENSOR_RECEIVED]


class _Spy:
    """Counts calls on the real engine method; ``fake`` skips Mooncake."""

    def __init__(self, real, fake=False):
        self.real, self.fake, self.calls = real, fake, []

    def __call__(self, *args):
        self.calls.append(args)
        return 0 if self.fake else self.real(*args)


def _manager(entity, monkeypatch, *, max_bytes=4096, step_max=None, bookkeeping="python",
             fake_engine=False, stats=False):
    monkeypatch.setenv(INLINE_TENSOR_MAX_BYTES_ENV, str(max_bytes))
    if step_max is None:
        monkeypatch.delenv(INLINE_STEP_MAX_BYTES_ENV, raising=False)
    else:
        monkeypatch.setenv(INLINE_STEP_MAX_BYTES_ENV, str(step_max))
    if stats:
        monkeypatch.setenv(TRANSPORT_STATS_ENV, "3600")
    else:
        monkeypatch.delenv(TRANSPORT_STATS_ENV, raising=False)
    mgr = create_tensor_communication_manager(
        protocol=CommProtocol.TCP, my_entity_id=entity, hostname="localhost",
        device="cpu", communicator=_Comm(), tcp_transfer_device="lo",
    )
    mgr.tensor_store = TensorStore(
        RustTensorBookkeeping() if bookkeeping == "rust" else PythonTensorBookkeeping()
    )
    engine = mgr.transfer_engine
    mgr.register_spy = engine.register_memory = _Spy(engine.register_memory, fake_engine)
    mgr.unregister_spy = engine.unregister_memory = _Spy(engine.unregister_memory, fake_engine)
    return mgr


def _send(prod, rid, signal, tensors):
    """Store, hold, register, and build the edge from the store's
    descriptors (as both runtimes do at send time), through the wire."""
    infos = prod.store_and_return_tensor_info(rid, {signal: tensors})[signal]
    uuids = [i.uuid for i in infos]
    prod.increment_ref_batch_uniform(uuids, 1)
    prod.register_for_send_uuids(ParallelList([rid], [uuids]))
    edge = GraphEdge(next_node="sink", name=signal,
                     tensor_info=[prod.tensor_store.get_info(u) for u in uuids])
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(
            request_id=rid, inputs=[edge],
            request_info=CurrentForwardPassInfo(
                request_id=rid, graph_walk="w", fwd_index=0, random_seed=0, max_tokens=1,
            ),
        ),
    )
    frame = encode(msg)
    return decode(frame).body.inputs, frame, uuids


def _receive(cons, rid, edges):
    for fut in cons.start_read_tensors(rid, edges) or []:
        fut.result()
    ready = cons.get_ready_tensors()
    return {e.name: [cons.get_tensor(i.uuid) for i in e.tensor_info] for e in ready.get(rid, [])}


def _settle_acks(prod, cons):
    for ack in cons.communicator.acks():
        prod.dereference_batch(list(ack.successful_tensors), list(ack.successful_tensors.values()))


SMALL = [
    torch.arange(6, dtype=torch.float32).reshape(2, 3),
    torch.tensor([1.5, -2.25, 3.0], dtype=torch.bfloat16),
    torch.tensor(7, dtype=torch.int64),  # 0-dim
    torch.tensor([True, False, True]),
    torch.randn(2, 2, 3).to(torch.float16),
]


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_small_tensors_round_trip_inline_without_mooncake(monkeypatch, bookkeeping):
    prod = _manager("worker_0", monkeypatch, bookkeeping=bookkeeping)
    cons = _manager("worker_1", monkeypatch, bookkeeping=bookkeeping)
    edges, _, uuids = _send(prod, "r", "states", [t.clone() for t in SMALL])
    assert all(i.inline_data is not None for i in edges[0].tensor_info)

    got = _receive(cons, "r", edges)["states"]
    for want, have in zip(SMALL, got, strict=True):
        assert have.dtype == want.dtype and have.shape == want.shape
        assert have.reshape(-1).view(torch.uint8).tolist() == want.reshape(-1).view(torch.uint8).tolist()
    assert prod.register_spy.calls == [] and cons.register_spy.calls == []

    # the ack of a read, then the producer frees them without unregistering
    acks = cons.communicator.acks()
    assert len(acks) == 1 and sorted(acks[0].successful_tensors) == sorted(uuids)
    _settle_acks(prod, cons)
    assert prod.tensor_store.get_all_uuids("r") == []
    assert prod.unregister_spy.calls == [] and not prod._inlined


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_one_edge_mixes_inline_and_read_tensors(monkeypatch, bookkeeping):
    prod = _manager("worker_0", monkeypatch, max_bytes=1024, bookkeeping=bookkeeping)
    cons = _manager("worker_1", monkeypatch, max_bytes=1024, bookkeeping=bookkeeping)
    small = torch.arange(16, dtype=torch.int64)          # 128 B
    large = torch.arange(1024, dtype=torch.float32)      # 4 KiB
    edges, _, (u_small, u_large) = _send(prod, "r", "x", [small, large])
    infos = {i.uuid: i for i in edges[0].tensor_info}
    assert infos[u_small].inline_data is not None and infos[u_large].inline_data is None
    assert [c[1] for c in prod.register_spy.calls] == [large.nbytes]

    got = _receive(cons, "r", edges)["x"]
    assert torch.equal(got[0], small) and torch.equal(got[1], large)
    assert [c[1] for c in cons.register_spy.calls] == [large.nbytes]
    _settle_acks(prod, cons)
    assert prod.tensor_store.get_all_uuids("r") == []
    assert len(prod.unregister_spy.calls) == 1


def test_threshold_boundary_and_what_is_never_inlined(monkeypatch):
    prod = _manager("worker_0", monkeypatch, max_bytes=1024, fake_engine=True)
    at = torch.zeros(256, dtype=torch.float32)            # exactly 1024 B
    over = torch.zeros(1025, dtype=torch.uint8)
    empty = torch.zeros(0, dtype=torch.float32)
    strided = torch.zeros(4, 64, dtype=torch.float32)[:, ::2]   # 512 B, not contiguous
    sharded = torch.zeros(8, dtype=torch.int64)
    infos = prod.store_and_return_tensor_info(
        "r", {"x": [at, over, empty, strided, sharded]})["x"]
    prod.uuid_to_shard_dim[infos[4].uuid] = 0
    uuids = [i.uuid for i in infos]
    prod.register_for_send_uuids(ParallelList(["r"], [uuids]))
    inline = [prod.tensor_store.get_info(u).inline_data is not None for u in uuids]
    # a sharded tensor qualifies: each TP slice carries its own byte range
    assert inline == [True, False, False, False, True]
    # the empty tensor stays off the inline path; Mooncake may skip pinning it
    assert sorted(c[1] for c in prod.register_spy.calls if c[1]) == sorted(
        [over.nbytes, strided.nbytes])


def test_the_step_cap_bounds_the_inline_bytes_of_one_call(monkeypatch):
    prod = _manager("worker_0", monkeypatch, max_bytes=4096, step_max=10_000, fake_engine=True)
    ts = [torch.zeros(4000, dtype=torch.uint8) for _ in range(3)] + [torch.zeros(1000, dtype=torch.uint8)]
    infos = prod.store_and_return_tensor_info("r", {"x": ts})["x"]
    uuids = [i.uuid for i in infos]
    prod.register_for_send_uuids(ParallelList(["r"], [uuids]))
    inline = [prod.tensor_store.get_info(u).inline_data is not None for u in uuids]
    # in order until the next one does not fit; a later one that fits still goes
    assert inline == [True, True, False, True]
    assert sum(ts[i].nbytes for i in range(4) if inline[i]) <= 10_000
    # a later call has its own budget, and an inlined uuid is not redone
    prod.register_for_send_uuids(ParallelList(["r"], [uuids]))
    assert len(prod.register_spy.calls) == 1


def test_a_later_inline_edge_waits_for_an_earlier_read_of_its_request(monkeypatch):
    prod = _manager("worker_0", monkeypatch, max_bytes=1024)
    cons = _manager("worker_1", monkeypatch, max_bytes=1024)
    first, _, _ = _send(prod, "r", "big", [torch.arange(1024, dtype=torch.float32)])
    second, _, _ = _send(prod, "r", "small", [torch.arange(4, dtype=torch.int64)])

    held = Future()
    real_submit = cons._async_reader.submit
    cons._async_reader.submit = lambda infos: held if infos else real_submit(infos)
    cons.start_read_tensors("r", first)
    cons.start_read_tensors("r", second)
    assert cons.get_ready_tensors() == {}, "the inline edge must not overtake the read"
    held.set_result(None)  # the read never ran; only its completion is under test
    ready = cons.get_ready_tensors()
    assert [e.name for e in ready["r"]] == ["big", "small"]


def test_a_descriptor_whose_bytes_do_not_match_is_refused(monkeypatch):
    cons = _manager("worker_1", monkeypatch)
    info = TensorPointerInfo(
        dims=[4], dtype=torch.int32, nbytes=16, address=0, stride=[1], uuid=(3 << 48) | 1,
        source_session_id="h:1", source_entity="worker_0", inline_data=b"\x00" * 12,
    )
    with pytest.raises(ValueError, match="12 inline bytes for a 16-byte"):
        cons.start_read_tensors("r", [GraphEdge(next_node="n", name="x", tensor_info=[info])])
    assert not cons.tensor_store.check_uuid_presence(info.uuid)


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_flag_off_sends_todays_frame_and_registers_as_before(monkeypatch, bookkeeping):
    assert inline_tensor_max_bytes() >= 0 and inline_step_max_bytes() > 0
    prod = _manager("worker_0", monkeypatch, max_bytes=0, bookkeeping=bookkeeping)
    small = torch.arange(4, dtype=torch.int64)
    edges, frame, (uuid,) = _send(prod, "r", "x", [small])
    assert edges[0].tensor_info[0].inline_data is None
    assert [c[1] for c in prod.register_spy.calls] == [small.nbytes]
    assert prod._d2h_stream is None and prod.transport_stats is None

    raw = msgpack.unpackb(frame, raw=False, strict_map_key=False)
    descriptor = raw[1]["body"][1]["inputs"][0]["tensor_info"][0]
    assert list(descriptor) == _MAIN_DESCRIPTOR_KEYS


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_the_consumer_keeps_the_bytes_for_a_later_re_send(monkeypatch, bookkeeping):
    """A descriptor received inline may be sent on (a persisted signal, a loop
    re-emitting its inputs); the producer never registered the tensor, so the
    bytes have to stay with it."""
    prod = _manager("worker_0", monkeypatch, bookkeeping=bookkeeping)
    cons = _manager("worker_1", monkeypatch, bookkeeping=bookkeeping)
    t = torch.arange(8, dtype=torch.int16)
    edges, _, (uuid,) = _send(prod, "r", "x", [t])
    _receive(cons, "r", edges)
    assert cons.tensor_store.get_info(uuid).inline_data == t.numpy().tobytes()


def test_refresh_gives_each_copy_its_own_range_of_the_bytes(monkeypatch):
    prod = _manager("worker_0", monkeypatch)
    infos = prod.store_and_return_tensor_info("r", {"x": [torch.arange(8, dtype=torch.int64)]})["x"]
    prod.register_for_send_uuids(ParallelList(["r"], [[infos[0].uuid]]))
    data = torch.arange(8, dtype=torch.int64).numpy().tobytes()
    whole = infos[0].clone()
    whole.inline_data = None
    half = whole.clone()
    half.nbytes, half.offset = 32, 32
    past_end = whole.clone()
    past_end.nbytes, past_end.offset = 32, 48
    edge = GraphEdge(next_node="n", name="x", tensor_info=[whole, half, past_end])
    prod.refresh_inline_data([edge])
    assert whole.inline_data == data
    assert half.inline_data == data[32:]
    assert past_end.inline_data is None


def _tp1_to_tp2_config() -> ShardingConfig:
    cfg = ShardingConfig(
        groups=[
            ShardingGroup(nodes=["A"], tp_size=1, graph_walks=["decode"], _tp_rank=0),
            ShardingGroup(nodes=["B"], tp_size=2, graph_walks=["decode"]),
        ],
        tp_enabled_nodes={"A", "B"},
        shard_dim={"x": 0},
    )
    cfg.setup({
        NodeAndGraphWalk("A", "decode"): ["worker_0"],
        NodeAndGraphWalk("B", "decode"): ["worker_1", "worker_2"],
    })
    return cfg


def _read_halves(monkeypatch, fanout, original, bookkeeping):
    for worker, expected in (("worker_1", original[:2]), ("worker_2", original[2:])):
        cons = _manager(worker, monkeypatch, bookkeeping=bookkeeping)
        (got,) = _receive(cons, 7, [fanout[worker]])["x"]
        assert torch.equal(got, expected), f"{worker} read the wrong rows"
        assert cons.register_spy.calls == [], f"{worker} fell back to a read"


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_an_api_input_fanned_out_to_tp2_carries_each_rank_its_rows(monkeypatch, bookkeeping):
    """The api server knows no sharding, so it inlines a whole input; the
    conductor's fanout then slices the descriptor per rank, bytes included."""
    api = _manager("api_server", monkeypatch, bookkeeping=bookkeeping)
    original = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    infos = api.store_and_return_tensor_info(7, {"x": [original]})["x"]
    api.register_for_send(7, infos)
    assert infos[0].inline_data is not None and api.register_spy.calls == []
    wire = decode(encode(GraphEdge(next_node="B", name="x", tensor_info=infos)))
    fanout = _tp1_to_tp2_config().fanout_graph_edges(
        wire, source_node="A", source_graph_walk="decode", dest_graph_walk="decode",
    )
    _read_halves(monkeypatch, fanout, original, bookkeeping)


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_a_tp_fanout_cloned_before_registration_is_refreshed_per_rank(monkeypatch, bookkeeping):
    """A worker's routing slices a sharded output (shard_dim 0) before the
    tensor is registered; the refresh at send time gives each slice its range."""
    prod = _manager("worker_0", monkeypatch, bookkeeping=bookkeeping)
    cfg = _tp1_to_tp2_config()
    prod.register_request(7, cfg)
    original = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    infos = prod.store_and_return_tensor_info(
        7, {"x": [original]}, node_name="A", graph_walk="decode",
    )["x"]
    edge = GraphEdge(next_node="B", name="x", tensor_info=infos)
    fanout = cfg.fanout_graph_edges(
        edge, source_node="A", source_graph_walk="decode", dest_graph_walk="decode",
    )
    prod.register_for_send_uuids(ParallelList([7], [[infos[0].uuid]]))
    assert prod.register_spy.calls == []
    for wire_edge in fanout.values():
        prod.refresh_inline_data([wire_edge])
    fanout = {w: decode(encode(e)) for w, e in fanout.items()}
    _read_halves(monkeypatch, fanout, original, bookkeeping)


def test_a_permuted_shard_and_a_fan_in_concat_stay_registered(monkeypatch):
    """Both are made by a kernel queued after the step's completion event,
    which the side-stream copy would not wait for."""
    prod = _manager("worker_0", monkeypatch, fake_engine=True)
    cfg = ShardingConfig(
        groups=[ShardingGroup(nodes=["A"], tp_size=1, graph_walks=["decode"], _tp_rank=0)],
        tp_enabled_nodes={"A"}, shard_dim={"x": 1, "y": 0},
    )
    cfg.setup({NodeAndGraphWalk("A", "decode"): ["worker_0"]})
    prod.register_request(7, cfg)
    infos = prod.store_and_return_tensor_info(
        7, {"x": [torch.zeros(2, 4)], "y": [torch.zeros(2, 4)]},
        node_name="A", graph_walk="decode",
    )
    uuids = [infos["x"][0].uuid, infos["y"][0].uuid]
    prod.register_for_send_uuids(ParallelList([7], [uuids]))
    assert [prod.tensor_store.get_info(u).inline_data is not None for u in uuids] == [False, True]

    cons = _manager("worker_1", monkeypatch, fake_engine=True)
    cons._no_inline.add(99)  # what get_ready_tensors does for a consolidated tensor
    cons.tensor_store.put_tensor(7, 99, torch.zeros(4), TensorPointerInfo(
        dims=[4], dtype=torch.float32, nbytes=16, address=0, stride=[1], uuid=99,
        source_session_id="h:1", source_entity="worker_1",
    ))
    cons.register_for_send_uuids(ParallelList([7], [[99]]))
    assert cons.tensor_store.get_info(99).inline_data is None
    assert len(cons.register_spy.calls) == 1


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_persisted_tensors_stay_registered_even_if_sent_inline_before(monkeypatch, bookkeeping):
    prod = _manager("worker_0", monkeypatch, bookkeeping=bookkeeping, fake_engine=True)
    infos = prod.store_and_return_tensor_info(
        "r", {"x": [torch.arange(4, dtype=torch.int64), torch.arange(2, dtype=torch.int64)]},
    )["x"]
    early, fresh = infos[0].uuid, infos[1].uuid
    prod.register_for_send_uuids(ParallelList(["r"], [[early]]))
    assert prod.tensor_store.get_info(early).inline_data is not None
    # a later pass persists both (worker._register_outputs)
    prod.keep_registered([early, fresh])
    prod.register_for_send_uuids(ParallelList(["r"], [[early, fresh]]))
    for uuid in (early, fresh):
        assert prod.tensor_store.get_info(uuid).inline_data is None
        assert prod.tensor_store.is_registered(uuid)
    assert len(prod.register_spy.calls) == 2


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_bytes_sent_again_count_against_the_step_cap(monkeypatch, bookkeeping):
    """A tensor inlined in an earlier step rides this step's frames too (a loop
    edge re-sending earlier outputs), so its bytes come off this budget; one
    that no longer fits is registered and read instead."""
    prod = _manager("worker_0", monkeypatch, max_bytes=4096, step_max=6000,
                    bookkeeping=bookkeeping, fake_engine=True)
    infos = prod.store_and_return_tensor_info(
        "r", {"x": [torch.zeros(4000, dtype=torch.uint8) for _ in range(3)]},
    )["x"]
    a, b, c = (i.uuid for i in infos)
    prod.register_for_send_uuids(ParallelList(["r"], [[a]]))
    prod.register_for_send_uuids(ParallelList(["r"], [[b]]))
    assert prod._inlined == {a, b}
    # one call re-sending a and b and adding c: 12000 bytes against 6000
    prod.register_for_send_uuids(ParallelList(["r"], [[a, b, c]]))
    inline = {u: prod.tensor_store.get_info(u).inline_data is not None for u in (a, b, c)}
    assert inline == {a: True, b: False, c: False}
    assert prod.tensor_store.is_registered(b) and prod.tensor_store.is_registered(c)


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_a_tensor_received_inline_keeps_its_bytes_when_sent_on(monkeypatch, bookkeeping):
    """Its descriptor points at the api server's memory, which was never
    registered: over the step cap or persisted, it still rides inline."""
    api = _manager("api_server", monkeypatch, bookkeeping=bookkeeping)
    infos = api.store_and_return_tensor_info("r", {"x": [torch.arange(8, dtype=torch.int64)]})["x"]
    api.register_for_send("r", infos)
    wire = decode(encode(GraphEdge(next_node="n", name="x", tensor_info=infos)))
    worker = _manager("worker_1", monkeypatch, step_max=16, bookkeeping=bookkeeping)
    _receive(worker, "r", [wire])
    uuid = infos[0].uuid
    worker.register_for_send_uuids(ParallelList(["r"], [[uuid]]))
    worker.keep_registered([uuid])
    worker.register_for_send_uuids(ParallelList(["r"], [[uuid]]))
    assert worker.tensor_store.get_info(uuid).inline_data == infos[0].inline_data
    assert worker.register_spy.calls == []


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_refresh_clears_bytes_from_copies_made_before_a_demotion(monkeypatch, bookkeeping):
    """A cached descriptor copy (the Python runtime under the Rust bookkeeper)
    still holds the bytes after its tensor is registered: the refresh at send
    time takes them off, so the frame follows the store."""
    prod = _manager("worker_0", monkeypatch, bookkeeping=bookkeeping, fake_engine=True)
    infos = prod.store_and_return_tensor_info("r", {"x": [torch.arange(2, dtype=torch.int64)]})["x"]
    uuid = infos[0].uuid
    prod.register_for_send_uuids(ParallelList(["r"], [[uuid]]))
    cached = prod.tensor_store.get_info(uuid).clone()
    assert cached.inline_data is not None
    prod.keep_registered([uuid])
    prod.refresh_inline_data([GraphEdge(next_node="n", name="x", tensor_info=[cached])])
    assert cached.inline_data is None


def test_the_worker_keeps_persisted_tensors_registered_with_nothing_to_register():
    """A single-worker persist has no remote reader, so nothing is staged; it
    still rides WORKER_GRAPHS_DONE and must be kept off the inline path."""
    from types import SimpleNamespace

    from mstar.graph.runtime.base import RouteOutput
    from mstar.worker.worker import Worker

    calls = []
    fake = SimpleNamespace(tensor_manager=SimpleNamespace(
        keep_registered=calls.append,
        register_for_send_uuids=lambda *a, **k: calls.append("registered"),
    ))
    Worker._register_outputs(fake, RouteOutput(
        completion_id=1, register_uuids=[], register_rids=[], new_token_output_idxs=[],
        local_streaming_by_signal={}, persist_uuids=[5],
    ))
    assert calls == [[5]]


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_a_contiguous_tensor_with_an_odd_singleton_stride_goes_inline(monkeypatch, bookkeeping):
    prod = _manager("worker_0", monkeypatch, bookkeeping=bookkeeping)
    cons = _manager("worker_1", monkeypatch, bookkeeping=bookkeeping)
    odd = torch.tensor([[11, 22]])[:, -1]
    assert odd.is_contiguous() and odd.stride() == (2,)
    edges, _, _ = _send(prod, "r", "x", [odd])
    assert edges[0].tensor_info[0].inline_data is not None
    (got,) = _receive(cons, "r", edges)["x"]
    assert got.tolist() == [22]


def test_every_outgoing_copy_counts_against_the_step_cap(monkeypatch):
    """One tensor routed to two nodes on the same peer rides that frame twice."""
    prod = _manager("worker_0", monkeypatch, max_bytes=4096, step_max=6000, fake_engine=True)
    infos = prod.store_and_return_tensor_info(
        "r", {"x": [torch.zeros(4000, dtype=torch.uint8), torch.zeros(1000, dtype=torch.uint8)]},
    )["x"]
    twice, once = infos[0].uuid, infos[1].uuid
    prod.note_copies([twice, once], [2, 1])
    prod.register_for_send_uuids(ParallelList(["r"], [[twice, once]]))
    assert prod.tensor_store.get_info(twice).inline_data is None
    assert prod.tensor_store.get_info(once).inline_data is not None
    # counts apply to one call only
    prod.register_for_send_uuids(ParallelList(["r"], [[twice]]))
    assert prod.tensor_store.is_registered(twice)


def test_a_descriptor_stripped_of_its_bytes_fails_the_read_loudly(monkeypatch):
    """An older peer drops the unknown field when it relays a descriptor (all
    peers must run the same version, as since #407). The read that follows is
    of memory the producer never registered: Mooncake refuses it, so the
    request fails instead of reading anything."""
    prod = _manager("worker_0", monkeypatch)
    cons = _manager("worker_1", monkeypatch)
    edges, _, _ = _send(prod, "r", "x", [torch.arange(4, dtype=torch.int64)])
    edges[0].tensor_info[0].inline_data = None
    futures = cons.start_read_tensors("r", edges)
    assert futures
    with pytest.raises(RuntimeError, match="Mooncake read failed"):
        futures[0].result(timeout=30)


@pytest.mark.parametrize("bookkeeping", _BOOKKEEPERS)
def test_the_descriptor_form_stamps_the_objects_it_was_given(monkeypatch, bookkeeping):
    """The api server registers descriptor objects and sends those same
    objects; under the Rust bookkeeper they are copies of the store's."""
    prod = _manager("api_server", monkeypatch, bookkeeping=bookkeeping)
    infos = prod.store_and_return_tensor_info("r", {"ids": [torch.arange(5, dtype=torch.int64)]})["ids"]
    copies = [i.clone() for i in infos]
    prod.register_for_send("r", copies)
    assert copies[0].inline_data == torch.arange(5, dtype=torch.int64).numpy().tobytes()
    assert prod.register_spy.calls == []


def test_transport_stats_count_inline_and_read_per_edge(monkeypatch):
    prod = _manager("worker_0", monkeypatch, max_bytes=1024, stats=True)
    cons = _manager("worker_1", monkeypatch, max_bytes=1024, stats=True)
    edges, _, _ = _send(prod, "r", "x", [torch.zeros(16, dtype=torch.uint8),
                                         torch.zeros(4096, dtype=torch.uint8)])
    _receive(cons, "r", edges)
    sent = prod.transport_stats.snapshot()["sent"]
    assert sent["inline"] == [1, 16] and sent["over_tensor_max"] == [1, 4096]
    assert sent["registered"] == [1, 4096]
    (row,) = cons.transport_stats.snapshot()["rx"]
    source, edge, inline_n, inline_b, read_n, read_b, read_s, lo, hi = row
    assert (source, edge, inline_n, inline_b, read_n, read_b, lo, hi) == (
        "worker_0", "x", 1, 16, 1, 4096, 16, 4096)
    assert read_s > 0
    (sizes,) = cons.transport_stats.snapshot()["rx_sizes"]
    assert sizes == ["worker_0", "x", [(4, 1), (12, 1)]]


def test_wire_round_trip_keeps_the_bytes_exactly():
    info = TensorPointerInfo(
        dims=[3], dtype=torch.uint8, nbytes=3, address=0, stride=[1], uuid=1,
        source_session_id="h:1", source_entity="worker_0", inline_data=b"\x00\xff\x10",
    )
    msg = WorkerMessage(
        message_type=WorkerMessageType.TENSOR_RECEIVED,
        body=TensorReceived(request_id="r", successful_tensors={1: 1}, failed_tensor_ids=[]),
    )
    assert decode(encode(msg)) == msg
    edge = GraphEdge(next_node="n", name="x", tensor_info=[info])
    back = decode(encode(InputSignals(
        request_id="r", inputs=[edge],
        request_info=CurrentForwardPassInfo(
            request_id="r", graph_walk="w", fwd_index=0, random_seed=0, max_tokens=1),
    )))
    assert back.inputs[0].tensor_info[0].inline_data == b"\x00\xff\x10"
    assert back.inputs[0].tensor_info[0] == info


def test_defaults_inline_up_to_16_kib_within_1_mib_a_step(monkeypatch):
    monkeypatch.delenv(INLINE_TENSOR_MAX_BYTES_ENV, raising=False)
    monkeypatch.delenv(INLINE_STEP_MAX_BYTES_ENV, raising=False)
    assert inline_tensor_max_bytes() == 16384
    assert inline_step_max_bytes() == 1 << 20


def test_env_values_are_checked(monkeypatch):
    monkeypatch.setenv(INLINE_TENSOR_MAX_BYTES_ENV, "-1")
    with pytest.raises(ValueError, match=INLINE_TENSOR_MAX_BYTES_ENV):
        inline_tensor_max_bytes()
    monkeypatch.setenv(INLINE_TENSOR_MAX_BYTES_ENV, "4k")
    with pytest.raises(ValueError):
        inline_tensor_max_bytes()
    monkeypatch.setenv(INLINE_STEP_MAX_BYTES_ENV, "65536")
    assert inline_step_max_bytes() == 65536


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_device_tensors_round_trip_through_the_side_streams(monkeypatch):
    monkeypatch.setenv(INLINE_TENSOR_MAX_BYTES_ENV, "16384")
    prod = create_tensor_communication_manager(
        protocol=CommProtocol.TCP, my_entity_id="worker_0", hostname="localhost",
        device="cuda:0", communicator=_Comm(), tcp_transfer_device="lo",
    )
    cons = create_tensor_communication_manager(
        protocol=CommProtocol.TCP, my_entity_id="worker_1", hostname="localhost",
        device="cuda:0", communicator=_Comm(), tcp_transfer_device="lo",
    )
    ts = [t.to("cuda:0") for t in SMALL] + [torch.randn(1, 4096, device="cuda:0").to(torch.bfloat16)]
    torch.cuda.synchronize()
    edges, _, _ = _send(prod, "r", "x", ts)
    got = _receive(cons, "r", edges)["x"]
    torch.cuda.synchronize()
    for want, have in zip(ts, got, strict=True):
        assert have.device.type == "cuda" and torch.equal(have.cpu(), want.cpu())
