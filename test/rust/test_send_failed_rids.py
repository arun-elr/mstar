"""The Rust runtime wrapper hands ``SendInput.failed_rids`` to Rust, which
then builds no frame for those rids (no WORKER_GRAPHS_DONE among them); the
Python runtime's side is in test/modular/test_send_registration_failure.py."""
import pytest
import torch

pytest.importorskip("mstar_rust")

from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, TensorPointerInfo
from mstar.graph.runtime import rust as rust_runtime
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput, SendInput
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList


@pytest.mark.parametrize("failed", [False, True])
def test_a_failed_rid_gets_no_completion_frame(failed):
    section = GraphNode(
        name="gen", input_names={"tok"},
        outputs=[GraphEdge(name="out", next_node=EMIT_TO_CLIENT)],
    )
    wg = WorkerGraph(section=section, graph_walks={"w"}, ranks=[0], worker_graph_id=0)
    book = RustTensorBookkeeping()
    rt = rust_runtime.RustGraphRuntime(
        my_worker_id="worker_0", my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={0: {"w"}}, all_wg_ids_to_dyn_loops={0: set()},
        all_wg_ids_to_nodes={0: {"gen"}}, node_to_partition={"gen": "default"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=book,
    )
    rid = rt.add_request(
        request_id="r", partition="default", graph_walk="w",
        partition_worker_graph_ids=[0],
        worker_graph_to_workers=ParallelList([0], [["worker_0"]]),
        loop_caps={},
    )
    block = ColumnarEdgeSpecs.empty()
    block.add(rid, "tok", [], False, next_node="gen")
    rt.ingest_inputs_batch(block)
    rt.pop_rids("gen", "w", [rid])
    book.put_tensor(1, TensorPointerInfo(
        dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=(1,),
        uuid=1, source_session_id="h:1", source_entity="worker_0",
    ))
    book.increment_ref(1, 1)
    out = rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk="w", node_name="gen",
        output_signals=["out"], wg_ids=ParallelList([rid], [0]),
        tensors=[1], num_tensors=[1],
    ))

    sent_rids = rt.send_outputs(SendInput(
        completion_id=out.completion_id,
        per_request_info=ParallelList([rid], [None]),
        new_token_counts=ParallelList([], []),
        failed_rids=frozenset({rid}) if failed else frozenset(),
    ))

    assert sent_rids == ([] if failed else [rid])
