"""A capture's rows run on SINK_PAGE, as the padding rows that replay it do.

Qwen3.5-0.8B with a 64-page pool (#728): the 8,192-token mixed and prefill
buckets need more pages than the pool holds, so capture admitted them for real
pages, the admit was refused, and the boot failed. A served prompt of 6,006
tokens pads into that bucket with 47 of the 63 pages, since a replay's padding
rows take none, so the bucket is reachable and has to be captured and sized.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

PAGE = 4
# a sink page and seven a request can hold
POOL = 8
BUCKET = BucketKey(graph_walk="mixed", bs=4, num_tokens=40)


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        return None

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager() -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512,
            max_num_pages=POOL, page_size=PAGE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._cg_max_seq_len = BUCKET.num_tokens
    return kv


def _capture_step(kv: KVManager) -> tuple[KVStep, StepContext]:
    """The bucket's capture: three one-token rows and a 37-token prompt row,
    13 pages if they were real, as `decode_rows_then_prompt` lays them out.
    Its rows are opened the way `DummyRowPool` opens them."""
    rids = (-1, -2, -3, -4)
    for rid in rids:
        kv.ingest_capture_row(rid)
    step = KVStep(segments=tuple(
        Segment(rid, "main", span) for rid, span in zip(rids, (1, 1, 1, 37), strict=True)
    ))
    ctx = StepContext(
        request_ids=rids, graph_walk=BUCKET.graph_walk, slot=0, capture=True,
        slot_lease=SlotLease(slot=0, bucket=BUCKET), force_prefill=True,
    )
    return step, ctx


def test_a_capture_wider_than_the_pool_runs_on_the_sink_page():
    kv = _manager()
    free = kv._arena.num_free
    step, ctx = _capture_step(kv)

    outcome = kv.admit(step, ctx)
    assert outcome.ok, f"a capture must not depend on the pool's size: {outcome.reason}"
    plan = kv.plan(step, ctx)["main"].cpu_indptrs

    assert set(plan.paged_kv_indices.tolist()) == {SINK_PAGE}, "a capture row must hold no pages"
    # the spans a real step of this shape would plan: a page per page of span
    assert plan.paged_kv_indptr.tolist() == [0, 1, 2, 3, 13]
    assert plan.qo_indptr.tolist() == [0, 1, 2, 3, 40]
    assert kv._arena.num_free == free, "capture must leave every page free"


def test_a_served_step_that_pads_into_that_bucket_still_takes_real_pages():
    """24 tokens, 6 of the 7 pages: the captured bucket is the one it replays."""
    kv = _manager()
    capture, capture_ctx = _capture_step(kv)
    assert kv.admit(capture, capture_ctx).ok
    kv.plan(capture, capture_ctx)

    kv.ingest_request("r", KVReqConfig())
    padded = ("r", -1, -2, -3)
    step = KVStep(segments=(Segment("r", "main", 24),) + tuple(Segment(rid, "main", 0) for rid in padded[1:]))
    ctx = StepContext(
        request_ids=("r",), graph_walk=BUCKET.graph_walk, slot=0, capture=False,
        slot_lease=SlotLease(slot=0, bucket=BUCKET), force_prefill=True,
    )
    ctx.set_padded_rids(padded)

    assert kv.admit(step, ctx).ok
    plan = kv.plan(step, ctx)["main"].cpu_indptrs
    real = plan.paged_kv_indices[: plan.paged_kv_indptr[1]].tolist()

    assert len(real) == 6 and SINK_PAGE not in real
    assert kv._arena.num_free == POOL - 1 - 6
