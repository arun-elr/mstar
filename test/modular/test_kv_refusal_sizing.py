"""A refused KV admit says how many of the step's requests fit.

The worker sizes the retry from it: ``rows_fit`` is what the free pages held,
``rows_fit_alone`` what the pool would hold with nothing else resident. A
request that alone needs more pages than the pool has is refused for good,
since no eviction makes it fit and retrying it only spins until the client
gives up.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import AdmitRuntimeError, AllocationFailed, Segment, StepContext

PAGE_SIZE = 16


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int) -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=8,
            max_seq_len=4096,
            max_num_pages=max_num_pages,
            page_size=PAGE_SIZE,
            cpu_offload_pages=0,
        ),
        name="kv",
        joint_comm_group=None,
        transfer_engine_info=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _ctx(*rids: str) -> StepContext:
    return StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False)


def _admit(mgr: KVManager, spans: dict[str, int], commit: bool = False):
    for rid in spans:
        if rid not in mgr._streams:
            mgr.ingest_request(rid)
    step = KVStep(segments=tuple(Segment(rid, "main", span) for rid, span in spans.items()))
    ctx = _ctx(*spans)
    outcome = mgr.admit(step, ctx)
    if outcome.ok and commit:
        mgr.plan(step, ctx)
        mgr.commit(step, ctx)
    elif not outcome.ok:
        mgr.rollback_admit(step, ctx)
    return outcome


def test_a_refusal_counts_the_rows_that_fit_now_and_alone():
    """15 usable pages. A resident request holds 6; five new ones need 3 each.
    The free 9 hold three of them, the pool alone five."""
    mgr = _manager(16)
    assert _admit(mgr, {"old": 6 * PAGE_SIZE}, commit=True).ok

    outcome = _admit(mgr, {f"n{i}": 3 * PAGE_SIZE for i in range(5)})

    assert not outcome.ok
    reason = outcome.reason
    assert isinstance(reason, AllocationFailed) and reason.request_id == "n3"
    assert (reason.rows_fit, reason.rows_fit_alone) == (3, 5)


def test_a_batch_bigger_than_the_pool_says_so():
    """The whisper case: 16 rows of 12 pages need 192, 191 are usable."""
    mgr = _manager(192)

    outcome = _admit(mgr, {f"p{i}": 12 * PAGE_SIZE for i in range(16)})

    assert not outcome.ok
    assert outcome.reason.rows_fit == 15 and outcome.reason.rows_fit_alone == 15


def test_a_request_bigger_than_the_pool_is_refused_for_good():
    """Its retry was refused about 19 times a second until the client gave up."""
    mgr = _manager(8)

    outcome = _admit(mgr, {"huge": 8 * PAGE_SIZE})

    assert not outcome.ok
    assert isinstance(outcome.reason, AdmitRuntimeError) and outcome.reason.terminal
    assert outcome.reason.request_ids == frozenset({"huge"})


def test_a_request_that_fits_the_pool_but_not_the_free_pages_is_retried():
    mgr = _manager(8)
    assert _admit(mgr, {"old": 4 * PAGE_SIZE}, commit=True).ok

    outcome = _admit(mgr, {"big": 7 * PAGE_SIZE})

    assert isinstance(outcome.reason, AllocationFailed)
    assert (outcome.reason.rows_fit, outcome.reason.rows_fit_alone) == (0, 1)


def test_a_growing_request_counts_the_pages_it_holds():
    """A decode row past a page boundary needs one page more than it holds;
    its footprint is all of them, so one that outgrew the pool is refused
    for good rather than evicting everything else for nothing."""
    mgr = _manager(8)
    assert _admit(mgr, {"long": 7 * PAGE_SIZE}, commit=True).ok

    outcome = _admit(mgr, {"long": 1})

    assert isinstance(outcome.reason, AdmitRuntimeError)
