# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: ${TMPDIR}/torchinductor_uid_1001/fu/cfunaihzimmbhpj7fbzyvvg4qf635g27gyeyn2wak7hw6ty5w4n6.py
# Topologically Sorted Source Nodes: [float_1, normalize], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   normalize => pow_1, sum_1
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2.0), kwargs = {})
#   %sum_1 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   return %buf0
triton_red_fused__to_copy_linalg_vector_norm_0 = async_compile.triton('triton_red_fused__to_copy_linalg_vector_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_linalg_vector_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 3294720, 'r0_': 0}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_linalg_vector_norm_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12480
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1560*r0_2 + 199680*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp4, xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/j6/cj6ywcib63cmpgabmlp3gazcryn32pnxu3k7hrggtl726xpjgsk5.py
# Topologically Sorted Source Nodes: [float_1, normalize], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   normalize => pow_1, sum_1
# Graph fragment:
#   %buf0 : Tensor "f32[1, 1, 1, 30, 52, 8][12544, 12544, 12544, 52, 1, 1568]cuda:0" = PlaceHolder[target=buf0]
#   %convert_element_type : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2.0), kwargs = {})
#   %sum_1 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   return %sum_1
triton_per_fused__to_copy_linalg_vector_norm_1 = async_compile.triton('triton_per_fused__to_copy_linalg_vector_norm_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__to_copy_linalg_vector_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 62400, 'r0_': 0}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_linalg_vector_norm_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1560
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1568*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/tn/ctnpgclkl2heyl4r7drgriudefbjh5x4skk2gypnhzauw57ucpk4.py
# Topologically Sorted Source Nodes: [float_1, normalize, normalized, mul, mul_1, x, x_1, x_2], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   mul => mul
#   mul_1 => mul_1
#   normalize => clamp_min, div, expand, pow_2
#   normalized => convert_element_type_1
#   x => add
#   x_1 => add_1, convert_element_type_2, convert_element_type_3, div_1, exp, neg
#   x_2 => cat_1
# Graph fragment:
#   %arg2_1 : Tensor "bf16[1, 1024, 2, 30, 52][3194880, 3120, 1560, 52, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-12), kwargs = {})
#   %expand : Tensor "f32[1, 1024, 1, 30, 52][1560, 0, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min, [1, 1024, 1, 30, 52]), kwargs = {})
#   %div : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %expand), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %mul : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, 32.0), kwargs = {})
#   %mul_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg1_1), kwargs = {})
#   %add : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, 0.0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %neg : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_2,), kwargs = {})
#   %exp : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_1 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div_1 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, %add_1), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   %cat_1 : Tensor "bf16[1, 1024, 3, 30, 52][4792320, 4680, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg2_1, %convert_element_type_3], 2), kwargs = {})
#   return %buf2
triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2 = async_compile.triton('triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 19169280}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3194880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 3120)
    x1 = xindex // 3120
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + 4680*x1), tmp0, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/wt/cwt6czukwa2oyyhbbfdmld5xlvovdqjo2mpvkgexrtqoyvbat5bi.py
# Topologically Sorted Source Nodes: [float_1, normalize, normalized, mul, mul_1, x, x_1], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   mul => mul
#   mul_1 => mul_1
#   normalize => clamp_min, div, expand, pow_2
#   normalized => convert_element_type_1
#   x => add
#   x_1 => add_1, convert_element_type_2, convert_element_type_3, div_1, exp, neg
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sum_1 : Tensor "f32[1, 1, 1, 30, 52][1568, 1568, 1568, 52, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %arg1_1 : Tensor "bf16[1024, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_2, 1e-12), kwargs = {})
#   %expand : Tensor "f32[1, 1024, 1, 30, 52][1560, 0, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min, [1, 1024, 1, 30, 52]), kwargs = {})
#   %div : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %expand), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %mul : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, 32.0), kwargs = {})
#   %mul_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg1_1), kwargs = {})
#   %add : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, 0.0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %neg : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_2,), kwargs = {})
#   %exp : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_1 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div_1 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, %add_1), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_3
triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_3 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 9592928}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.sqrt_rn(tmp2)
    tmp4 = tl.full([1], 1e-12, tl.float32)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1], 32.0, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1], 0.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = -tmp14
    tmp16 = libdevice.exp(tmp15)
    tmp17 = tl.full([1], 1.0, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = (tmp14 / tmp18)
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 4680*x1), tmp20, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/3x/c3xicc4rhms6q7gqpq2ejz2s6gg3kqvho6ningvhr45k4b4qmqro.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_3 => constant_pad_nd
# Graph fragment:
#   %cat_1 : Tensor "bf16[1, 1024, 3, 30, 52][4792320, 4680, 1560, 52, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %constant_pad_nd : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_1, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   return %constant_pad_nd
triton_poi_fused_constant_pad_nd_4 = async_compile.triton('triton_poi_fused_constant_pad_nd_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_constant_pad_nd_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 30818304}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 54) % 32)
    x0 = (xindex % 54)
    x2 = xindex // 1728
    x4 = xindex
    tmp0 = ((-1) + x1).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = ((-1) + x0).to(tl.int32)
    tmp6 = tmp5 >= tmp1
    tmp7 = tl.full([1], 52, tl.int64)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-53) + x0 + 52*x1 + 1560*x2), tmp11, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/k6/ck6poezku2jp6ywf4zelpowci7gijcmkhwifzfhdt2gegou4re4r.py
# Topologically Sorted Source Nodes: [x_3, x_4, float_2, normalize_1], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   float_2 => convert_element_type_4
#   normalize_1 => pow_3, sum_2
#   x_3 => constant_pad_nd
#   x_4 => convolution
# Graph fragment:
#   %buf6 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf6]
#   %arg4_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %constant_pad_nd : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_1, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %arg3_1, %arg4_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_4, 2.0), kwargs = {})
#   %sum_2 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})
#   return %buf7
triton_red_fused__to_copy_constant_pad_nd_convolution_linalg_vector_norm_5 = async_compile.triton('triton_red_fused__to_copy_constant_pad_nd_convolution_linalg_vector_norm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_constant_pad_nd_convolution_linalg_vector_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 3294720, 'r0_': 2048}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_constant_pad_nd_convolution_linalg_vector_norm_5(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12480
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1560*r0_2 + 199680*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp6, xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/ua/cua3cz4xww62smjb6wapv3gbcztqwfjecsytwhqio5ntkenkn6tz.py
# Topologically Sorted Source Nodes: [x_3, x_4, float_2, normalize_1, normalized_1, mul_2, mul_3, x_5, x_6], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   float_2 => convert_element_type_4
#   mul_2 => mul_2
#   mul_3 => mul_3
#   normalize_1 => clamp_min_1, div_2, expand_1, pow_4
#   normalized_1 => convert_element_type_5
#   x_3 => constant_pad_nd
#   x_4 => convolution
#   x_5 => add_2
#   x_6 => add_3, convert_element_type_6, convert_element_type_7, div_3, exp_1, neg_1
# Graph fragment:
#   %buf6 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf6]
#   %arg4_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %sum_2 : Tensor "f32[1, 1, 1, 30, 52][1568, 1568, 1568, 52, 1]cuda:0" = PlaceHolder[target=sum_2]
#   %arg5_1 : Tensor "bf16[1024, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %constant_pad_nd : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_1, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %arg3_1, %arg4_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution, torch.float32), kwargs = {})
#   %pow_4 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_1 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_4, 1e-12), kwargs = {})
#   %expand_1 : Tensor "f32[1, 1024, 1, 30, 52][1560, 0, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_1, [1, 1024, 1, 30, 52]), kwargs = {})
#   %div_2 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %expand_1), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.bfloat16), kwargs = {})
#   %mul_2 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 32.0), kwargs = {})
#   %mul_3 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg5_1), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 0.0), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %neg_1 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_6,), kwargs = {})
#   %exp_1 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %add_3 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div_3 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_6, %add_3), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_7
triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_6 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 9594976}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 1560
    x0 = (xindex % 1560)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tl.sqrt_rn(tmp4)
    tmp6 = tl.full([1], 1e-12, tl.float32)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = (tmp3 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.full([1], 32.0, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full([1], 0.0, tl.float32)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = -tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tl.full([1], 1.0, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = (tmp16 / tmp20)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 4680*x1), tmp22, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/tn/ctn5c6kasguo6jwgnyb2gl67bfllz32pmj4dv4w75qmzlyqslick.py
# Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   float_3 => convert_element_type_8
#   normalize_2 => pow_5, sum_3
#   permute => permute
#   x_10 => convolution_1
#   x_11 => add_4
#   x_12 => view
#   x_9 => constant_pad_nd_1
# Graph fragment:
#   %buf13 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf13]
#   %arg8_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg0_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %constant_pad_nd_1 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_3, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %arg7_1, %arg8_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_4 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %arg0_1), kwargs = {})
#   %permute : Tensor "bf16[1, 1, 1024, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [0, 2, 1, 3, 4]), kwargs = {})
#   %view : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute, [1, 1024, 30, 52]), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_8, 2.0), kwargs = {})
#   %sum_3 : Tensor "f32[1, 1, 30, 52][1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1], True), kwargs = {})
#   return %buf14
triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_permute_view_7 = async_compile.triton('triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_permute_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_permute_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 6489600, 'r0_': 2048}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_permute_view_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12480
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1560*r0_2 + 199680*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 1560*r0_2 + 199680*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp8, xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/zu/czuuqhc2bmtxc25qqefllpg6ntlfudlyl3jcvh6jmvfvzdd4njvh.py
# Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
# Source node to ATen node mapping:
#   float_3 => convert_element_type_8
#   mul_4 => mul_4
#   mul_5 => mul_5
#   normalize_2 => clamp_min_2, div_4, expand_2, pow_6
#   normalized_2 => convert_element_type_9
#   permute => permute
#   x_10 => convolution_1
#   x_11 => add_4
#   x_12 => view
#   x_13 => add_5
#   x_9 => constant_pad_nd_1
# Graph fragment:
#   %buf13 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf13]
#   %arg8_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg0_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sum_3 : Tensor "f32[1, 1, 30, 52][1568, 1568, 52, 1]cuda:0" = PlaceHolder[target=sum_3]
#   %arg9_1 : Tensor "bf16[1024, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %constant_pad_nd_1 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_3, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %arg7_1, %arg8_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_4 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %arg0_1), kwargs = {})
#   %permute : Tensor "bf16[1, 1, 1024, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [0, 2, 1, 3, 4]), kwargs = {})
#   %view : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute, [1, 1024, 30, 52]), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %pow_6 : Tensor "f32[1, 1, 30, 52][1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %clamp_min_2 : Tensor "f32[1, 1, 30, 52][1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_6, 1e-12), kwargs = {})
#   %expand_2 : Tensor "f32[1, 1024, 30, 52][1560, 0, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_2, [1, 1024, 30, 52]), kwargs = {})
#   %div_4 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_8, %expand_2), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_4, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, 32.0), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg9_1), kwargs = {})
#   %add_5 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, 0.0), kwargs = {})
#   return %add_5
triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_view_8 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 6389760, 'x': 6400096}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1560
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1560*y0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1 + 1560*y0), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tl.sqrt_rn(tmp6)
    tmp8 = tl.full([1, 1], 1e-12, tl.float32)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = (tmp5 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.full([1, 1], 32.0, tl.float32)
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full([1, 1], 0.0, tl.float32)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (y0 + 1024*x1), tmp17, xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/zq/czqen34qla637h5k3jk4e4laexan3i2zyo4fhakv3ow72zlz7tmi.py
# Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13, qkv, qkv_1, permute_1, qkv_2, chunk, x_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.clone, aten.split, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   chunk => split
#   float_3 => convert_element_type_8
#   mul_4 => mul_4
#   mul_5 => mul_5
#   normalize_2 => clamp_min_2, div_4, expand_2, pow_6
#   normalized_2 => convert_element_type_9
#   permute => permute
#   permute_1 => permute_1
#   qkv => convolution_2
#   qkv_1 => view_1
#   qkv_2 => clone_3
#   x_10 => convolution_1
#   x_11 => add_4
#   x_12 => view
#   x_13 => add_5
#   x_14 => _scaled_dot_product_efficient_attention
#   x_9 => constant_pad_nd_1
# Graph fragment:
#   %buf17 : Tensor "bf16[1, 3072, 30, 52][4792320, 1, 159744, 3072]cuda:0" = PlaceHolder[target=buf17]
#   %arg11_1 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %constant_pad_nd_1 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_3, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %arg7_1, %arg8_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_4 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %arg0_1), kwargs = {})
#   %permute : Tensor "bf16[1, 1, 1024, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [0, 2, 1, 3, 4]), kwargs = {})
#   %view : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute, [1, 1024, 30, 52]), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %pow_6 : Tensor "f32[1, 1, 30, 52][1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %clamp_min_2 : Tensor "f32[1, 1, 30, 52][1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_6, 1e-12), kwargs = {})
#   %expand_2 : Tensor "f32[1, 1024, 30, 52][1560, 0, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_2, [1, 1024, 30, 52]), kwargs = {})
#   %div_4 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_8, %expand_2), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_4, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, 32.0), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg9_1), kwargs = {})
#   %add_5 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, 0.0), kwargs = {})
#   %convolution_2 : Tensor "bf16[1, 3072, 30, 52][4792320, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_5, %arg10_1, %arg11_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_1 : Tensor "bf16[1, 1, 3072, 1560][4792320, 4792320, 1560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_2, [1, 1, 3072, -1]), kwargs = {})
#   %permute_1 : Tensor "bf16[1, 1, 1560, 3072][4792320, 4792320, 1, 1560]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 1, 1560, 3072][4792320, 4792320, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%clone_3, 1024, -1), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%getitem, %getitem_1, %getitem_2, None, False), kwargs = {})
#   return %buf18
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_9 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 9586688}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 3072*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/tr/ctrvhtg45wuubd5ymu3d7eq4rwxailajm2ahzdopes4hy3mxdk6d.py
# Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13, qkv, qkv_1, permute_1, qkv_2, chunk, x_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.clone, aten.split, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   chunk => split
#   float_3 => convert_element_type_8
#   mul_4 => mul_4
#   mul_5 => mul_5
#   normalize_2 => clamp_min_2, div_4, expand_2, pow_6
#   normalized_2 => convert_element_type_9
#   permute => permute
#   permute_1 => permute_1
#   qkv => convolution_2
#   qkv_1 => view_1
#   qkv_2 => clone_3
#   x_10 => convolution_1
#   x_11 => add_4
#   x_12 => view
#   x_13 => add_5
#   x_14 => _scaled_dot_product_efficient_attention
#   x_9 => constant_pad_nd_1
# Graph fragment:
#   %buf17 : Tensor "bf16[1, 3072, 30, 52][4792320, 1, 159744, 3072]cuda:0" = PlaceHolder[target=buf17]
#   %arg11_1 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %constant_pad_nd_1 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_3, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %arg7_1, %arg8_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_4 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %arg0_1), kwargs = {})
#   %permute : Tensor "bf16[1, 1, 1024, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [0, 2, 1, 3, 4]), kwargs = {})
#   %view : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute, [1, 1024, 30, 52]), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %pow_6 : Tensor "f32[1, 1, 30, 52][1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %clamp_min_2 : Tensor "f32[1, 1, 30, 52][1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_6, 1e-12), kwargs = {})
#   %expand_2 : Tensor "f32[1, 1024, 30, 52][1560, 0, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_2, [1, 1024, 30, 52]), kwargs = {})
#   %div_4 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_8, %expand_2), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_4, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, 32.0), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg9_1), kwargs = {})
#   %add_5 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, 0.0), kwargs = {})
#   %convolution_2 : Tensor "bf16[1, 3072, 30, 52][4792320, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_5, %arg10_1, %arg11_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_1 : Tensor "bf16[1, 1, 3072, 1560][4792320, 4792320, 1560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_2, [1, 1, 3072, -1]), kwargs = {})
#   %permute_1 : Tensor "bf16[1, 1, 1560, 3072][4792320, 4792320, 1, 1560]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 1, 1560, 3072][4792320, 4792320, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%clone_3, 1024, -1), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%getitem, %getitem_1, %getitem_2, None, False), kwargs = {})
#   return %buf19
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_10 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 9586688}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + 3072*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (1024 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/23/c23lhceasg4ifipdcp6wdx64wyhuinu6rov67ru23ypywxpx3ca3.py
# Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13, qkv, qkv_1, permute_1, qkv_2, chunk, x_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.clone, aten.split, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   chunk => split
#   float_3 => convert_element_type_8
#   mul_4 => mul_4
#   mul_5 => mul_5
#   normalize_2 => clamp_min_2, div_4, expand_2, pow_6
#   normalized_2 => convert_element_type_9
#   permute => permute
#   permute_1 => permute_1
#   qkv => convolution_2
#   qkv_1 => view_1
#   qkv_2 => clone_3
#   x_10 => convolution_1
#   x_11 => add_4
#   x_12 => view
#   x_13 => add_5
#   x_14 => _scaled_dot_product_efficient_attention
#   x_9 => constant_pad_nd_1
# Graph fragment:
#   %buf17 : Tensor "bf16[1, 3072, 30, 52][4792320, 1, 159744, 3072]cuda:0" = PlaceHolder[target=buf17]
#   %arg11_1 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %constant_pad_nd_1 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_3, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %arg7_1, %arg8_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_4 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %arg0_1), kwargs = {})
#   %permute : Tensor "bf16[1, 1, 1024, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [0, 2, 1, 3, 4]), kwargs = {})
#   %view : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute, [1, 1024, 30, 52]), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %pow_6 : Tensor "f32[1, 1, 30, 52][1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %clamp_min_2 : Tensor "f32[1, 1, 30, 52][1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_6, 1e-12), kwargs = {})
#   %expand_2 : Tensor "f32[1, 1024, 30, 52][1560, 0, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_2, [1, 1024, 30, 52]), kwargs = {})
#   %div_4 : Tensor "f32[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_8, %expand_2), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_4, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, 32.0), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg9_1), kwargs = {})
#   %add_5 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, 0.0), kwargs = {})
#   %convolution_2 : Tensor "bf16[1, 3072, 30, 52][4792320, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_5, %arg10_1, %arg11_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_1 : Tensor "bf16[1, 1, 3072, 1560][4792320, 4792320, 1560, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_2, [1, 1, 3072, -1]), kwargs = {})
#   %permute_1 : Tensor "bf16[1, 1, 1560, 3072][4792320, 4792320, 1, 1560]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 1, 1560, 3072][4792320, 4792320, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%clone_3, 1024, -1), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%getitem, %getitem_1, %getitem_2, None, False), kwargs = {})
#   return %buf20
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_11 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 9586688}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + 3072*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (2048 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/bj/cbjd6eedklh5c4kyp5vwhtr6bpgu4mjs5r6dhl5euk2lvmmbpdhj.py
# Topologically Sorted Source Nodes: [x_9, x_10, x_11, squeeze, permute_2, x_15, x_16, x_17, x_18, x_19], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.squeeze, aten.permute, aten.view]
# Source node to ATen node mapping:
#   permute_2 => permute_2
#   squeeze => squeeze
#   x_10 => convolution_1
#   x_11 => add_4
#   x_15 => view_2
#   x_16 => convolution_3
#   x_17 => view_3
#   x_18 => permute_3
#   x_19 => add_6
#   x_9 => constant_pad_nd_1
# Graph fragment:
#   %buf26 : Tensor "bf16[1, 1024, 30, 52][1597440, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf26]
#   %arg13_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %buf13 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf13]
#   %arg8_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg0_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %constant_pad_nd_1 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_3, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %arg7_1, %arg8_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_4 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %arg0_1), kwargs = {})
#   %squeeze : Tensor "bf16[1, 1560, 1024][1597440, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%getitem_3, 1), kwargs = {})
#   %permute_2 : Tensor "bf16[1, 1024, 1560][1597440, 1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%squeeze, [0, 2, 1]), kwargs = {})
#   %view_2 : Tensor "bf16[1, 1024, 30, 52][1597440, 1, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_2, [1, 1024, 30, 52]), kwargs = {})
#   %convolution_3 : Tensor "bf16[1, 1024, 30, 52][1597440, 1, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_2, %arg12_1, %arg13_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_3 : Tensor "bf16[1, 1, 1024, 30, 52][1597440, 1024, 1, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_3, [1, 1, 1024, 30, 52]), kwargs = {})
#   %permute_3 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3, 4]), kwargs = {})
#   %add_6 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_3, %add_4), kwargs = {})
#   return %add_6
triton_poi_fused_add_constant_pad_nd_convolution_permute_squeeze_view_12 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_permute_squeeze_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_permute_squeeze_view_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 15978496}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_permute_squeeze_view_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 1560
    x0 = (xindex % 1560)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tl.store(out_ptr0 + (x0 + 1600*x1), tmp8, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/cd/ccd4eap3ljboi76kbjizacsx4ifd2dadks24pitnc6ttg5ndcmb2.py
# Topologically Sorted Source Nodes: [float_4, normalize_3], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   float_4 => convert_element_type_10
#   normalize_3 => pow_7, sum_4
# Graph fragment:
#   %add_6 : Tensor "bf16[1, 1024, 1, 30, 52][1638400, 1600, 1638400, 52, 1]cuda:0" = PlaceHolder[target=add_6]
#   %convert_element_type_10 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_10, 2.0), kwargs = {})
#   %sum_4 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1], True), kwargs = {})
#   return %buf28
triton_red_fused__to_copy_linalg_vector_norm_13 = async_compile.triton('triton_red_fused__to_copy_linalg_vector_norm_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_linalg_vector_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 3294720, 'r0_': 0}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_linalg_vector_norm_13(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12480
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1600*r0_2 + 204800*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp4, xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/z2/cz2hui2kotnu32ptws67qvhuy3gcxdvolrsg5nywnwtpoai3drab.py
# Topologically Sorted Source Nodes: [float_4, normalize_3, normalized_3, mul_6, mul_7, x_20, x_21], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   float_4 => convert_element_type_10
#   mul_6 => mul_6
#   mul_7 => mul_7
#   normalize_3 => clamp_min_3, div_5, expand_3, pow_8
#   normalized_3 => convert_element_type_11
#   x_20 => add_7
#   x_21 => add_8, convert_element_type_12, convert_element_type_13, div_6, exp_2, neg_2
# Graph fragment:
#   %add_6 : Tensor "bf16[1, 1024, 1, 30, 52][1638400, 1600, 1638400, 52, 1]cuda:0" = PlaceHolder[target=add_6]
#   %sum_4 : Tensor "f32[1, 1, 1, 30, 52][1568, 1568, 1568, 52, 1]cuda:0" = PlaceHolder[target=sum_4]
#   %arg14_1 : Tensor "bf16[1024, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg14_1]
#   %convert_element_type_10 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.float32), kwargs = {})
#   %pow_8 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %clamp_min_3 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_8, 1e-12), kwargs = {})
#   %expand_3 : Tensor "f32[1, 1024, 1, 30, 52][1560, 0, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_3, [1, 1024, 1, 30, 52]), kwargs = {})
#   %div_5 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_10, %expand_3), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_5, torch.bfloat16), kwargs = {})
#   %mul_6 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_11, 32.0), kwargs = {})
#   %mul_7 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %arg14_1), kwargs = {})
#   %add_7 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, 0.0), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.float32), kwargs = {})
#   %neg_2 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_12,), kwargs = {})
#   %exp_2 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_8 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_2, 1), kwargs = {})
#   %div_6 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_12, %add_8), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1, 1024, 53248, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.bfloat16), kwargs = {})
#   return %convert_element_type_13
triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_14 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 9592928}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    tmp0 = tl.load(in_ptr0 + (x0 + 1600*x1), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.sqrt_rn(tmp2)
    tmp4 = tl.full([1], 1e-12, tl.float32)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1], 32.0, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1], 0.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = -tmp14
    tmp16 = libdevice.exp(tmp15)
    tmp17 = tl.full([1], 1.0, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = (tmp14 / tmp18)
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 4680*x1), tmp20, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/cs/ccsxv5kotrl6hzjbzjn4s3xw5rps3xvdfiolxvul3nfpexaa2fr5.py
# Topologically Sorted Source Nodes: [x_29, x_30, x_31, float_6, normalize_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   float_6 => convert_element_type_18
#   normalize_5 => pow_11, sum_6
#   x_29 => constant_pad_nd_3
#   x_30 => convolution_5
#   x_31 => add_11
# Graph fragment:
#   %buf41 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf41]
#   %arg21_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %add_6 : Tensor "bf16[1, 1024, 1, 30, 52][1638400, 1600, 1638400, 52, 1]cuda:0" = PlaceHolder[target=add_6]
#   %constant_pad_nd_3 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_7, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_5 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_3, %arg20_1, %arg21_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_11 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1597440, 52, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add_6), kwargs = {})
#   %convert_element_type_18 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1597440, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float32), kwargs = {})
#   %pow_11 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_18, 2.0), kwargs = {})
#   %sum_6 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1], True), kwargs = {})
#   return %buf42
triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_15 = async_compile.triton('triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 6489600, 'r0_': 2048}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12480
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1560*r0_2 + 199680*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 1600*r0_2 + 204800*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp8, xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/wm/cwmrf22e7mveb22xjw3yc465a7yvlzmv7rpwxdk2y4if7esh2uiw.py
# Topologically Sorted Source Nodes: [x_29, x_30, x_31, float_6, normalize_5, normalized_5, mul_10, mul_11, x_32, x_33], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.silu]
# Source node to ATen node mapping:
#   float_6 => convert_element_type_18
#   mul_10 => mul_10
#   mul_11 => mul_11
#   normalize_5 => clamp_min_5, div_9, expand_5, pow_12
#   normalized_5 => convert_element_type_19
#   x_29 => constant_pad_nd_3
#   x_30 => convolution_5
#   x_31 => add_11
#   x_32 => add_12
#   x_33 => add_13, convert_element_type_20, convert_element_type_21, div_10, exp_4, neg_4
# Graph fragment:
#   %buf41 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf41]
#   %arg21_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %add_6 : Tensor "bf16[1, 1024, 1, 30, 52][1638400, 1600, 1638400, 52, 1]cuda:0" = PlaceHolder[target=add_6]
#   %sum_6 : Tensor "f32[1, 1, 1, 30, 52][1568, 1568, 1568, 52, 1]cuda:0" = PlaceHolder[target=sum_6]
#   %arg22_1 : Tensor "bf16[1024, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_20 : Tensor "f32[1, 1024, 1, 30, 52][1605632, 1568, 1605632, 52, 1]cuda:0" = PlaceHolder[target=convert_element_type_20]
#   %constant_pad_nd_3 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_7, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_5 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_3, %arg20_1, %arg21_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_11 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1597440, 52, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add_6), kwargs = {})
#   %convert_element_type_18 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1597440, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float32), kwargs = {})
#   %pow_12 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_6, 0.5), kwargs = {})
#   %clamp_min_5 : Tensor "f32[1, 1, 1, 30, 52][1560, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_12, 1e-12), kwargs = {})
#   %expand_5 : Tensor "f32[1, 1024, 1, 30, 52][1560, 0, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_5, [1, 1024, 1, 30, 52]), kwargs = {})
#   %div_9 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1597440, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_18, %expand_5), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1597440, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_9, torch.bfloat16), kwargs = {})
#   %mul_10 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, 32.0), kwargs = {})
#   %mul_11 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg22_1), kwargs = {})
#   %add_12 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, 0.0), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %neg_4 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_20,), kwargs = {})
#   %exp_4 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_4,), kwargs = {})
#   %add_13 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_4, 1), kwargs = {})
#   %div_10 : Tensor "f32[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_20, %add_13), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_10, torch.bfloat16), kwargs = {})
#   return %convert_element_type_20,%convert_element_type_21
triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_16 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 12789856}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 1560
    x0 = (xindex % 1560)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 1600*x1), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tl.sqrt_rn(tmp6)
    tmp8 = tl.full([1], 1e-12, tl.float32)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = (tmp5 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.full([1], 32.0, tl.float32)
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = -tmp18
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tl.full([1], 1.0, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = (tmp18 / tmp22)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x0 + 4680*x1), tmp24, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/zk/czkmyrefpwtucffpffpvrllxs75ut7jo7fom5r3ablithul5vce7.py
# Topologically Sorted Source Nodes: [x_29, x_30, x_31, x_41, x_42, x_43], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   x_29 => constant_pad_nd_3
#   x_30 => convolution_5
#   x_31 => add_11
#   x_41 => constant_pad_nd_5
#   x_42 => convolution_7
#   x_43 => add_16
# Graph fragment:
#   %buf56 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf56]
#   %arg29_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %buf41 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf41]
#   %arg21_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %add_6 : Tensor "bf16[1, 1024, 1, 30, 52][1638400, 1600, 1638400, 52, 1]cuda:0" = PlaceHolder[target=add_6]
#   %constant_pad_nd_3 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_7, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_5 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_3, %arg20_1, %arg21_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_11 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1597440, 52, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add_6), kwargs = {})
#   %constant_pad_nd_5 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_11, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_7 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_5, %arg28_1, %arg29_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_16 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %add_11), kwargs = {})
#   return %add_16
triton_poi_fused_add_constant_pad_nd_convolution_17 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 15978496}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 1560
    x0 = (xindex % 1560)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x0 + 1600*x1), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tl.store(out_ptr0 + (x0 + 1600*x1), tmp8, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/vk/cvklu5lyj4zlxnmjyi5q72kurge6nwf4ueo5digp2hznnkpv4jee.py
# Topologically Sorted Source Nodes: [x_53, x_54, x_55, x_65, x_66, x_67], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   x_53 => constant_pad_nd_7
#   x_54 => convolution_9
#   x_55 => add_21
#   x_65 => constant_pad_nd_9
#   x_66 => convolution_11
#   x_67 => add_26
# Graph fragment:
#   %buf86 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf86]
#   %arg45_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg45_1]
#   %buf71 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf71]
#   %arg37_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %add_16 : Tensor "bf16[1, 1024, 1, 30, 52][1638400, 1600, 1638400, 52, 1]cuda:0" = PlaceHolder[target=add_16]
#   %constant_pad_nd_7 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_15, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_9 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_7, %arg36_1, %arg37_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_21 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_9, %add_16), kwargs = {})
#   %constant_pad_nd_9 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_19, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_11 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_9, %arg44_1, %arg45_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_26 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_11, %add_21), kwargs = {})
#   return %add_26
triton_poi_fused_add_constant_pad_nd_convolution_18 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 15978496}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 1560
    x0 = (xindex % 1560)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x0 + 1600*x1), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 + tmp7
    tl.store(out_ptr0 + (x0 + 4680*x1), tmp8, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/st/cstwim755qhc2l35nu7v52stplkr52imytjs26rsv6g4jrd6be2c.py
# Topologically Sorted Source Nodes: [x_70, x_71, getitem_25, getitem_26, x_72, x_73, permute_4, x_74, float_12, interpolate, input_1], Original ATen: [aten.convolution, aten.view, aten.select, aten.stack, aten.permute, aten._to_copy, aten.arange, aten.add, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   float_12 => convert_element_type_42
#   getitem_25 => select_11
#   getitem_26 => select_12
#   input_1 => convert_element_type_47
#   interpolate => _unsafe_index, add_27, add_28, add_29, add_30, clone_18, convert_element_type_43, convert_element_type_44, convert_element_type_45, convert_element_type_46, iota, iota_1, mul_22, mul_23, mul_24, mul_25, unsqueeze_11
#   permute_4 => permute_4
#   x_70 => convolution_12
#   x_71 => view_4
#   x_72 => cat_22, view_5
#   x_73 => view_6
#   x_74 => view_7
# Graph fragment:
#   %buf90 : Tensor "bf16[1, 2048, 1, 30, 52][3194880, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf90]
#   %arg48_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg48_1]
#   %_unsafe_index : Tensor "f32[2, 1024, 60, 104][6389760, 1, 106496, 1024]cuda:0" = PlaceHolder[target=_unsafe_index]
#   %convolution_12 : Tensor "bf16[1, 2048, 1, 30, 52][3194880, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_21, %arg47_1, %arg48_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %view_4 : Tensor "bf16[1, 2, 1024, 1, 30, 52][3194880, 1597440, 1560, 1560, 52, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_12, [1, 2, 1024, 1, 30, 52]), kwargs = {})
#   %select_11 : Tensor "bf16[1, 1024, 1, 30, 52][3194880, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_4, 1, 0), kwargs = {})
#   %select_12 : Tensor "bf16[1, 1024, 1, 30, 52][3194880, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_4, 1, 1), kwargs = {})
#   %cat_22 : Tensor "bf16[1, 1024, 1, 60, 52][3194880, 3120, 3120, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%select_11, %select_12], 3), kwargs = {})
#   %view_5 : Tensor "bf16[1, 1024, 1, 2, 30, 52][3194880, 3120, 3120, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_22, [1, 1024, 1, 2, 30, 52]), kwargs = {})
#   %view_6 : Tensor "bf16[1, 1024, 2, 30, 52][3194880, 3120, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1024, 2, 30, 52]), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 2, 1024, 30, 52][3194880, 1560, 3120, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3, 4]), kwargs = {})
#   %view_7 : Tensor "bf16[2, 1024, 30, 52][1560, 3120, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [2, 1024, 30, 52]), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[2, 1024, 30, 52][1560, 3120, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_7, torch.float32), kwargs = {})
#   %iota : Tensor "i64[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (60,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_22 : Tensor "i64[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_27 : Tensor "i64[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, 0), kwargs = {})
#   %convert_element_type_43 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_27, torch.float32), kwargs = {})
#   %add_28 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_43, 0.5), kwargs = {})
#   %mul_23 : Tensor "f32[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, 0.5), kwargs = {})
#   %convert_element_type_44 : Tensor "i64[60][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.int64), kwargs = {})
#   %unsqueeze_11 : Tensor "i64[60, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_44, -1), kwargs = {})
#   %iota_1 : Tensor "i64[104][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (104,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_24 : Tensor "i64[104][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_29 : Tensor "i64[104][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, 0), kwargs = {})
#   %convert_element_type_45 : Tensor "f32[104][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_29, torch.float32), kwargs = {})
#   %add_30 : Tensor "f32[104][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_45, 0.5), kwargs = {})
#   %mul_25 : Tensor "f32[104][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_30, 0.5), kwargs = {})
#   %convert_element_type_46 : Tensor "i64[104][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_25, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[2, 1024, 60, 104][6240, 12480, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_42, [None, None, %unsqueeze_11, %convert_element_type_46]), kwargs = {})
#   %clone_18 : Tensor "f32[2, 1024, 60, 104][6389760, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_47 : Tensor "bf16[2, 1024, 60, 104][6389760, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_18, torch.bfloat16), kwargs = {})
#   return %_unsafe_index,%convert_element_type_47
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_19 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 51122176}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_19(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12779520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = ((xindex // 106496) % 60)
    x1 = ((xindex // 1024) % 104)
    x3 = xindex // 6389760
    x0 = (xindex % 1024)
    x4 = xindex
    tmp0 = (x2).to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4.to(tl.int32)
    tmp6 = (x1).to(tl.float32)
    tmp7 = tmp6 + tmp2
    tmp8 = tmp7 * tmp2
    tmp9 = tmp8.to(tl.int32)
    tmp10 = (tmp5 + 30*x3).to(tl.int32)
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = (tmp5 + 30*x3).to(tl.int64)
    tmp14 = (tmp13).to(tl.int64)
    tmp15 = tl.full([1], 30, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tl.load(in_ptr0 + (tmp9 + 52*(tmp5 + 30*x3) + 1560*x0), tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr1 + (x0), tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tmp10 >= tmp15
    tmp23 = tl.full([1], 60, tl.int64)
    tmp24 = tmp10 < tmp23
    tmp25 = tl.load(in_ptr0 + (1597440 + tmp9 + 52*((-30) + tmp5 + 30*x3) + 1560*x0), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (1024 + x0), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp22, tmp27, tmp28)
    tmp30 = tl.where(tmp16, tmp21, tmp29)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr1 + (x4), tmp32, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/6o/c6o4r6jftmpe3v57q672vxstcv5dxgo2flzv3rpth5d6ffo2vcfk.py
# Topologically Sorted Source Nodes: [interpolate, input_1, input_2], Original ATen: [aten.clone, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_47
#   input_2 => convolution_13
#   interpolate => clone_18
# Graph fragment:
#   %arg49_1 : Tensor "bf16[1024, 1024, 3, 3][9216, 9, 3, 1]cuda:0" = PlaceHolder[target=arg49_1]
#   %clone_18 : Tensor "f32[2, 1024, 60, 104][6389760, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_47 : Tensor "bf16[2, 1024, 60, 104][6389760, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_18, torch.bfloat16), kwargs = {})
#   %convolution_13 : Tensor "bf16[2, 1024, 60, 104][6389760, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_47, %arg49_1, %arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf93
triton_poi_fused__to_copy_clone_convolution_20 = async_compile.triton('triton_poi_fused__to_copy_clone_convolution_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused__to_copy_clone_convolution_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 37748736, 'x': 18874368}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_convolution_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (y0 + 1024*x2 + 9216*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/as/casqstv66ewt4d7473jj4cs7kiq7v554s2cosckxtx6u7lhoev2a.py
# Topologically Sorted Source Nodes: [x_29, x_30, x_31, interpolate, input_1, input_2, view_1, x_75, x_76, x_77, permute_6, x_78, x_79, x_80, float_13, normalize_11, normalized_11, mul_22, mul_23, x_81, x_82, getitem_27], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.clone, aten._to_copy, aten.view, aten.permute, aten.unsqueeze, aten.expand, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul, aten.silu, aten.slice]
# Source node to ATen node mapping:
#   float_13 => convert_element_type_48
#   getitem_27 => slice_12
#   input_1 => convert_element_type_47
#   input_2 => convolution_13
#   interpolate => clone_18
#   mul_22 => mul_26
#   mul_23 => mul_27
#   normalize_11 => clamp_min_11, div_21, expand_12, pow_23, pow_24, sum_12
#   normalized_11 => convert_element_type_49
#   permute_6 => permute_6
#   view_1 => view_8
#   x_29 => constant_pad_nd_3
#   x_30 => convolution_5
#   x_31 => add_11
#   x_75 => permute_5
#   x_76 => clone_19, expand_11, unsqueeze_12, view_9
#   x_77 => view_10
#   x_78 => clone_20
#   x_79 => view_11
#   x_80 => add_31
#   x_81 => add_32
#   x_82 => add_33, convert_element_type_50, convert_element_type_51, div_22, exp_10, neg_10
# Graph fragment:
#   %buf94 : Tensor "bf16[2, 1024, 60, 104][6389760, 1, 106496, 1024]cuda:0" = PlaceHolder[target=buf94]
#   %arg50_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg50_1]
#   %buf41 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0" = PlaceHolder[target=buf41]
#   %arg21_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %add_6 : Tensor "bf16[1, 1024, 1, 30, 52][1638400, 1600, 1638400, 52, 1]cuda:0" = PlaceHolder[target=add_6]
#   %add_31 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 1, 6389760, 106496, 1024]cuda:0" = PlaceHolder[target=add_31]
#   %sum_12 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=sum_12]
#   %arg51_1 : Tensor "bf16[1024, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg51_1]
#   %convert_element_type_51 : Tensor "bf16[1, 1024, 2, 60, 104][25559040, 24960, 6240, 104, 1]cuda:0" = PlaceHolder[target=convert_element_type_51]
#   %constant_pad_nd_3 : Tensor "bf16[1, 1024, 3, 32, 54][5308416, 5184, 1728, 54, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_7, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_5 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_3, %arg20_1, %arg21_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_11 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1597440, 52, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %add_6), kwargs = {})
#   %clone_18 : Tensor "f32[2, 1024, 60, 104][6389760, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_47 : Tensor "bf16[2, 1024, 60, 104][6389760, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_18, torch.bfloat16), kwargs = {})
#   %convolution_13 : Tensor "bf16[2, 1024, 60, 104][6389760, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_47, %arg49_1, %arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_8 : Tensor "bf16[1, 2, 1024, 60, 104][12779520, 6389760, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_13, [1, 2, 1024, 60, 104]), kwargs = {})
#   %permute_5 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3, 4]), kwargs = {})
#   %unsqueeze_12 : Tensor "bf16[1, 1024, 1, 1, 30, 52][1597440, 1560, 1597440, 1597440, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_11, 2), kwargs = {})
#   %expand_11 : Tensor "bf16[1, 1024, 8, 1, 30, 52][1597440, 1560, 0, 1597440, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_12, [1, 1024, 8, 1, 30, 52]), kwargs = {})
#   %clone_19 : Tensor "bf16[1, 1024, 8, 1, 30, 52][12779520, 12480, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_11,), kwargs = {memory_format: torch.contiguous_format})
#   %view_9 : Tensor "bf16[1, 8192, 1, 30, 52][12779520, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_19, [1, 8192, 1, 30, 52]), kwargs = {})
#   %view_10 : Tensor "bf16[1, 1024, 2, 2, 2, 1, 30, 52][12779520, 12480, 6240, 3120, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_9, [1, 1024, 2, 2, 2, 1, 30, 52]), kwargs = {})
#   %permute_6 : Tensor "bf16[1, 1024, 1, 2, 30, 2, 52, 2][12779520, 12480, 1560, 6240, 52, 3120, 1, 1560]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 1, 5, 2, 6, 3, 7, 4]), kwargs = {})
#   %clone_20 : Tensor "bf16[1, 1024, 1, 2, 30, 2, 52, 2][12779520, 12480, 12480, 6240, 208, 104, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_11 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_20, [1, 1024, 2, 60, 104]), kwargs = {})
#   %add_31 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_5, %view_11), kwargs = {})
#   %convert_element_type_48 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %pow_23 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_48, 2.0), kwargs = {})
#   %sum_12 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_23, [1], True), kwargs = {})
#   %pow_24 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_12, 0.5), kwargs = {})
#   %clamp_min_11 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_24, 1e-12), kwargs = {})
#   %expand_12 : Tensor "f32[1, 1024, 2, 60, 104][12480, 0, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_11, [1, 1024, 2, 60, 104]), kwargs = {})
#   %div_21 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_48, %expand_12), kwargs = {})
#   %convert_element_type_49 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_21, torch.bfloat16), kwargs = {})
#   %mul_26 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_49, 32.0), kwargs = {})
#   %mul_27 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %arg51_1), kwargs = {})
#   %add_32 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, 0.0), kwargs = {})
#   %convert_element_type_50 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_32, torch.float32), kwargs = {})
#   %neg_10 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_50,), kwargs = {})
#   %exp_10 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_10,), kwargs = {})
#   %add_33 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_10, 1), kwargs = {})
#   %div_22 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_50, %add_33), kwargs = {})
#   %convert_element_type_51 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_22, torch.bfloat16), kwargs = {})
#   %slice_12 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_51, 2, -2, 9223372036854775807), kwargs = {})
#   return %add_31,%sum_12,%convert_element_type_51,%slice_12
triton_red_fused__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_silu_slice_unsqueeze_view_21 = async_compile.triton('triton_red_fused__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_silu_slice_unsqueeze_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_silu_slice_unsqueeze_view_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 102236160, 'r0_': 76683264}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_silu_slice_unsqueeze_view_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12480
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 104)
    x1 = ((xindex // 104) % 60)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_3 + 1024*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (52*(x1 // 2) + 1560*r0_3 + (x0 // 2)), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (52*(x1 // 2) + 1600*r0_3 + (x0 // 2)), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tl.store(in_out_ptr0 + (r0_3 + 1024*x4), tmp8, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    x2 = xindex // 6240
    x5 = (xindex % 6240)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_3 + 1024*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr4 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.sqrt_rn(tmp12)
        tmp17 = tl.full([1, 1], 1e-12, tl.float32)
        tmp18 = triton_helpers.maximum(tmp16, tmp17)
        tmp19 = (tmp15 / tmp18)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.full([1, 1], 32.0, tl.float32)
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 * tmp23
        tmp25 = tl.full([1, 1], 0.0, tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = -tmp27
        tmp29 = libdevice.exp(tmp28)
        tmp30 = tl.full([1, 1], 1.0, tl.float32)
        tmp31 = tmp29 + tmp30
        tmp32 = (tmp27 / tmp31)
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr1 + (x4 + 24960*r0_3), tmp33, r0_mask & xmask)
        tl.store(out_ptr2 + (x5 + 6240*r0_3 + 6389760*x2), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/my/cmyq3l5zzspiwp7blfzww3q5cyckxvjujus3lu2xkz2rmr52gmjy.py
# Topologically Sorted Source Nodes: [float_13, normalize_11, normalized_11, mul_22, mul_23, x_81, x_82, x_83], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
# Source node to ATen node mapping:
#   float_13 => convert_element_type_48
#   mul_22 => mul_26
#   mul_23 => mul_27
#   normalize_11 => clamp_min_11, div_21, expand_12, pow_24
#   normalized_11 => convert_element_type_49
#   x_81 => add_32
#   x_82 => add_33, convert_element_type_50, convert_element_type_51, div_22, exp_10, neg_10
#   x_83 => cat_23
# Graph fragment:
#   %arg52_1 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0" = PlaceHolder[target=arg52_1]
#   %convert_element_type_48 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %pow_24 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_12, 0.5), kwargs = {})
#   %clamp_min_11 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_24, 1e-12), kwargs = {})
#   %expand_12 : Tensor "f32[1, 1024, 2, 60, 104][12480, 0, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_11, [1, 1024, 2, 60, 104]), kwargs = {})
#   %div_21 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_48, %expand_12), kwargs = {})
#   %convert_element_type_49 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_21, torch.bfloat16), kwargs = {})
#   %mul_26 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_49, 32.0), kwargs = {})
#   %mul_27 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %arg51_1), kwargs = {})
#   %add_32 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, 0.0), kwargs = {})
#   %convert_element_type_50 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_32, torch.float32), kwargs = {})
#   %neg_10 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_50,), kwargs = {})
#   %exp_10 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_10,), kwargs = {})
#   %add_33 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_10, 1), kwargs = {})
#   %div_22 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_50, %add_33), kwargs = {})
#   %convert_element_type_51 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 6240, 6389760, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_22, torch.bfloat16), kwargs = {})
#   %cat_23 : Tensor "bf16[1, 1024, 4, 60, 104][25559040, 24960, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg52_1, %convert_element_type_51], 2), kwargs = {})
#   return %buf97
triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_22 = async_compile.triton('triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 76677120}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12779520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 6240)
    x1 = ((xindex // 6240) % 2)
    x2 = xindex // 12480
    x3 = (xindex % 12480)
    tmp0 = tl.load(in_ptr0 + (x0 + 6240*x2 + 6389760*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x3 + 24960*x2), tmp0, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/be/cbe2ft522kxfdysae6y4sutg6zmtdzoefd6dmqbkpvrqov6zv72k.py
# Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_84 => constant_pad_nd_11
# Graph fragment:
#   %cat_23 : Tensor "bf16[1, 1024, 4, 60, 104][25559040, 24960, 6240, 104, 1]cuda:0" = PlaceHolder[target=cat_23]
#   %constant_pad_nd_11 : Tensor "bf16[1, 1024, 4, 62, 106][26918912, 26288, 6572, 106, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_23, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   return %constant_pad_nd_11
triton_poi_fused_constant_pad_nd_23 = async_compile.triton('triton_poi_fused_constant_pad_nd_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_constant_pad_nd_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 158793728}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26918912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 106) % 62)
    x0 = (xindex % 106)
    x2 = xindex // 6572
    x4 = xindex
    tmp0 = ((-1) + x1).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = ((-1) + x0).to(tl.int32)
    tmp6 = tmp5 >= tmp1
    tmp7 = tl.full([1], 104, tl.int64)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp2 & tmp4
    tmp10 = tmp9 & tmp6
    tmp11 = tmp10 & tmp8
    tmp12 = tl.load(in_ptr0 + ((-105) + x0 + 104*x1 + 6240*x2), tmp11, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/lz/clzhs5l7uepizp6pis4fqdjxncpobeqrjunl2pjyr2t2gcabktu2.py
# Topologically Sorted Source Nodes: [x_84, x_85, float_14, normalize_12, normalized_12, mul_24, mul_25, x_86, x_87], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   float_14 => convert_element_type_52
#   mul_24 => mul_28
#   mul_25 => mul_29
#   normalize_12 => clamp_min_12, div_23, expand_13, pow_25, pow_26, sum_13
#   normalized_12 => convert_element_type_53
#   x_84 => constant_pad_nd_11
#   x_85 => convolution_14
#   x_86 => add_34
#   x_87 => add_35, convert_element_type_54, convert_element_type_55, div_24, exp_11, neg_11
# Graph fragment:
#   %buf101 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=buf101]
#   %arg54_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg54_1]
#   %sum_13 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=sum_13]
#   %arg55_1 : Tensor "bf16[1024, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg55_1]
#   %constant_pad_nd_11 : Tensor "bf16[1, 1024, 4, 62, 106][26918912, 26288, 6572, 106, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_23, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_14 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_11, %arg53_1, %arg54_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convert_element_type_52 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_14, torch.float32), kwargs = {})
#   %pow_25 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_52, 2.0), kwargs = {})
#   %sum_13 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_25, [1], True), kwargs = {})
#   %pow_26 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_13, 0.5), kwargs = {})
#   %clamp_min_12 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_26, 1e-12), kwargs = {})
#   %expand_13 : Tensor "f32[1, 1024, 2, 60, 104][12480, 0, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_12, [1, 1024, 2, 60, 104]), kwargs = {})
#   %div_23 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_52, %expand_13), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_23, torch.bfloat16), kwargs = {})
#   %mul_28 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_53, 32.0), kwargs = {})
#   %mul_29 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %arg55_1), kwargs = {})
#   %add_34 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, 0.0), kwargs = {})
#   %convert_element_type_54 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_34, torch.float32), kwargs = {})
#   %neg_11 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_54,), kwargs = {})
#   %exp_11 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_11,), kwargs = {})
#   %add_35 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_11, 1), kwargs = {})
#   %div_24 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_54, %add_35), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_24, torch.bfloat16), kwargs = {})
#   return %sum_13,%convert_element_type_55
triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_24 = async_compile.triton('triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 76677120, 'r0_': 4096}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_24(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12480
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 12480*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (x0 + 12480*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.sqrt_rn(tmp6)
        tmp13 = tl.full([1, 1], 1e-12, tl.float32)
        tmp14 = triton_helpers.maximum(tmp12, tmp13)
        tmp15 = (tmp11 / tmp14)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full([1, 1], 32.0, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 * tmp19
        tmp21 = tl.full([1, 1], 0.0, tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = -tmp23
        tmp25 = libdevice.exp(tmp24)
        tmp26 = tl.full([1, 1], 1.0, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = (tmp23 / tmp27)
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr1 + (x0 + 24960*r0_1), tmp29, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/6z/c6zuigrbgjwfo6gghb57ohkw62idtv4xhwpjqr377wcz5u52eyxo.py
# Topologically Sorted Source Nodes: [x_84, x_85, float_14, normalize_12, normalized_12, mul_24, mul_25, x_86, x_87, x_89], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
# Source node to ATen node mapping:
#   float_14 => convert_element_type_52
#   mul_24 => mul_28
#   mul_25 => mul_29
#   normalize_12 => clamp_min_12, div_23, expand_13, pow_26
#   normalized_12 => convert_element_type_53
#   x_84 => constant_pad_nd_11
#   x_85 => convolution_14
#   x_86 => add_34
#   x_87 => add_35, convert_element_type_54, convert_element_type_55, div_24, exp_11, neg_11
#   x_89 => cat_24
# Graph fragment:
#   %arg56_1 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=arg56_1]
#   %constant_pad_nd_11 : Tensor "bf16[1, 1024, 4, 62, 106][26918912, 26288, 6572, 106, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_23, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_14 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_11, %arg53_1, %arg54_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convert_element_type_52 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_14, torch.float32), kwargs = {})
#   %pow_26 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_13, 0.5), kwargs = {})
#   %clamp_min_12 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_26, 1e-12), kwargs = {})
#   %expand_13 : Tensor "f32[1, 1024, 2, 60, 104][12480, 0, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_12, [1, 1024, 2, 60, 104]), kwargs = {})
#   %div_23 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_52, %expand_13), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_23, torch.bfloat16), kwargs = {})
#   %mul_28 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_53, 32.0), kwargs = {})
#   %mul_29 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %arg55_1), kwargs = {})
#   %add_34 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, 0.0), kwargs = {})
#   %convert_element_type_54 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_34, torch.float32), kwargs = {})
#   %neg_11 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_54,), kwargs = {})
#   %exp_11 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_11,), kwargs = {})
#   %add_35 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_11, 1), kwargs = {})
#   %div_24 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_54, %add_35), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_24, torch.bfloat16), kwargs = {})
#   %cat_24 : Tensor "bf16[1, 1024, 4, 60, 104][25559040, 24960, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg56_1, %convert_element_type_55], 2), kwargs = {})
#   return %buf103
triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25 = async_compile.triton('triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 76677120}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12779520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 12480)
    x1 = xindex // 12480
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + 24960*x1), tmp0, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/55/c55aiex6nslh7gedqmefomiqx6ojsn675oizlnxz6yk3nnorzs6k.py
# Topologically Sorted Source Nodes: [x_90, x_91, x_92, float_15, normalize_13, normalized_13, mul_26, mul_27, x_93, x_94], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.silu]
# Source node to ATen node mapping:
#   float_15 => convert_element_type_56
#   mul_26 => mul_30
#   mul_27 => mul_31
#   normalize_13 => clamp_min_13, div_25, expand_14, pow_27, pow_28, sum_14
#   normalized_13 => convert_element_type_57
#   x_90 => constant_pad_nd_12
#   x_91 => convolution_15
#   x_92 => add_36
#   x_93 => add_37
#   x_94 => add_38, convert_element_type_58, convert_element_type_59, div_26, exp_12, neg_12
# Graph fragment:
#   %buf107 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=buf107]
#   %arg58_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %add_31 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 1, 6389760, 106496, 1024]cuda:0" = PlaceHolder[target=add_31]
#   %sum_14 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=sum_14]
#   %arg59_1 : Tensor "bf16[1024, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg59_1]
#   %convert_element_type_58 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=convert_element_type_58]
#   %constant_pad_nd_12 : Tensor "bf16[1, 1024, 4, 62, 106][26918912, 26288, 6572, 106, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_24, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_15 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_12, %arg57_1, %arg58_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_36 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_15, %add_31), kwargs = {})
#   %convert_element_type_56 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_36, torch.float32), kwargs = {})
#   %pow_27 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_56, 2.0), kwargs = {})
#   %sum_14 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_27, [1], True), kwargs = {})
#   %pow_28 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_14, 0.5), kwargs = {})
#   %clamp_min_13 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_28, 1e-12), kwargs = {})
#   %expand_14 : Tensor "f32[1, 1024, 2, 60, 104][12480, 0, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_13, [1, 1024, 2, 60, 104]), kwargs = {})
#   %div_25 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_56, %expand_14), kwargs = {})
#   %convert_element_type_57 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_25, torch.bfloat16), kwargs = {})
#   %mul_30 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_57, 32.0), kwargs = {})
#   %mul_31 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %arg59_1), kwargs = {})
#   %add_37 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, 0.0), kwargs = {})
#   %convert_element_type_58 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_37, torch.float32), kwargs = {})
#   %neg_12 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_58,), kwargs = {})
#   %exp_12 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_12,), kwargs = {})
#   %add_38 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_12, 1), kwargs = {})
#   %div_26 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_58, %add_38), kwargs = {})
#   %convert_element_type_59 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_26, torch.bfloat16), kwargs = {})
#   return %sum_14,%convert_element_type_58,%convert_element_type_59
triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_26 = async_compile.triton('triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 76677120, 'r0_': 25563136}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12480
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 12480*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 1024*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (x0 + 12480*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r0_1 + 1024*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tmp10 + tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.sqrt_rn(tmp8)
        tmp17 = tl.full([1, 1], 1e-12, tl.float32)
        tmp18 = triton_helpers.maximum(tmp16, tmp17)
        tmp19 = (tmp15 / tmp18)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.full([1, 1], 32.0, tl.float32)
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 * tmp23
        tmp25 = tl.full([1, 1], 0.0, tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = -tmp27
        tmp29 = libdevice.exp(tmp28)
        tmp30 = tl.full([1, 1], 1.0, tl.float32)
        tmp31 = tmp29 + tmp30
        tmp32 = (tmp27 / tmp31)
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr2 + (x0 + 24960*r0_1), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/nk/cnk5jozgg374k7feft7dyl6fk45wggmud5r62zzwjflxmnqaku23.py
# Topologically Sorted Source Nodes: [x_90, x_91, x_92, x_102, x_103, x_104, float_17, normalize_15, normalized_15, mul_30, mul_31, x_105, x_106], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.silu]
# Source node to ATen node mapping:
#   float_17 => convert_element_type_64
#   mul_30 => mul_34
#   mul_31 => mul_35
#   normalize_15 => clamp_min_15, div_29, expand_16, pow_31, pow_32, sum_16
#   normalized_15 => convert_element_type_65
#   x_102 => constant_pad_nd_14
#   x_103 => convolution_17
#   x_104 => add_41
#   x_105 => add_42
#   x_106 => add_43, convert_element_type_66, convert_element_type_67, div_30, exp_14, neg_14
#   x_90 => constant_pad_nd_12
#   x_91 => convolution_15
#   x_92 => add_36
# Graph fragment:
#   %buf120 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=buf120]
#   %arg66_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg66_1]
#   %buf107 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=buf107]
#   %arg58_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %add_31 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 1, 6389760, 106496, 1024]cuda:0" = PlaceHolder[target=add_31]
#   %add_41 : Tensor "bf16[1, 1024, 2, 60, 104][12845056, 12544, 6272, 104, 1]cuda:0" = PlaceHolder[target=add_41]
#   %sum_16 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=sum_16]
#   %arg67_1 : Tensor "bf16[1024, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=arg67_1]
#   %constant_pad_nd_12 : Tensor "bf16[1, 1024, 4, 62, 106][26918912, 26288, 6572, 106, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_24, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_15 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_12, %arg57_1, %arg58_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_36 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_15, %add_31), kwargs = {})
#   %constant_pad_nd_14 : Tensor "bf16[1, 1024, 4, 62, 106][26918912, 26288, 6572, 106, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_26, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_17 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_14, %arg65_1, %arg66_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_41 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %add_36), kwargs = {})
#   %convert_element_type_64 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_41, torch.float32), kwargs = {})
#   %pow_31 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_64, 2.0), kwargs = {})
#   %sum_16 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_31, [1], True), kwargs = {})
#   %pow_32 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %clamp_min_15 : Tensor "f32[1, 1, 2, 60, 104][12480, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_32, 1e-12), kwargs = {})
#   %expand_16 : Tensor "f32[1, 1024, 2, 60, 104][12480, 0, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_15, [1, 1024, 2, 60, 104]), kwargs = {})
#   %div_29 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_64, %expand_16), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_29, torch.bfloat16), kwargs = {})
#   %mul_34 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_65, 32.0), kwargs = {})
#   %mul_35 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %arg67_1), kwargs = {})
#   %add_42 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, 0.0), kwargs = {})
#   %convert_element_type_66 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_42, torch.float32), kwargs = {})
#   %neg_14 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_66,), kwargs = {})
#   %exp_14 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_14,), kwargs = {})
#   %add_43 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_14, 1), kwargs = {})
#   %div_30 : Tensor "f32[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_66, %add_43), kwargs = {})
#   %convert_element_type_67 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_30, torch.bfloat16), kwargs = {})
#   return %add_41,%sum_16,%convert_element_type_67
triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_27 = async_compile.triton('triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 153354240, 'r0_': 25565184}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12480
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 6240)
    x1 = xindex // 6240
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x3 + 12480*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x3 + 12480*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr4 + (r0_2 + 1024*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tl.store(out_ptr0 + (x0 + 6272*x1 + 12544*r0_2), tmp8, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp14 = tl.load(out_ptr0 + (x0 + 6272*x1 + 12544*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.sqrt_rn(tmp12)
        tmp17 = tl.full([1, 1], 1e-12, tl.float32)
        tmp18 = triton_helpers.maximum(tmp16, tmp17)
        tmp19 = (tmp15 / tmp18)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.full([1, 1], 32.0, tl.float32)
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 * tmp23
        tmp25 = tl.full([1, 1], 0.0, tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = -tmp27
        tmp29 = libdevice.exp(tmp28)
        tmp30 = tl.full([1, 1], 1.0, tl.float32)
        tmp31 = tmp29 + tmp30
        tmp32 = (tmp27 / tmp31)
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr2 + (x3 + 24960*r0_2), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/wh/cwhhlv3274otmdk3c4cxfdchaetqfultheetjluvynqpgsik7lpg.py
# Topologically Sorted Source Nodes: [x_114, x_115, x_116], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   x_114 => constant_pad_nd_16
#   x_115 => convolution_19
#   x_116 => add_46
# Graph fragment:
#   %buf133 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=buf133]
#   %arg74_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg74_1]
#   %add_41 : Tensor "bf16[1, 1024, 2, 60, 104][12845056, 12544, 6272, 104, 1]cuda:0" = PlaceHolder[target=add_41]
#   %constant_pad_nd_16 : Tensor "bf16[1, 1024, 4, 62, 106][26918912, 26288, 6572, 106, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%cat_28, [1, 1, 1, 1, 0, 0], 0.0), kwargs = {})
#   %convolution_19 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_16, %arg73_1, %arg74_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_46 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_19, %add_41), kwargs = {})
#   return %add_46
triton_poi_fused_add_constant_pad_nd_convolution_28 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 102238208}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_28(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12779520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x2 = xindex // 12480
    x0 = (xindex % 6240)
    x4 = xindex // 6240
    x5 = (xindex % 12480)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 6272*x4), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x5 + 24960*x2), tmp4, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/bw/cbw4yegf5lo2zqtcukmpav3v36ceuvjofpkphlh7qy6c3v7kk777.py
# Topologically Sorted Source Nodes: [x_119, x_120, getitem_34, getitem_35, x_121, x_122, permute_7, x_123, float_19, interpolate_1, input_3], Original ATen: [aten.convolution, aten.view, aten.select, aten.stack, aten.permute, aten._to_copy, aten.arange, aten.add, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   float_19 => convert_element_type_72
#   getitem_34 => select_13
#   getitem_35 => select_14
#   input_3 => convert_element_type_77
#   interpolate_1 => _unsafe_index_1, add_47, add_48, add_49, add_50, clone_32, convert_element_type_73, convert_element_type_74, convert_element_type_75, convert_element_type_76, iota_2, iota_3, mul_38, mul_39, mul_40, mul_41, unsqueeze_13
#   permute_7 => permute_7
#   x_119 => convolution_20
#   x_120 => view_12
#   x_121 => cat_30, view_13
#   x_122 => view_14
#   x_123 => view_15
# Graph fragment:
#   %buf137 : Tensor "bf16[1, 2048, 2, 60, 104][25559040, 12480, 6240, 104, 1]cuda:0" = PlaceHolder[target=buf137]
#   %arg77_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg77_1]
#   %_unsafe_index_1 : Tensor "f32[4, 1024, 120, 208][25559040, 1, 212992, 1024]cuda:0" = PlaceHolder[target=_unsafe_index_1]
#   %convolution_20 : Tensor "bf16[1, 2048, 2, 60, 104][25559040, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_29, %arg76_1, %arg77_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %view_12 : Tensor "bf16[1, 2, 1024, 2, 60, 104][25559040, 12779520, 12480, 6240, 104, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [1, 2, 1024, 2, 60, 104]), kwargs = {})
#   %select_13 : Tensor "bf16[1, 1024, 2, 60, 104][25559040, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_12, 1, 0), kwargs = {})
#   %select_14 : Tensor "bf16[1, 1024, 2, 60, 104][25559040, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_12, 1, 1), kwargs = {})
#   %cat_30 : Tensor "bf16[1, 1024, 2, 120, 104][25559040, 24960, 12480, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%select_13, %select_14], 3), kwargs = {})
#   %view_13 : Tensor "bf16[1, 1024, 2, 2, 60, 104][25559040, 24960, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_30, [1, 1024, 2, 2, 60, 104]), kwargs = {})
#   %view_14 : Tensor "bf16[1, 1024, 4, 60, 104][25559040, 24960, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_13, [1, 1024, 4, 60, 104]), kwargs = {})
#   %permute_7 : Tensor "bf16[1, 4, 1024, 60, 104][25559040, 6240, 24960, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [0, 2, 1, 3, 4]), kwargs = {})
#   %view_15 : Tensor "bf16[4, 1024, 60, 104][6240, 24960, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_7, [4, 1024, 60, 104]), kwargs = {})
#   %convert_element_type_72 : Tensor "f32[4, 1024, 60, 104][6240, 24960, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_15, torch.float32), kwargs = {})
#   %iota_2 : Tensor "i64[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (120,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_38 : Tensor "i64[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_47 : Tensor "i64[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, 0), kwargs = {})
#   %convert_element_type_73 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_47, torch.float32), kwargs = {})
#   %add_48 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_73, 0.5), kwargs = {})
#   %mul_39 : Tensor "f32[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, 0.5), kwargs = {})
#   %convert_element_type_74 : Tensor "i64[120][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_39, torch.int64), kwargs = {})
#   %unsqueeze_13 : Tensor "i64[120, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_74, -1), kwargs = {})
#   %iota_3 : Tensor "i64[208][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (208,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_40 : Tensor "i64[208][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_49 : Tensor "i64[208][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_40, 0), kwargs = {})
#   %convert_element_type_75 : Tensor "f32[208][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_49, torch.float32), kwargs = {})
#   %add_50 : Tensor "f32[208][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_75, 0.5), kwargs = {})
#   %mul_41 : Tensor "f32[208][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_50, 0.5), kwargs = {})
#   %convert_element_type_76 : Tensor "i64[208][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_41, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[4, 1024, 120, 208][24960, 99840, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_72, [None, None, %unsqueeze_13, %convert_element_type_76]), kwargs = {})
#   %clone_32 : Tensor "f32[4, 1024, 120, 208][25559040, 24960, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_77 : Tensor "bf16[4, 1024, 120, 208][25559040, 24960, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_32, torch.bfloat16), kwargs = {})
#   return %_unsafe_index_1,%convert_element_type_77
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_29 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 408948736}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_29(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102236160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = ((xindex // 212992) % 120)
    x1 = ((xindex // 1024) % 208)
    x3 = xindex // 25559040
    x0 = (xindex % 1024)
    x4 = xindex
    tmp0 = (x2).to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4.to(tl.int32)
    tmp6 = (x1).to(tl.float32)
    tmp7 = tmp6 + tmp2
    tmp8 = tmp7 * tmp2
    tmp9 = tmp8.to(tl.int32)
    tmp10 = (tmp5 + 60*((x3 % 2))).to(tl.int32)
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = (tmp5 + 60*((x3 % 2))).to(tl.int64)
    tmp14 = (tmp13).to(tl.int64)
    tmp15 = tl.full([1], 60, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tl.load(in_ptr0 + (tmp9 + 104*(tmp5 + 60*((x3 % 2))) + 6240*(x3 // 2) + 12480*x0), tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr1 + (x0), tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tmp10 >= tmp15
    tmp23 = tl.full([1], 120, tl.int64)
    tmp24 = tmp10 < tmp23
    tmp25 = tl.load(in_ptr0 + (12779520 + tmp9 + 104*((-60) + tmp5 + 60*((x3 % 2))) + 6240*(x3 // 2) + 12480*x0), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (1024 + x0), tmp22, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp22, tmp27, tmp28)
    tmp30 = tl.where(tmp16, tmp21, tmp29)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr1 + (x4), tmp32, None)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/al/callmgx32shn7cxq42a4tmgphgcildepcst6w6hgdbg2hvsqr4iu.py
# Topologically Sorted Source Nodes: [interpolate_1, input_3, input_4, view_4, x_124, x_125, x_126, permute_9, x_127, x_128, x_129], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.view, aten.permute, aten.unsqueeze, aten.expand, aten.add]
# Source node to ATen node mapping:
#   input_3 => convert_element_type_77
#   input_4 => convolution_21
#   interpolate_1 => clone_32
#   permute_9 => permute_9
#   view_4 => view_16
#   x_124 => permute_8
#   x_125 => clone_33, expand_18, unsqueeze_14, view_17
#   x_126 => view_18
#   x_127 => clone_34
#   x_128 => view_19
#   x_129 => add_51
# Graph fragment:
#   %buf141 : Tensor "bf16[4, 1024, 120, 208][25559040, 1, 212992, 1024]cuda:0" = PlaceHolder[target=buf141]
#   %arg79_1 : Tensor "bf16[1024][1]cuda:0" = PlaceHolder[target=arg79_1]
#   %add_31 : Tensor "bf16[1, 1024, 2, 60, 104][12779520, 1, 6389760, 106496, 1024]cuda:0" = PlaceHolder[target=add_31]
#   %clone_32 : Tensor "f32[4, 1024, 120, 208][25559040, 24960, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_77 : Tensor "bf16[4, 1024, 120, 208][25559040, 24960, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_32, torch.bfloat16), kwargs = {})
#   %convolution_21 : Tensor "bf16[4, 1024, 120, 208][25559040, 24960, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_77, %arg78_1, %arg79_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_16 : Tensor "bf16[1, 4, 1024, 120, 208][102236160, 25559040, 24960, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_21, [1, 4, 1024, 120, 208]), kwargs = {})
#   %permute_8 : Tensor "bf16[1, 1024, 4, 120, 208][102236160, 24960, 25559040, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_16, [0, 2, 1, 3, 4]), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[1, 1024, 1, 2, 60, 104][12779520, 6240, 12779520, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_31, 2), kwargs = {})
#   %expand_18 : Tensor "bf16[1, 1024, 8, 2, 60, 104][12779520, 6240, 0, 6389760, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_14, [1, 1024, 8, 2, 60, 104]), kwargs = {})
#   %clone_33 : Tensor "bf16[1, 1024, 8, 2, 60, 104][102236160, 99840, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_18,), kwargs = {memory_format: torch.contiguous_format})
#   %view_17 : Tensor "bf16[1, 8192, 2, 60, 104][102236160, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_33, [1, 8192, 2, 60, 104]), kwargs = {})
#   %view_18 : Tensor "bf16[1, 1024, 2, 2, 2, 2, 60, 104][102236160, 99840, 49920, 24960, 12480, 6240, 104, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_17, [1, 1024, 2, 2, 2, 2, 60, 104]), kwargs = {})
#   %permute_9 : Tensor "bf16[1, 1024, 2, 2, 60, 2, 104, 2][102236160, 99840, 6240, 49920, 104, 24960, 1, 12480]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 1, 5, 2, 6, 3, 7, 4]), kwargs = {})
#   %clone_34 : Tensor "bf16[1, 1024, 2, 2, 60, 2, 104, 2][102236160, 99840, 49920, 24960, 416, 208, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "bf16[1, 1024, 4, 120, 208][102236160, 99840, 24960, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_34, [1, 1024, 4, 120, 208]), kwargs = {})
#   %add_51 : Tensor "bf16[1, 1024, 4, 120, 208][102236160, 24960, 25559040, 208, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_8, %view_19), kwargs = {})
#   return %add_51
triton_poi_fused__to_copy_add_clone_convolution_expand_permute_unsqueeze_view_30 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_expand_permute_unsqueeze_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_expand_permute_unsqueeze_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 204472320, 'x': 434505728}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_expand_permute_unsqueeze_view_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 24960
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    x2 = (xindex % 208)
    x3 = xindex // 208
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024*x4 + 25559040*y1), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (104*(x3 // 2) + 6272*(y1 // 2) + 12544*y0 + (x2 // 2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x4 + 24960*y5), tmp4, xmask)
''', device_str='cuda')


# kernel path: ${TMPDIR}/torchinductor_uid_1001/fq/cfqhvuuohh6zi7lxzissavyy6owwbf4zxatohhbgjhai4xgbyzhc.py
# Topologically Sorted Source Nodes: [getitem_1, unsqueeze, getitem, cache_x_1], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
# Source node to ATen node mapping:
#   cache_x_1 => cat
#   getitem => slice_1
#   getitem_1 => select
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg2_1 : Tensor "bf16[1, 1024, 2, 30, 52][3194880, 3120, 1560, 52, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type_3 : Tensor "bf16[1, 1024, 1, 30, 52][4792320, 4680, 1560, 52, 1]cuda:0" = PlaceHolder[target=convert_element_type_3]
#   %select : Tensor "bf16[1, 1024, 30, 52][3194880, 3120, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg2_1, 2, -1), kwargs = {})
#   %unsqueeze : Tensor "bf16[1, 1024, 1, 30, 52][3194880, 3120, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select, 2), kwargs = {})
#   %slice_1 : Tensor "bf16[1, 1024, 1, 30, 52][1597440, 1560, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_3, 2, -2, 9223372036854775807), kwargs = {})
#   %cat : Tensor "bf16[1, 1024, 2, 30, 52][3194880, 3120, 1560, 52, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %slice_1], 2), kwargs = {})
#   return %cat
triton_poi_fused_cat_select_slice_unsqueeze_31 = async_compile.triton('triton_poi_fused_cat_select_slice_unsqueeze_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_cat_select_slice_unsqueeze_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 19169280}, 'backend_hash': '766E63A602A73A3AEE5C1B46A9AA11C08FF33942F1973B11586C10EA05BA7B4C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_select_slice_unsqueeze_31(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3194880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 1560) % 2)
    x0 = (xindex % 1560)
    x2 = xindex // 3120
    x3 = xindex
    tmp0 = (x1).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x1).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (1560 + x0 + 3120*x2), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp0 >= tmp5
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (x0 + 4680*x2), tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp6, tmp7, tmp11)
    tl.store(out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((1, 1, 1, 30, 52, 8), (12544, 12544, 12544, 52, 1, 1568), torch.float32)
            # Topologically Sorted Source Nodes: [float_1, normalize], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_linalg_vector_norm_0.run(arg0_1, buf0, 12480, 128, stream=raw_stream0)
            buf1 = empty_strided_cuda((1, 1, 1, 30, 52), (1568, 1568, 1568, 52, 1), torch.float32)
            # Topologically Sorted Source Nodes: [float_1, normalize], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf0, buf1, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg2_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg2_1 = copy_if_misaligned(arg2_1)
            buf4 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf2 = reinterpret_tensor(buf4, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [float_1, normalize, normalized, mul, mul_1, x, x_1, x_2], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg2_1, buf2, 3194880, stream=raw_stream0)
            assert_size_stride(arg1_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf3 = reinterpret_tensor(buf4, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [float_1, normalize, normalized, mul, mul_1, x, x_1], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_3.run(arg0_1, buf1, arg1_1, buf3, 1597440, stream=raw_stream0)
            del arg1_1
            buf5 = empty_strided_cuda((1, 1024, 3, 32, 54), (5308416, 5184, 1728, 54, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf4, buf5, 5308416, stream=raw_stream0)
            del buf2
            assert_size_stride(arg3_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf6 = extern_kernels.convolution(buf5, arg3_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf6, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg3_1
            assert_size_stride(arg4_1, (1024, ), (1, ), 'input')
            buf7 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_4, float_2, normalize_1], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_constant_pad_nd_convolution_linalg_vector_norm_5.run(buf6, arg4_1, buf7, 12480, 128, stream=raw_stream0)
            buf8 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [x_3, x_4, float_2, normalize_1], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf7, buf8, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg6_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg6_1 = copy_if_misaligned(arg6_1)
            buf11 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf9 = reinterpret_tensor(buf11, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_3, x_4, float_2, normalize_1, normalized_1, mul_2, mul_3, x_5, x_6, x_8], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg6_1, buf9, 3194880, stream=raw_stream0)
            assert_size_stride(arg5_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf10 = reinterpret_tensor(buf11, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [x_3, x_4, float_2, normalize_1, normalized_1, mul_2, mul_3, x_5, x_6], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_6.run(buf6, arg4_1, buf8, arg5_1, buf10, 1597440, stream=raw_stream0)
            del arg4_1
            del arg5_1
            buf12 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf11, buf12, 5308416, stream=raw_stream0)
            del buf9
            assert_size_stride(arg7_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_9, x_10], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf13 = extern_kernels.convolution(buf12, arg7_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf13, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg7_1
            assert_size_stride(arg8_1, (1024, ), (1, ), 'input')
            buf14 = reinterpret_tensor(buf7, (1, 1, 30, 52, 8), (12544, 12544, 52, 1, 1568), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_permute_view_7.run(buf13, arg8_1, arg0_1, buf14, 12480, 128, stream=raw_stream0)
            buf15 = reinterpret_tensor(buf8, (1, 1, 30, 52), (1568, 1568, 52, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf14, buf15, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg9_1, (1024, 1, 1), (1, 1, 1), 'input')
            buf16 = reinterpret_tensor(buf6, (1, 1024, 30, 52), (1597440, 1, 53248, 1024), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_view_8.run(buf13, arg8_1, arg0_1, buf15, arg9_1, buf16, 1024, 1560, stream=raw_stream0)
            del arg9_1
            assert_size_stride(arg10_1, (3072, 1024, 1, 1), (1024, 1, 1, 1), 'input')
            # Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13, qkv], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul]
            buf17 = extern_kernels.convolution(buf16, arg10_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf17, (1, 3072, 30, 52), (4792320, 1, 159744, 3072), 'torch.ops.aten.convolution.default')
            del arg10_1
            assert_size_stride(arg11_1, (3072, ), (1, ), 'input')
            buf18 = reinterpret_tensor(buf16, (1, 1, 1560, 1024), (1597440, 1597440, 1024, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13, qkv, qkv_1, permute_1, qkv_2, chunk, x_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.clone, aten.split, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_9.run(buf17, arg11_1, buf18, 1597440, stream=raw_stream0)
            buf19 = empty_strided_cuda((1, 1, 1560, 1024), (1597440, 1597440, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13, qkv, qkv_1, permute_1, qkv_2, chunk, x_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.clone, aten.split, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_10.run(buf17, arg11_1, buf19, 1597440, stream=raw_stream0)
            buf20 = empty_strided_cuda((1, 1, 1560, 1024), (1597440, 1597440, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13, qkv, qkv_1, permute_1, qkv_2, chunk, x_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.clone, aten.split, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_split_view_11.run(buf17, arg11_1, buf20, 1597440, stream=raw_stream0)
            del arg11_1
            # Topologically Sorted Source Nodes: [x_9, x_10, x_11, permute, x_12, float_3, normalize_2, normalized_2, mul_4, mul_5, x_13, qkv, qkv_1, permute_1, qkv_2, chunk, x_14], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.permute, aten.view, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.clone, aten.split, aten._scaled_dot_product_efficient_attention]
            buf21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf18, buf19, buf20, None, False)
            del buf18
            del buf19
            del buf20
            buf22 = buf21[0]
            assert_size_stride(buf22, (1, 1, 1560, 1024), (1597440, 1024, 1024, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf22, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf21
            assert_size_stride(arg12_1, (1024, 1024, 1, 1), (1024, 1, 1, 1), 'input')
            # Topologically Sorted Source Nodes: [squeeze, permute_2, x_15, x_16], Original ATen: [aten.squeeze, aten.permute, aten.view, aten.convolution]
            buf26 = extern_kernels.convolution(reinterpret_tensor(buf22, (1, 1024, 30, 52), (1024, 1, 53248, 1024), 0), arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf26, (1, 1024, 30, 52), (1597440, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg12_1
            del buf22
            assert_size_stride(arg13_1, (1024, ), (1, ), 'input')
            buf27 = empty_strided_cuda((1, 1024, 1, 30, 52), (1638400, 1600, 1638400, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_9, x_10, x_11, squeeze, permute_2, x_15, x_16, x_17, x_18, x_19], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.squeeze, aten.permute, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_constant_pad_nd_convolution_permute_squeeze_view_12.run(buf26, arg13_1, buf13, arg8_1, arg0_1, buf27, 1597440, stream=raw_stream0)
            del arg0_1
            del arg13_1
            del arg8_1
            del buf13
            del buf26
            buf28 = reinterpret_tensor(buf14, (1, 1, 1, 30, 52, 8), (12544, 12544, 12544, 52, 1, 1568), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [float_4, normalize_3], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_linalg_vector_norm_13.run(buf27, buf28, 12480, 128, stream=raw_stream0)
            buf29 = reinterpret_tensor(buf15, (1, 1, 1, 30, 52), (1568, 1568, 1568, 52, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [float_4, normalize_3], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf28, buf29, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg15_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg15_1 = copy_if_misaligned(arg15_1)
            buf32 = reinterpret_tensor(buf17, (1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), 0); del buf17  # reuse
            buf30 = reinterpret_tensor(buf32, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [float_4, normalize_3, normalized_3, mul_6, mul_7, x_20, x_21, x_22], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg15_1, buf30, 3194880, stream=raw_stream0)
            assert_size_stride(arg14_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf31 = reinterpret_tensor(buf32, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [float_4, normalize_3, normalized_3, mul_6, mul_7, x_20, x_21], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_14.run(buf27, buf29, arg14_1, buf31, 1597440, stream=raw_stream0)
            del arg14_1
            buf33 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf32, buf33, 5308416, stream=raw_stream0)
            del buf30
            assert_size_stride(arg16_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_23, x_24], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf34 = extern_kernels.convolution(buf33, arg16_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf34, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg16_1
            assert_size_stride(arg17_1, (1024, ), (1, ), 'input')
            buf35 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [x_23, x_24, float_5, normalize_4], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_constant_pad_nd_convolution_linalg_vector_norm_5.run(buf34, arg17_1, buf35, 12480, 128, stream=raw_stream0)
            buf36 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [x_23, x_24, float_5, normalize_4], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf35, buf36, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg19_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg19_1 = copy_if_misaligned(arg19_1)
            buf39 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf37 = reinterpret_tensor(buf39, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_23, x_24, float_5, normalize_4, normalized_4, mul_8, mul_9, x_25, x_26, x_28], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg19_1, buf37, 3194880, stream=raw_stream0)
            assert_size_stride(arg18_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf38 = reinterpret_tensor(buf39, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [x_23, x_24, float_5, normalize_4, normalized_4, mul_8, mul_9, x_25, x_26], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_6.run(buf34, arg17_1, buf36, arg18_1, buf38, 1597440, stream=raw_stream0)
            del arg17_1
            del arg18_1
            del buf34
            buf40 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf39, buf40, 5308416, stream=raw_stream0)
            del buf37
            assert_size_stride(arg20_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_29, x_30], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf41 = extern_kernels.convolution(buf40, arg20_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf41, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg20_1
            assert_size_stride(arg21_1, (1024, ), (1, ), 'input')
            buf42 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [x_29, x_30, x_31, float_6, normalize_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_15.run(buf41, arg21_1, buf27, buf42, 12480, 128, stream=raw_stream0)
            buf43 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [x_29, x_30, x_31, float_6, normalize_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf42, buf43, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg22_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf47 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf46 = reinterpret_tensor(buf47, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [x_29, x_30, x_31, float_6, normalize_5, normalized_5, mul_10, mul_11, x_32, x_33], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_16.run(buf41, arg21_1, buf27, buf43, arg22_1, buf46, 1597440, stream=raw_stream0)
            del arg22_1
            assert_size_stride(arg23_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg23_1 = copy_if_misaligned(arg23_1)
            buf45 = reinterpret_tensor(buf47, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_33, x_34], Original ATen: [aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg23_1, buf45, 3194880, stream=raw_stream0)
            buf48 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf47, buf48, 5308416, stream=raw_stream0)
            del buf45
            assert_size_stride(arg24_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_35, x_36], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf49 = extern_kernels.convolution(buf48, arg24_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf49, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg24_1
            assert_size_stride(arg25_1, (1024, ), (1, ), 'input')
            buf50 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [x_35, x_36, float_7, normalize_6], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_constant_pad_nd_convolution_linalg_vector_norm_5.run(buf49, arg25_1, buf50, 12480, 128, stream=raw_stream0)
            buf51 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [x_35, x_36, float_7, normalize_6], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf50, buf51, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg27_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg27_1 = copy_if_misaligned(arg27_1)
            buf54 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf52 = reinterpret_tensor(buf54, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_35, x_36, float_7, normalize_6, normalized_6, mul_12, mul_13, x_37, x_38, x_40], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg27_1, buf52, 3194880, stream=raw_stream0)
            assert_size_stride(arg26_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf53 = reinterpret_tensor(buf54, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [x_35, x_36, float_7, normalize_6, normalized_6, mul_12, mul_13, x_37, x_38], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_6.run(buf49, arg25_1, buf51, arg26_1, buf53, 1597440, stream=raw_stream0)
            del arg25_1
            del arg26_1
            del buf49
            buf55 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf54, buf55, 5308416, stream=raw_stream0)
            del buf52
            assert_size_stride(arg28_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_41, x_42], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf56 = extern_kernels.convolution(buf55, arg28_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf56, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg28_1
            assert_size_stride(arg29_1, (1024, ), (1, ), 'input')
            buf57 = empty_strided_cuda((1, 1024, 1, 30, 52), (1638400, 1600, 1638400, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_29, x_30, x_31, x_41, x_42, x_43], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_constant_pad_nd_convolution_17.run(buf56, arg29_1, buf41, arg21_1, buf27, buf57, 1597440, stream=raw_stream0)
            del arg29_1
            del buf56
            buf58 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [float_8, normalize_7], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_linalg_vector_norm_13.run(buf57, buf58, 12480, 128, stream=raw_stream0)
            buf59 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [float_8, normalize_7], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf58, buf59, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg31_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg31_1 = copy_if_misaligned(arg31_1)
            buf62 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf60 = reinterpret_tensor(buf62, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [float_8, normalize_7, normalized_7, mul_14, mul_15, x_44, x_45, x_46], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg31_1, buf60, 3194880, stream=raw_stream0)
            assert_size_stride(arg30_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf61 = reinterpret_tensor(buf62, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [float_8, normalize_7, normalized_7, mul_14, mul_15, x_44, x_45], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_14.run(buf57, buf59, arg30_1, buf61, 1597440, stream=raw_stream0)
            del arg30_1
            buf63 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf62, buf63, 5308416, stream=raw_stream0)
            del buf60
            assert_size_stride(arg32_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_47, x_48], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf64 = extern_kernels.convolution(buf63, arg32_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf64, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg32_1
            assert_size_stride(arg33_1, (1024, ), (1, ), 'input')
            buf65 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [x_47, x_48, float_9, normalize_8], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_constant_pad_nd_convolution_linalg_vector_norm_5.run(buf64, arg33_1, buf65, 12480, 128, stream=raw_stream0)
            buf66 = buf59; del buf59  # reuse
            # Topologically Sorted Source Nodes: [x_47, x_48, float_9, normalize_8], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf65, buf66, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg35_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg35_1 = copy_if_misaligned(arg35_1)
            buf69 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf67 = reinterpret_tensor(buf69, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_47, x_48, float_9, normalize_8, normalized_8, mul_16, mul_17, x_49, x_50, x_52], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg35_1, buf67, 3194880, stream=raw_stream0)
            assert_size_stride(arg34_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf68 = reinterpret_tensor(buf69, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [x_47, x_48, float_9, normalize_8, normalized_8, mul_16, mul_17, x_49, x_50], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_6.run(buf64, arg33_1, buf66, arg34_1, buf68, 1597440, stream=raw_stream0)
            del arg33_1
            del arg34_1
            del buf64
            buf70 = buf63; del buf63  # reuse
            # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf69, buf70, 5308416, stream=raw_stream0)
            del buf67
            assert_size_stride(arg36_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_53, x_54], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf71 = extern_kernels.convolution(buf70, arg36_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf71, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg36_1
            assert_size_stride(arg37_1, (1024, ), (1, ), 'input')
            buf72 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [x_53, x_54, x_55, float_10, normalize_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_constant_pad_nd_convolution_linalg_vector_norm_15.run(buf71, arg37_1, buf57, buf72, 12480, 128, stream=raw_stream0)
            buf73 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [x_53, x_54, x_55, float_10, normalize_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf72, buf73, 1560, 8, stream=raw_stream0)
            assert_size_stride(arg38_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf77 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf76 = reinterpret_tensor(buf77, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [x_53, x_54, x_55, float_10, normalize_9, normalized_9, mul_18, mul_19, x_56, x_57], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_16.run(buf71, arg37_1, buf57, buf73, arg38_1, buf76, 1597440, stream=raw_stream0)
            del arg38_1
            assert_size_stride(arg39_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg39_1 = copy_if_misaligned(arg39_1)
            buf75 = reinterpret_tensor(buf77, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_57, x_58], Original ATen: [aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg39_1, buf75, 3194880, stream=raw_stream0)
            buf78 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf77, buf78, 5308416, stream=raw_stream0)
            del buf75
            assert_size_stride(arg40_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_59, x_60], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf79 = extern_kernels.convolution(buf78, arg40_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf79, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg40_1
            assert_size_stride(arg41_1, (1024, ), (1, ), 'input')
            buf80 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [x_59, x_60, float_11, normalize_10], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_constant_pad_nd_convolution_linalg_vector_norm_5.run(buf79, arg41_1, buf80, 12480, 128, stream=raw_stream0)
            buf81 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [x_59, x_60, float_11, normalize_10], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_linalg_vector_norm_1.run(buf80, buf81, 1560, 8, stream=raw_stream0)
            del buf80
            assert_size_stride(arg43_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg43_1 = copy_if_misaligned(arg43_1)
            buf84 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf82 = reinterpret_tensor(buf84, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_59, x_60, float_11, normalize_10, normalized_10, mul_20, mul_21, x_61, x_62, x_64], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg43_1, buf82, 3194880, stream=raw_stream0)
            assert_size_stride(arg42_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf83 = reinterpret_tensor(buf84, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [x_59, x_60, float_11, normalize_10, normalized_10, mul_20, mul_21, x_61, x_62], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_6.run(buf79, arg41_1, buf81, arg42_1, buf83, 1597440, stream=raw_stream0)
            del arg41_1
            del arg42_1
            del buf79
            del buf81
            buf85 = buf78; del buf78  # reuse
            # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_4.run(buf84, buf85, 5308416, stream=raw_stream0)
            del buf82
            assert_size_stride(arg44_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_65, x_66], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf86 = extern_kernels.convolution(buf85, arg44_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf86, (1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg44_1
            del buf85
            assert_size_stride(arg45_1, (1024, ), (1, ), 'input')
            buf89 = empty_strided_cuda((1, 1024, 3, 30, 52), (4792320, 4680, 1560, 52, 1), torch.bfloat16)
            buf87 = reinterpret_tensor(buf89, (1, 1024, 1, 30, 52), (4792320, 4680, 1560, 52, 1), 3120)  # alias
            # Topologically Sorted Source Nodes: [x_53, x_54, x_55, x_65, x_66, x_67], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_constant_pad_nd_convolution_18.run(buf86, arg45_1, buf71, arg37_1, buf57, buf87, 1597440, stream=raw_stream0)
            del arg37_1
            del arg45_1
            del buf57
            del buf71
            del buf86
            assert_size_stride(arg46_1, (1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), 'input')
            arg46_1 = copy_if_misaligned(arg46_1)
            buf88 = reinterpret_tensor(buf89, (1, 1024, 2, 30, 52), (4792320, 4680, 1560, 52, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_2.run(arg46_1, buf88, 3194880, stream=raw_stream0)
            assert_size_stride(arg47_1, (2048, 1024, 3, 1, 1), (3072, 3, 1, 1, 1), 'input')
            del buf88
            # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
            buf90 = extern_kernels.convolution(buf89, arg47_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf90, (1, 2048, 1, 30, 52), (3194880, 1560, 1560, 52, 1), 'torch.ops.aten.convolution.default')
            del arg47_1
            assert_size_stride(arg48_1, (2048, ), (1, ), 'input')
            buf92 = empty_strided_cuda((2, 1024, 60, 104), (6389760, 1, 106496, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_70, x_71, getitem_25, getitem_26, x_72, x_73, permute_4, x_74, float_12, interpolate, input_1], Original ATen: [aten.convolution, aten.view, aten.select, aten.stack, aten.permute, aten._to_copy, aten.arange, aten.add, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_19.run(buf90, arg48_1, buf92, 12779520, stream=raw_stream0)
            del arg48_1
            del buf90
            assert_size_stride(arg49_1, (1024, 1024, 3, 3), (9216, 9, 3, 1), 'input')
            buf93 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [interpolate, input_1, input_2], Original ATen: [aten.clone, aten._to_copy, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_20.run(arg49_1, buf93, 1048576, 9, stream=raw_stream0)
            del arg49_1
            # Topologically Sorted Source Nodes: [interpolate, input_1, input_2], Original ATen: [aten.clone, aten._to_copy, aten.convolution]
            buf94 = extern_kernels.convolution(buf92, buf93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf94, (2, 1024, 60, 104), (6389760, 1, 106496, 1024), 'torch.ops.aten.convolution.default')
            del buf93
            assert_size_stride(arg50_1, (1024, ), (1, ), 'input')
            assert_size_stride(arg51_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf95 = reinterpret_tensor(buf94, (1, 1024, 2, 60, 104), (12779520, 1, 6389760, 106496, 1024), 0); del buf94  # reuse
            buf99 = empty_strided_cuda((1, 1024, 4, 60, 104), (25559040, 24960, 6240, 104, 1), torch.bfloat16)
            buf98 = reinterpret_tensor(buf99, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 12480)  # alias
            buf154 = reinterpret_tensor(buf92, (1, 1024, 2, 60, 104), (12779520, 6240, 6389760, 104, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [x_29, x_30, x_31, interpolate, input_1, input_2, view_1, x_75, x_76, x_77, permute_6, x_78, x_79, x_80, float_13, normalize_11, normalized_11, mul_22, mul_23, x_81, x_82, getitem_27], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.clone, aten._to_copy, aten.view, aten.permute, aten.unsqueeze, aten.expand, aten.linalg_vector_norm, aten.clamp_min, aten.div, aten.mul, aten.silu, aten.slice]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clamp_min_clone_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_permute_silu_slice_unsqueeze_view_21.run(buf95, arg50_1, buf41, arg21_1, buf27, arg51_1, buf98, buf154, 12480, 1024, stream=raw_stream0)
            del arg21_1
            del arg50_1
            del arg51_1
            del buf27
            del buf41
            assert_size_stride(arg52_1, (1, 1024, 2, 60, 104), (12779520, 6240, 6389760, 104, 1), 'input')
            arg52_1 = copy_if_misaligned(arg52_1)
            buf97 = reinterpret_tensor(buf99, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [float_13, normalize_11, normalized_11, mul_22, mul_23, x_81, x_82, x_83], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_22.run(arg52_1, buf97, 12779520, stream=raw_stream0)
            del arg52_1
            buf100 = empty_strided_cuda((1, 1024, 4, 62, 106), (26918912, 26288, 6572, 106, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_23.run(buf99, buf100, 26918912, stream=raw_stream0)
            del buf97
            del buf98
            assert_size_stride(arg53_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_84, x_85], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf101 = extern_kernels.convolution(buf100, arg53_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf101, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'torch.ops.aten.convolution.default')
            del arg53_1
            assert_size_stride(arg54_1, (1024, ), (1, ), 'input')
            assert_size_stride(arg55_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf105 = buf99; del buf99  # reuse
            buf104 = reinterpret_tensor(buf105, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 12480)  # alias
            # Topologically Sorted Source Nodes: [x_84, x_85, float_14, normalize_12, normalized_12, mul_24, mul_25, x_86, x_87], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_24.run(buf101, arg54_1, arg55_1, buf104, 12480, 1024, stream=raw_stream0)
            del arg54_1
            del arg55_1
            del buf101
            assert_size_stride(arg56_1, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'input')
            arg56_1 = copy_if_misaligned(arg56_1)
            buf103 = reinterpret_tensor(buf105, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_84, x_85, float_14, normalize_12, normalized_12, mul_24, mul_25, x_86, x_87, x_89], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25.run(arg56_1, buf103, 12779520, stream=raw_stream0)
            del arg56_1
            buf106 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_23.run(buf105, buf106, 26918912, stream=raw_stream0)
            assert_size_stride(arg57_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_90, x_91], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf107 = extern_kernels.convolution(buf106, arg57_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf107, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'torch.ops.aten.convolution.default')
            del arg57_1
            assert_size_stride(arg58_1, (1024, ), (1, ), 'input')
            assert_size_stride(arg59_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf112 = empty_strided_cuda((1, 1024, 4, 60, 104), (25559040, 24960, 6240, 104, 1), torch.bfloat16)
            buf111 = reinterpret_tensor(buf112, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 12480)  # alias
            # Topologically Sorted Source Nodes: [x_90, x_91, x_92, float_15, normalize_13, normalized_13, mul_26, mul_27, x_93, x_94], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_26.run(buf107, arg58_1, buf95, arg59_1, buf111, 12480, 1024, stream=raw_stream0)
            del arg59_1
            assert_size_stride(arg60_1, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'input')
            arg60_1 = copy_if_misaligned(arg60_1)
            buf110 = reinterpret_tensor(buf112, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_94, x_95], Original ATen: [aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25.run(arg60_1, buf110, 12779520, stream=raw_stream0)
            del arg60_1
            buf113 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_23.run(buf112, buf113, 26918912, stream=raw_stream0)
            assert_size_stride(arg61_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_96, x_97], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf114 = extern_kernels.convolution(buf113, arg61_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf114, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'torch.ops.aten.convolution.default')
            del arg61_1
            assert_size_stride(arg62_1, (1024, ), (1, ), 'input')
            assert_size_stride(arg63_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf118 = empty_strided_cuda((1, 1024, 4, 60, 104), (25559040, 24960, 6240, 104, 1), torch.bfloat16)
            buf117 = reinterpret_tensor(buf118, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 12480)  # alias
            # Topologically Sorted Source Nodes: [x_96, x_97, float_16, normalize_14, normalized_14, mul_28, mul_29, x_98, x_99], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_24.run(buf114, arg62_1, arg63_1, buf117, 12480, 1024, stream=raw_stream0)
            del arg62_1
            del arg63_1
            del buf114
            assert_size_stride(arg64_1, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'input')
            arg64_1 = copy_if_misaligned(arg64_1)
            buf116 = reinterpret_tensor(buf118, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_96, x_97, float_16, normalize_14, normalized_14, mul_28, mul_29, x_98, x_99, x_101], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25.run(arg64_1, buf116, 12779520, stream=raw_stream0)
            del arg64_1
            buf119 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_23.run(buf118, buf119, 26918912, stream=raw_stream0)
            assert_size_stride(arg65_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_102, x_103], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf120 = extern_kernels.convolution(buf119, arg65_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf120, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'torch.ops.aten.convolution.default')
            del arg65_1
            assert_size_stride(arg66_1, (1024, ), (1, ), 'input')
            assert_size_stride(arg67_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf121 = empty_strided_cuda((1, 1024, 2, 60, 104), (12845056, 12544, 6272, 104, 1), torch.bfloat16)
            buf125 = empty_strided_cuda((1, 1024, 4, 60, 104), (25559040, 24960, 6240, 104, 1), torch.bfloat16)
            buf124 = reinterpret_tensor(buf125, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 12480)  # alias
            # Topologically Sorted Source Nodes: [x_90, x_91, x_92, x_102, x_103, x_104, float_17, normalize_15, normalized_15, mul_30, mul_31, x_105, x_106], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_27.run(buf120, arg66_1, buf107, arg58_1, buf95, arg67_1, buf121, buf124, 12480, 1024, stream=raw_stream0)
            del arg58_1
            del arg66_1
            del arg67_1
            del buf107
            del buf120
            assert_size_stride(arg68_1, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'input')
            arg68_1 = copy_if_misaligned(arg68_1)
            buf123 = reinterpret_tensor(buf125, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [float_17, normalize_15, normalized_15, mul_30, mul_31, x_105, x_106, x_107], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25.run(arg68_1, buf123, 12779520, stream=raw_stream0)
            del arg68_1
            buf126 = buf119; del buf119  # reuse
            # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_23.run(buf125, buf126, 26918912, stream=raw_stream0)
            assert_size_stride(arg69_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_108, x_109], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf127 = extern_kernels.convolution(buf126, arg69_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf127, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'torch.ops.aten.convolution.default')
            del arg69_1
            assert_size_stride(arg70_1, (1024, ), (1, ), 'input')
            assert_size_stride(arg71_1, (1024, 1, 1, 1), (1, 1, 1, 1), 'input')
            buf131 = empty_strided_cuda((1, 1024, 4, 60, 104), (25559040, 24960, 6240, 104, 1), torch.bfloat16)
            buf130 = reinterpret_tensor(buf131, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 12480)  # alias
            # Topologically Sorted Source Nodes: [x_108, x_109, float_18, normalize_16, normalized_16, mul_32, mul_33, x_110, x_111], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_24.run(buf127, arg70_1, arg71_1, buf130, 12480, 1024, stream=raw_stream0)
            del arg70_1
            del arg71_1
            del buf127
            assert_size_stride(arg72_1, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'input')
            arg72_1 = copy_if_misaligned(arg72_1)
            buf129 = reinterpret_tensor(buf131, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_108, x_109, float_18, normalize_16, normalized_16, mul_32, mul_33, x_110, x_111, x_113], Original ATen: [aten.constant_pad_nd, aten.convolution, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.mul, aten.add, aten.silu, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25.run(arg72_1, buf129, 12779520, stream=raw_stream0)
            del arg72_1
            buf132 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_23.run(buf131, buf132, 26918912, stream=raw_stream0)
            assert_size_stride(arg73_1, (1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), 'input')
            # Topologically Sorted Source Nodes: [x_114, x_115], Original ATen: [aten.constant_pad_nd, aten.convolution]
            buf133 = extern_kernels.convolution(buf132, arg73_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf133, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'torch.ops.aten.convolution.default')
            del arg73_1
            del buf132
            assert_size_stride(arg75_1, (1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), 'input')
            arg75_1 = copy_if_misaligned(arg75_1)
            buf136 = empty_strided_cuda((1, 1024, 4, 60, 104), (25559040, 24960, 6240, 104, 1), torch.bfloat16)
            buf134 = reinterpret_tensor(buf136, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [x_114, x_115, x_116, x_117], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_25.run(arg75_1, buf134, 12779520, stream=raw_stream0)
            del arg75_1
            assert_size_stride(arg74_1, (1024, ), (1, ), 'input')
            buf135 = reinterpret_tensor(buf136, (1, 1024, 2, 60, 104), (25559040, 24960, 6240, 104, 1), 12480)  # alias
            # Topologically Sorted Source Nodes: [x_114, x_115, x_116], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_constant_pad_nd_convolution_28.run(buf133, arg74_1, buf121, buf135, 12779520, stream=raw_stream0)
            del arg74_1
            del buf121
            del buf133
            assert_size_stride(arg76_1, (2048, 1024, 3, 1, 1), (3072, 3, 1, 1, 1), 'input')
            # Topologically Sorted Source Nodes: [x_119], Original ATen: [aten.convolution]
            buf137 = extern_kernels.convolution(buf136, arg76_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
            assert_size_stride(buf137, (1, 2048, 2, 60, 104), (25559040, 12480, 6240, 104, 1), 'torch.ops.aten.convolution.default')
            del arg76_1
            assert_size_stride(arg77_1, (2048, ), (1, ), 'input')
            buf139 = empty_strided_cuda((4, 1024, 120, 208), (25559040, 1, 212992, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_119, x_120, getitem_34, getitem_35, x_121, x_122, permute_7, x_123, float_19, interpolate_1, input_3], Original ATen: [aten.convolution, aten.view, aten.select, aten.stack, aten.permute, aten._to_copy, aten.arange, aten.add, aten.mul, aten.unsqueeze, aten._unsafe_index, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_select_stack_unsqueeze_view_29.run(buf137, arg77_1, buf139, 102236160, stream=raw_stream0)
            del arg77_1
            del buf137
            assert_size_stride(arg78_1, (1024, 1024, 3, 3), (9216, 9, 3, 1), 'input')
            buf140 = empty_strided_cuda((1024, 1024, 3, 3), (9216, 1, 3072, 1024), torch.bfloat16)
            # Topologically Sorted Source Nodes: [interpolate_1, input_3, input_4], Original ATen: [aten.clone, aten._to_copy, aten.convolution]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_convolution_20.run(arg78_1, buf140, 1048576, 9, stream=raw_stream0)
            del arg78_1
            # Topologically Sorted Source Nodes: [interpolate_1, input_3, input_4], Original ATen: [aten.clone, aten._to_copy, aten.convolution]
            buf141 = extern_kernels.convolution(buf139, buf140, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf141, (4, 1024, 120, 208), (25559040, 1, 212992, 1024), 'torch.ops.aten.convolution.default')
            del buf140
            assert_size_stride(arg79_1, (1024, ), (1, ), 'input')
            buf142 = reinterpret_tensor(buf139, (1, 1024, 4, 120, 208), (102236160, 24960, 25559040, 208, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [interpolate_1, input_3, input_4, view_4, x_124, x_125, x_126, permute_9, x_127, x_128, x_129], Original ATen: [aten.clone, aten._to_copy, aten.convolution, aten.view, aten.permute, aten.unsqueeze, aten.expand, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_expand_permute_unsqueeze_view_30.run(buf141, arg79_1, buf95, buf142, 4096, 24960, stream=raw_stream0)
            del arg79_1
            del buf141
            del buf95
            buf143 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_1, unsqueeze, getitem, cache_x_1], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg2_1, buf3, buf143, 3194880, stream=raw_stream0)
            del arg2_1
            del buf3
            del buf4
            buf144 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_3, unsqueeze_1, getitem_2, cache_x_4], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg6_1, buf10, buf144, 3194880, stream=raw_stream0)
            del arg6_1
            del buf10
            del buf11
            buf145 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_8, unsqueeze_2, getitem_7, cache_x_7], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg15_1, buf31, buf145, 3194880, stream=raw_stream0)
            del arg15_1
            del buf31
            del buf32
            buf146 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_10, unsqueeze_3, getitem_9, cache_x_10], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg19_1, buf38, buf146, 3194880, stream=raw_stream0)
            del arg19_1
            del buf38
            del buf39
            buf147 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_12, unsqueeze_4, getitem_11, cache_x_13], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg23_1, buf46, buf147, 3194880, stream=raw_stream0)
            del arg23_1
            del buf46
            del buf47
            buf148 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_14, unsqueeze_5, getitem_13, cache_x_16], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg27_1, buf53, buf148, 3194880, stream=raw_stream0)
            del arg27_1
            del buf53
            del buf54
            buf149 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_16, unsqueeze_6, getitem_15, cache_x_19], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg31_1, buf61, buf149, 3194880, stream=raw_stream0)
            del arg31_1
            del buf61
            del buf62
            buf150 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_18, unsqueeze_7, getitem_17, cache_x_22], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg35_1, buf68, buf150, 3194880, stream=raw_stream0)
            del arg35_1
            del buf68
            del buf69
            buf151 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_20, unsqueeze_8, getitem_19, cache_x_25], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg39_1, buf76, buf151, 3194880, stream=raw_stream0)
            del arg39_1
            del buf76
            del buf77
            buf152 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_22, unsqueeze_9, getitem_21, cache_x_28], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg43_1, buf83, buf152, 3194880, stream=raw_stream0)
            del arg43_1
            del buf83
            del buf84
            buf153 = empty_strided_cuda((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_24, unsqueeze_10, getitem_23, cache_x_31], Original ATen: [aten.select, aten.unsqueeze, aten.slice, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_select_slice_unsqueeze_31.run(arg46_1, buf87, buf153, 3194880, stream=raw_stream0)
            del arg46_1
            del buf87
            del buf89
        return (buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf104, buf111, buf117, buf124, buf130, buf135, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 1024, 1, 30, 52), (1597440, 1560, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((3072, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((1, 1024, 2, 30, 52), (3194880, 3120, 1560, 52, 1), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((2048, 1024, 3, 1, 1), (3072, 3, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((1, 1024, 2, 60, 104), (12779520, 6240, 6389760, 104, 1), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((1024, 1024, 3, 3, 3), (27648, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((1, 1024, 2, 60, 104), (12779520, 12480, 6240, 104, 1), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((2048, 1024, 3, 1, 1), (3072, 3, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
