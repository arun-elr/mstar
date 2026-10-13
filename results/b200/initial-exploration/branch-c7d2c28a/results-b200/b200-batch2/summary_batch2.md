# Batch 2 results: b200-batch2

## Run-to-run variance (median ms per chunk; fresh process + fresh compile each)

| variant | run1 | run2 | run3 | mean of medians | spread (max-min) | p90 (mean) |
|---|---:|---:|---:|---:|---:|---:|
| diffusers_bf16 | 191.96 | 192.03 | 191.96 | 191.98 | 0.07 (0.0%) | 192.20 |
| diffusers_bf16_compile | 95.85 | 95.93 | 95.92 | 95.90 | 0.08 (0.1%) | 96.72 |
| mstar_ar_ctx8 | 335.02 | 335.47 | 335.27 | 335.25 | 0.45 (0.1%) | 337.05 |
| mstar_ar_ctx8_eager | 656.84 | 656.85 | 656.85 | 656.85 | 0.01 (0.0%) | 657.11 |
| px_nopad_bench | 55.83 | 55.91 | 55.49 | 55.74 | 0.42 (0.8%) | 56.82 |
| rx_cl_chunk_compile | 63.29 | 63.17 | 63.33 | 63.26 | 0.16 (0.3%) | 64.30 |
| rx_cl_compile | 68.30 | 68.39 | 68.36 | 68.35 | 0.09 (0.1%) | 70.24 |
| rx_full | 63.19 | 63.17 | 63.21 | 63.19 | 0.04 (0.1%) | 64.25 |
| rx_full_fp16 | 64.91 | 64.89 | 64.93 | 64.91 | 0.04 (0.1%) | 66.34 |

## TF32 check (fp32 reference)

| variant | median ms | p90 ms | peak GiB | PSNR float dB (peak 2) | PSNR float dB (peak 1) | PSNR u8 dB | non-CL conv inputs |
|---|---:|---:|---:|---:|---:|---:|---|
| ref_fp32 | 2407.48 | 2407.55 | 7.7 | inf | inf | inf |  |
| ref_fp32_tf32 | 234.30 | 234.44 | 8.9 | 86.02 | 80.00 | 68.26 |  |

## Beyond the reference ladder (Wan 2.1)

| variant | median ms | p90 ms | peak GiB | PSNR float dB (peak 2) | PSNR float dB (peak 1) | PSNR u8 dB | non-CL conv inputs |
|---|---:|---:|---:|---:|---:|---:|---|
| px_base | 63.44 | 64.71 | 20.0 | 59.93 | 53.91 | 54.98 | 3/39 |
| px_bench | 62.43 | 63.41 | 20.0 | 59.93 | 53.91 | 54.98 |  |
| px_cg | 63.38 | 64.20 | 12.2 | 59.93 | 53.91 | 54.98 |  |
| px_cg_nopad | 59.75 | 60.68 | 12.2 | 59.93 | 53.91 | 54.98 |  |
| px_cg_nopad_bench | 55.83 | 57.02 | 12.2 | 59.93 | 53.91 | 54.98 |  |
| px_cg_nopad_bench_fp16 | 58.21 | 59.48 | 12.2 | 80.05 | 74.03 | 61.83 |  |
| px_maxat | 62.49 | 63.52 | 20.0 | 59.99 | 53.97 | 55.01 |  |
| px_nopad | 59.11 | 60.48 | 19.0 | 59.93 | 53.91 | 54.98 | 0/39 |
| px_nopad_bench | 55.75 | 56.77 | 19.0 | 59.93 | 53.91 | 54.98 | 0/39 |
| px_ro | 65.17 | 66.17 | 3.2 | 59.93 | 53.91 | 54.98 |  |

## Best prototype on the Wan2.2 VAE

| variant | median ms | p90 ms | peak GiB | PSNR float dB (peak 2) | PSNR float dB (peak 1) | PSNR u8 dB | non-CL conv inputs |
|---|---:|---:|---:|---:|---:|---:|---|
| px_base | 75.89 | 77.29 | 16.6 | 62.58 | 56.56 | 55.27 |  |
| px_nopad_bench | 72.99 | 75.31 | 15.9 | 62.58 | 56.56 | 55.27 |  |

## Where the time goes (wan21)

Useful work per 12-frame chunk: 40.65 TFLOP (conv 40.47); minimum conv traffic 26.26 GB → intensity 1548 FLOP/B; roof min(2250, 1548×8) = 2250 TFLOP/s (compute-bound). M* ctx-8 streaming executes 139.9 TFLOP per chunk.

| variant | kernels/chunk | GPU ms/chunk | convolution | layout conversion | norm / SiLU | concat / pad | upsample | attention | copies | other elementwise | other | wall ms (bench) | achieved TFLOP/s (useful) | % of roof |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diffusers_bf16 | 2046 | 192.8 | 39.6 | 24.1 | 8.4 | 4.0 | 3.9 | 1.9 | 62.5 | 48.3 | 0.1 | 192.0 | 212 | 9% |
| mstar_ar_ctx8 | 3432 | 331.1 | 136.7 | 77.3 | 29.9 | 40.1 | 28.9 | 7.1 | 3.8 | 7.1 | 0.2 | 335.3 | 121 | 5% |
| px_base | 170 | 62.2 | 40.7 | 0.0 | 9.3 | 5.1 | 1.7 | 1.3 | 3.3 | 0.7 | 0.0 | 63.4 | 641 | 28% |
| px_bench | 170 | 61.2 | 39.6 | 0.0 | 9.4 | 5.1 | 1.7 | 1.3 | 3.3 | 0.8 | 0.0 | 62.4 | 651 | 29% |
| px_cg | 203 | 63.0 | 40.7 | 0.0 | 9.4 | 5.1 | 1.8 | 1.3 | 4.0 | 0.8 | 0.0 | 63.4 | 641 | 29% |
| px_cg_nopad | 172 | 58.4 | 40.2 | 0.0 | 9.3 | 1.1 | 3.8 | 1.3 | 1.4 | 1.2 | 0.0 | 59.8 | 680 | 30% |
| px_cg_nopad_bench | 172 | 55.5 | 37.8 | 0.0 | 9.1 | 1.1 | 3.5 | 1.3 | 1.4 | 1.2 | 0.1 | 55.8 | 728 | 32% |
| px_cg_nopad_bench_fp16 | 172 | 56.9 | 39.2 | 0.0 | 9.1 | 1.1 | 3.5 | 1.4 | 1.4 | 1.2 | 0.1 | 58.2 | 698 | 31% |
| px_maxat | 164 | 61.3 | 38.2 | 0.0 | 8.6 | 7.5 | 1.6 | 1.3 | 3.3 | 0.8 | 0.0 | 62.5 | 651 | 29% |
| px_nopad | 139 | 58.2 | 40.6 | 0.0 | 9.4 | 1.1 | 3.6 | 1.4 | 0.8 | 1.2 | 0.0 | 59.1 | 688 | 31% |
| px_nopad_bench | 139 | 55.0 | 38.0 | 0.0 | 9.2 | 1.1 | 3.6 | 1.3 | 0.7 | 1.2 | 0.1 | 55.7 | 729 | 32% |
| px_ro | 223 | 64.0 | 40.5 | 0.0 | 9.3 | 5.1 | 1.7 | 1.3 | 5.4 | 0.7 | 0.0 | 65.2 | 624 | 28% |
| rx_full | 170 | 62.5 | 40.8 | 0.0 | 9.4 | 5.1 | 1.8 | 1.3 | 3.3 | 0.8 | 0.0 | 63.2 | 643 | 29% |

<details><summary>Top-15 kernels: diffusers_bf16</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 415.0 | 41.36 |
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 21.0 | 18.42 |
| `void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, float, false, true, (cudnnKern` | layout conversion | 210.0 | 16.33 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 121.0 | 14.54 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | other elementwise | 90.0 | 14.31 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | copies | 99.0 | 13.98 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::` | other elementwise | 90.0 | 11.57 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 18.0 | 10.74 |
| `void cudnn::engines_precompiled::nhwcToNchwKernel<__nv_bfloat16, __nv_bfloat16, float, true, false, (cudnnKern` | layout conversion | 108.0 | 7.80 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | copies | 99.0 | 5.48 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 21.0 | 5.03 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::(anonymous namespace)::silu_kernel(at::TensorIte` | norm / SiLU | 87.0 | 4.42 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1u` | concat / pad | 90.0 | 4.04 |
| `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::NormTwoOps<float, float, float,` | norm / SiLU | 90.0 | 4.03 |
| `void at::native::(anonymous namespace)::upsample_nearest2d_out_frame<float, &at::native::nearest_neighbor_exac` | upsample | 9.0 | 3.90 |

</details>

<details><summary>Top-15 kernels: mstar_ar_ctx8</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 77.0 | 62.87 |
| `void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, float, false, true, (cudnnKern` | layout conversion | 736.0 | 52.37 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 66.0 | 36.64 |
| `void cudnn::engines_precompiled::nhwcToNchwKernel<__nv_bfloat16, __nv_bfloat16, float, true, false, (cudnnKern` | layout conversion | 379.0 | 24.92 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_mul_permute_unsqueeze_vie` | upsample | 10.0 | 22.62 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 76.0 | 17.67 |
| `triton_poi_fused_constant_pad_nd_4` | concat / pad | 160.0 | 13.30 |
| `triton_poi_fused_constant_pad_nd_5` | concat / pad | 99.0 | 8.00 |
| `cutlass3x_sm100_tensorop_s256x64x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x64x64_dyn_cga_ndhwc_ndh` | convolution | 11.0 | 7.60 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 11.0 | 7.07 |
| `triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_5` | norm / SiLU | 33.0 | 6.49 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 131.0 | 6.41 |
| `triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_8` | norm / SiLU | 20.0 | 4.79 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 11.0 | 4.63 |
| `triton_poi_fused_cat_constant_pad_nd_0` | concat / pad | 20.0 | 4.01 |

</details>

<details><summary>Top-15 kernels: px_base</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 8.0 | 19.28 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 13.0 | 15.99 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 14.0 | 4.31 |
| `triton_poi_fused_constant_pad_nd_2` | concat / pad | 23.0 | 3.25 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 8.0 | 3.17 |
| `cutlass3x_sm100_tensorop_s256x64x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x64x64_dyn_cga_ndhwc_ndh` | convolution | 1.0 | 2.62 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_5` | norm / SiLU | 3.0 | 1.57 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 12.0 | 1.36 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.32 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 2.0 | 1.31 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 3.0 | 1.21 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.11 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_mul_permute_unsqueeze_vie` | upsample | 1.0 | 1.00 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.67 |
| `triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_1` | concat / pad | 23.0 | 0.51 |

</details>

<details><summary>Top-15 kernels: px_bench</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 6.0 | 17.74 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 6.0 | 10.81 |
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 9.0 | 5.56 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 14.0 | 4.35 |
| `triton_poi_fused_constant_pad_nd_2` | concat / pad | 23.0 | 3.26 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 8.0 | 3.18 |
| `sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256x32x32_stage4_warpsize4x1x1_g1_ten` | convolution | 1.0 | 2.67 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_5` | norm / SiLU | 3.0 | 1.57 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 11.0 | 1.34 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.33 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 3.0 | 1.22 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.12 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 1.0 | 1.06 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_mul_permute_unsqueeze_vie` | upsample | 1.0 | 0.99 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.67 |

</details>

<details><summary>Top-15 kernels: px_cg</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 8.0 | 19.33 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 13.0 | 16.04 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 14.0 | 4.35 |
| `triton_poi_fused_constant_pad_nd_2` | concat / pad | 23.0 | 3.24 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 8.0 | 3.21 |
| `cutlass3x_sm100_tensorop_s256x64x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x64x64_dyn_cga_ndhwc_ndh` | convolution | 1.0 | 2.52 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_5` | norm / SiLU | 3.0 | 1.58 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 12.0 | 1.38 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.34 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 2.0 | 1.32 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 3.0 | 1.21 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.13 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_mul_permute_unsqueeze_vie` | upsample | 1.0 | 1.01 |
| `memcpy128` | copies | 32.0 | 0.72 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.67 |

</details>

<details><summary>Top-15 kernels: px_cg_nopad</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 8.0 | 19.14 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 13.0 | 15.94 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_2` | norm / SiLU | 14.0 | 4.32 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_mul_unsqueeze_0` | upsample | 2.0 | 2.53 |
| `cutlass3x_sm100_tensorop_s32x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t3` | convolution | 1.0 | 2.24 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 4.0 | 1.68 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 12.0 | 1.39 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 2.0 | 1.34 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.34 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 3.0 | 1.19 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.12 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_unsqueeze_view_5` | upsample | 1.0 | 1.00 |
| `memcpy128` | copies | 32.0 | 0.72 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.67 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 6.0 | 0.67 |

</details>

<details><summary>Top-15 kernels: px_cg_nopad_bench</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t3` | convolution | 6.0 | 17.77 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 6.0 | 10.71 |
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 8.0 | 5.51 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_2` | norm / SiLU | 14.0 | 4.24 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_mul_unsqueeze_0` | upsample | 2.0 | 2.22 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 4.0 | 1.65 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 11.0 | 1.31 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.30 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 3.0 | 1.18 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.09 |
| `cutlass3x_sm100_tensorop_s32x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t3` | convolution | 1.0 | 1.08 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 1.0 | 1.02 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_unsqueeze_view_5` | upsample | 1.0 | 0.98 |
| `memcpy128` | copies | 32.0 | 0.72 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.66 |

</details>

<details><summary>Top-15 kernels: px_cg_nopad_bench_fp16</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_f16_f16_f32_void_f16_void_t3xr3` | convolution | 6.0 | 18.39 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_f16_f16_f32_void_f16_f32_256x256x64_dyn_cga_ndhwc_ndhw` | convolution | 6.0 | 11.07 |
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_f16_f16_f32_void_f16_f32_256x128x64_dyn_cga_ndhwc_ndhw` | convolution | 9.0 | 5.72 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_2` | norm / SiLU | 14.0 | 4.32 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_mul_unsqueeze_0` | upsample | 2.0 | 2.25 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 4.0 | 1.53 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_f16_f16_f32_void_f16_f32_128x128x64_dyn_cga_ndhwc_ndhw` | convolution | 11.0 | 1.36 |
| `fmha_cutlassF_f16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::half_t, cutlass::a` | attention | 1.0 | 1.34 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 3.0 | 1.14 |
| `cutlass3x_sm100_tensorop_s32x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_f16_f16_f32_void_f16_void_t3xr3` | convolution | 1.0 | 1.14 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.07 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_f16_f16_f32_void_f16_void_t1xr3` | convolution | 1.0 | 1.06 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_unsqueeze_view_5` | upsample | 1.0 | 1.00 |
| `memcpy128` | copies | 32.0 | 0.72 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 6.0 | 0.66 |

</details>

<details><summary>Top-15 kernels: px_maxat</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 8.0 | 19.34 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 13.0 | 16.14 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 14.0 | 3.86 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 8.0 | 3.22 |
| `triton_poi_fused_constant_pad_nd_2` | concat / pad | 23.0 | 2.90 |
| `triton_tem_fused_cat_constant_pad_nd_convolution_1` | concat / pad | 2.0 | 2.87 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_5` | norm / SiLU | 3.0 | 1.46 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 11.0 | 1.35 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.34 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 2.0 | 1.33 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.10 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 3.0 | 1.09 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_mul_permute_unsqueeze_vie` | upsample | 1.0 | 1.00 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.68 |
| `triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_1` | concat / pad | 23.0 | 0.47 |

</details>

<details><summary>Top-15 kernels: px_nopad</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 8.0 | 19.28 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 13.0 | 16.14 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_2` | norm / SiLU | 14.0 | 4.37 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_mul_unsqueeze_0` | upsample | 2.0 | 2.37 |
| `cutlass3x_sm100_tensorop_s32x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t3` | convolution | 1.0 | 2.35 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 4.0 | 1.69 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 12.0 | 1.41 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.36 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 2.0 | 1.34 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 3.0 | 1.21 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.13 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_unsqueeze_view_5` | upsample | 1.0 | 1.00 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.68 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 5.0 | 0.68 |
| `triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_1` | concat / pad | 23.0 | 0.52 |

</details>

<details><summary>Top-15 kernels: px_nopad_bench</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t3` | convolution | 6.0 | 17.90 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 6.0 | 10.70 |
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 10.0 | 5.52 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_2` | norm / SiLU | 14.0 | 4.26 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_mul_unsqueeze_0` | upsample | 2.0 | 2.31 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 4.0 | 1.66 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 11.0 | 1.29 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.28 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 3.0 | 1.18 |
| `cutlass3x_sm100_tensorop_s32x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t3` | convolution | 1.0 | 1.10 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.09 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 1.0 | 1.03 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_unsqueeze_view_5` | upsample | 1.0 | 0.97 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.66 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 5.0 | 0.64 |

</details>

<details><summary>Top-15 kernels: px_ro</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 8.0 | 19.01 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 13.0 | 15.96 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 14.0 | 4.30 |
| `triton_poi_fused_constant_pad_nd_2` | concat / pad | 23.0 | 3.23 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 8.0 | 3.16 |
| `cutlass3x_sm100_tensorop_s256x64x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x64x64_dyn_cga_ndhwc_ndh` | convolution | 1.0 | 2.74 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_5` | norm / SiLU | 3.0 | 1.55 |
| `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorLis` | copies | 20.0 | 1.49 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 12.0 | 1.36 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.32 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 2.0 | 1.31 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 3.0 | 1.20 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.12 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_mul_permute_unsqueeze_vie` | upsample | 1.0 | 1.00 |
| `Memcpy DtoD (Device -> Device)` | copies | 35.0 | 0.74 |

</details>

<details><summary>Top-15 kernels: rx_full</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 8.0 | 19.33 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 13.0 | 16.08 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 14.0 | 4.36 |
| `triton_poi_fused_constant_pad_nd_2` | concat / pad | 23.0 | 3.26 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 8.0 | 3.17 |
| `cutlass3x_sm100_tensorop_s256x64x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x64x64_dyn_cga_ndhwc_ndh` | convolution | 1.0 | 2.63 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_5` | norm / SiLU | 3.0 | 1.59 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 12.0 | 1.37 |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutla` | attention | 1.0 | 1.32 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 2.0 | 1.31 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 3.0 | 1.21 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.13 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_mul_permute_unsqueeze_vie` | upsample | 1.0 | 1.01 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.67 |
| `triton_poi_fused__to_copy_add_cat_clamp_min_div_expand_linalg_vector_norm_mul_silu_1` | concat / pad | 23.0 | 0.51 |

</details>

## Where the time goes (wan22)

Useful work per 12-frame chunk: 88.18 TFLOP (conv 88.15); minimum conv traffic 23.36 GB → intensity 3774 FLOP/B; roof min(2250, 3774×8) = 2250 TFLOP/s (compute-bound). M* ctx-8 streaming executes 303.0 TFLOP per chunk.

| variant | kernels/chunk | GPU ms/chunk | convolution | layout conversion | norm / SiLU | concat / pad | upsample | attention | copies | other elementwise | other | wall ms (bench) | achieved TFLOP/s (useful) | % of roof |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diffusers_bf16 | 2173 | 183.9 | 55.4 | 19.4 | 6.7 | 3.1 | 2.7 | 0.8 | 55.6 | 39.7 | 0.5 |  | 479 | 21% |
| mstar_ar_ctx8 | 3536 | 362.2 | 196.3 | 66.7 | 26.7 | 33.0 | 23.1 | 3.2 | 3.1 | 8.1 | 2.0 |  | 243 | 11% |
| px_base | 186 | 74.2 | 59.1 | 0.0 | 5.2 | 4.0 | 1.8 | 0.3 | 2.4 | 1.3 | 0.0 | 75.9 | 1162 | 52% |
| px_nopad_bench | 154 | 70.8 | 58.4 | 0.0 | 5.2 | 0.8 | 3.3 | 0.3 | 0.6 | 1.6 | 0.5 | 73.0 | 1208 | 54% |

<details><summary>Top-15 kernels: diffusers_bf16</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 437.0 | 38.42 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 36.0 | 34.58 |
| `void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, float, false, true, (cudnnKern` | layout conversion | 213.0 | 14.01 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 130.0 | 13.38 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | other elementwise | 90.0 | 10.95 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | copies | 99.0 | 10.58 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 30.0 | 10.01 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::` | other elementwise | 90.0 | 8.98 |
| `void cudnn::engines_precompiled::nhwcToNchwKernel<__nv_bfloat16, __nv_bfloat16, float, true, false, (cudnnKern` | layout conversion | 111.0 | 5.42 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | copies | 99.0 | 4.07 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 3.0 | 3.78 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::(anonymous namespace)::silu_kernel(at::TensorIte` | norm / SiLU | 87.0 | 3.42 |
| `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::NormTwoOps<float, float, float,` | norm / SiLU | 90.0 | 3.28 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 3.0 | 3.23 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1u` | concat / pad | 90.0 | 3.14 |

</details>

<details><summary>Top-15 kernels: mstar_ar_ctx8</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 126.0 | 118.01 |
| `void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, float, false, true, (cudnnKern` | layout conversion | 736.0 | 49.73 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 98.0 | 36.62 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_mul_permute_unsqueeze_vie` | upsample | 10.0 | 18.72 |
| `void cudnn::engines_precompiled::nhwcToNchwKernel<__nv_bfloat16, __nv_bfloat16, float, true, false, (cudnnKern` | layout conversion | 379.0 | 16.98 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 11.0 | 13.90 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 16.0 | 12.43 |
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 122.0 | 11.43 |
| `triton_poi_fused_constant_pad_nd_7` | concat / pad | 50.0 | 8.40 |
| `triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_5` | norm / SiLU | 40.0 | 5.46 |
| `triton_poi_fused_constant_pad_nd_8` | concat / pad | 50.0 | 4.08 |
| `triton_red_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_9` | norm / SiLU | 20.0 | 4.07 |
| `triton_poi_fused_constant_pad_nd_4` | concat / pad | 109.0 | 3.81 |
| `cutlass3x_sm100_tensorop_s256x64x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x64x64_dyn_cga_ndhwc_ndh` | convolution | 11.0 | 3.53 |
| `triton_red_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_2` | norm / SiLU | 16.0 | 3.30 |

</details>

<details><summary>Top-15 kernels: px_base</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 12.0 | 36.24 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 7.0 | 11.09 |
| `cutlass3x_sm100_tensorop_s64x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t1` | convolution | 1.0 | 3.64 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 1.0 | 3.56 |
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 11.0 | 2.67 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 9.0 | 2.29 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_3` | norm / SiLU | 14.0 | 1.84 |
| `triton_poi_fused_constant_pad_nd_5` | concat / pad | 10.0 | 1.80 |
| `cutlass3x_sm100_tensorop_s256x64x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x64x64_dyn_cga_ndhwc_ndh` | convolution | 2.0 | 1.21 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.17 |
| `triton_poi_fused_constant_pad_nd_2` | concat / pad | 18.0 | 1.02 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.95 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_mul_permute_unsqueeze_vie` | upsample | 1.0 | 0.84 |
| `triton_per_fused__to_copy_add_clamp_min_constant_pad_nd_convolution_div_expand_linalg_vector_norm_mul_silu_7` | norm / SiLU | 2.0 | 0.81 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_constant_pad_nd_convolution_expand_mul_permute_unsque` | upsample | 1.0 | 0.67 |

</details>

<details><summary>Top-15 kernels: px_nopad_bench</summary>

| kernel | category | calls/chunk | ms/chunk |
|---|---|---:|---:|
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 7.0 | 21.95 |
| `cutlass3x_sm100_tensorop_s128x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x256x64_dyn_cga_ndhwc_n` | convolution | 9.0 | 13.51 |
| `cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_128x128x64_dyn_cga_ndhwc_n` | convolution | 7.0 | 11.05 |
| `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x256x64_dyn_cga_ndhwc_n` | convolution | 2.0 | 8.67 |
| `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_bf16_f32_256x128x64_dyn_cga_ndhwc_n` | convolution | 11.0 | 2.50 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_2` | norm / SiLU | 14.0 | 1.84 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_mul_unsqueeze_0` | upsample | 2.0 | 1.79 |
| `triton_per_fused__to_copy_add_clamp_min_div_expand_linalg_vector_norm_mul_silu_0` | norm / SiLU | 5.0 | 1.18 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10` | other elementwise | 3.0 | 0.97 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_mul_permute_unsqueeze_view_6` | upsample | 1.0 | 0.86 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_5` | norm / SiLU | 2.0 | 0.83 |
| `triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_expand_mul_permute_unsqueeze_view_7` | upsample | 1.0 | 0.68 |
| `triton_per_fused__to_copy_add_clamp_min_convolution_div_expand_linalg_vector_norm_mul_silu_4` | norm / SiLU | 2.0 | 0.62 |
| `cutlass3x_sm100_tensorop_s32x256x16_conv3d_fprop_weight_stationary_nq_2d_tiled_bf16_bf16_f32_void_bf16_void_t3` | convolution | 1.0 | 0.61 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | copies | 6.0 | 0.52 |

</details>

