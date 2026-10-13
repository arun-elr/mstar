GPU: NVIDIA B200  torch 2.12.1+cu132  cuDNN 92000  diffusers 0.39.0  mstar 99f5108b

| variant | median ms/chunk | p90 ms | decoder fps (12/median) | peak alloc GiB | PSNR float dB | PSNR u8 dB | note |
|---|---:|---:|---:|---:|---:|---:|---|
| ref_fp32 | 3247.96 | 3249.29 | 3.7 | 10.25 |  |  |  |
| diffusers_bf16 | 184.57 | 184.75 | 65.0 | 5.53 | 62.19 | 55.27 |  |
| diffusers_bf16_compile | 105.18 | 106.15 | 114.1 | 6.32 | 62.58 | 55.27 |  |
| mstar_ar_ctx8_eager | 630.64 | 631.99 | 19.0 | 5.65 |  | 51.13 | decodes 11 latents/chunk |
| mstar_ar_ctx8 | 370.13 | 372.19 | 32.4 | 6.44 |  | 51.08 | decodes 11 latents/chunk |
| mstar_ar_ctx4 | 226.57 | 228.24 | 53.0 | 6.38 |  | 46.03 | decodes 7 latents/chunk |
| mstar_ar_ctx2 | 155.48 | 157.1 | 77.2 | 6.35 |  | 42.0 | decodes 5 latents/chunk |
| mstar_ar_ctx1 | 120.26 | 121.26 | 99.8 | 6.34 |  | 30.93 | decodes 4 latents/chunk |
| mstar_wan22_oneshot_81f | 184.53 | 184.6 | 65.0 | 5.72 |  |  | whole 81-frame clip 1246 ms, per-12-frame share |
| mstar_wan22_oneshot_241f | 184.97 | 185.06 | 64.9 | 6.27 |  |  | whole 241-frame clip 3715 ms, per-12-frame share |
| mstar_wan22_oneshot_241f_tiled | 475.9 | 476.05 | 25.2 | 3.84 |  |  | whole 241-frame clip 9558 ms, per-12-frame share |
| rx_cl | 161.38 | 162.31 | 74.4 | 5.14 | 62.19 | 55.27 | non-CL conv inputs 4/103 |
| rx_cl_chunk | 155.06 | 156.08 | 77.4 | 9.88 | 62.19 | 55.27 | non-CL conv inputs 1/35 |
| rx_cl_chunk_compile | 75.89 | 77.15 | 158.1 | 16.59 | 62.58 | 55.27 |  |
| rx_full | 75.75 | 77.17 | 158.4 | 16.59 | 62.58 | 55.27 |  |
| rx_full_fp16 | 79.18 | 80.86 | 151.6 | 16.59 | 80.66 | 61.17 |  |
| diffusers_bf16 (drift) | 184.26 | 184.38 | 65.1 | 5.53 | 62.19 | 55.27 |  |

