GPU: NVIDIA B200  torch 2.12.1+cu132  cuDNN 92000  diffusers 0.39.0  mstar 99f5108b

| variant | median ms/chunk | p90 ms | decoder fps (12/median) | peak alloc GiB | PSNR float dB | PSNR u8 dB | note |
|---|---:|---:|---:|---:|---:|---:|---|
| ref_fp32 | 2407.46 | 2407.61 | 5.0 | 7.73 |  |  |  |
| diffusers_bf16 | 192.11 | 192.33 | 62.5 | 4.44 | 59.75 | 54.91 |  |
| diffusers_bf16_compile | 95.9 | 96.55 | 125.1 | 6.19 | 59.93 | 54.98 |  |
| mstar_ar_ctx8_eager | 657.71 | 658.0 | 18.2 | 4.58 |  | 52.05 | decodes 11 latents/chunk |
| mstar_ar_ctx8 | 336.2 | 338.39 | 35.7 | 6.33 |  | 52.09 | decodes 11 latents/chunk |
| mstar_ar_ctx4 | 206.55 | 208.07 | 58.1 | 6.27 |  | 49.01 | decodes 7 latents/chunk |
| mstar_ar_ctx2 | 142.14 | 143.05 | 84.4 | 6.24 |  | 38.45 | decodes 5 latents/chunk |
| mstar_ar_ctx1 | 110.12 | 110.85 | 109.0 | 6.23 |  | 28.36 | decodes 4 latents/chunk |
| mstar_wan22_oneshot_81f | 192.23 | 192.25 | 62.4 | 4.62 |  |  | whole 81-frame clip 1298 ms, per-12-frame share |
| mstar_wan22_oneshot_241f | 192.76 | 192.77 | 62.3 | 5.19 |  |  | whole 241-frame clip 3871 ms, per-12-frame share |
| mstar_wan22_oneshot_241f_tiled | 429.54 | 429.96 | 27.9 | 2.96 |  |  | whole 241-frame clip 8627 ms, per-12-frame share |
| rx_cl | 162.12 | 162.28 | 74.0 | 4.01 | 59.75 | 54.91 | non-CL conv inputs 4/100 |
| rx_cl_chunk | 149.15 | 149.3 | 80.5 | 8.01 | 59.75 | 54.91 | non-CL conv inputs 1/34 |
| rx_cl_compile | 68.6 | 70.25 | 174.9 | 8.63 | 59.92 | 54.98 |  |
| rx_chunk_compile | 93.22 | 94.67 | 128.7 | 11.82 | 59.93 | 54.98 |  |
| rx_cl_chunk_compile | 63.56 | 64.74 | 188.8 | 20.0 | 59.93 | 54.98 |  |
| rx_full | 63.53 | 64.55 | 188.9 | 20.0 | 59.93 | 54.98 |  |
| rx_full_fp16 | 65.15 | 66.56 | 184.2 | 20.0 | 80.05 | 61.83 |  |
| diffusers_bf16 (drift) | 192.14 | 192.34 | 62.5 | 4.44 | 59.75 | 54.91 |  |
| diffusers_bf16 (drift2) | 192.12 | 192.37 | 62.5 | 4.44 | 59.75 | 54.91 |  |

