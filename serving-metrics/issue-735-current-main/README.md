# M* #735: current-main metrics throughput check

This is the raw evidence for an H200 A/B of `--enable-metrics` on M* main
`2c8ffde8f7`. It is a diagnostic run, not a performance fix. Both arms use
the same source tree: that commit plus the local-only
[`metrics_diagnostic.patch.gz`](scripts/metrics_diagnostic.patch.gz). The patch
selects `full` when metrics are enabled in this off/full comparison. No
diagnostic switch was proposed for production.

## Setup

- One NVIDIA H200 (143771 MiB, driver 580.173.02), Qwen3.5-0.8B from a local
  checkpoint, [`qwen35-metrics.yaml`](scripts/qwen35-metrics.yaml), SHM
  transport and `TORCHDYNAMO_DISABLE=1`.
- Python frontend and default Rust frontend were measured separately. The
  Rust server binary was built from the pinned main commit. The frontend used
  for a result is named in its chart and metadata.
- Fresh server for each arm. Four off/full pairs in ABBA order. Metrics arms
  were scraped every five seconds. Each start ran a discarded 256-request
  wave, then 2,600 measured requests at 64 clients, fixed 256 output tokens,
  `text_to_text`, `closed_loop`, `--ignore-eos`. All measured requests
  completed without failures.
- The workload is text-only. It does not measure metrics costs on audio,
  video, image, or upload routes. The paired deltas are observations from
  four starts, not confidence intervals.

## Results

| Frontend | Off mean tok/s | Full mean tok/s | Mean paired change | Observed paired range | TTFT p50 off/full (mean of run p50s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 17,639 | 17,460 | -1.02% | -1.24% to -0.77% | 45.06 / 50.74 ms |
| Rust | 18,264 | 18,031 | -1.27% | -2.29% to -0.21% | 51.11 / 55.82 ms |

The earlier roughly 4% Rust throughput gap did not reproduce on this main
commit. These runs do not isolate which metrics path accounts for the smaller
current gap; no production change follows from this measurement alone.

## Files

- `python/` and `rust/` each hold `manifest.csv`, per-start measured and warm
  `results.json`, chart PNG/SVG, and tabular chart inputs. `results.json`
  includes per-request latency data. The manifest preserves actual startup
  order and uses relative paths, so it can be re-plotted from this directory.
- `logs/` in each frontend has available server, client and metrics-scrape
  logs compressed as `.gz`. The first Python pair's server/client logs were
  not saved; its measured and warm JSON are present. The later six Python
  starts and all eight Rust starts have logs.
- [`plot_metrics_diagnostic.py`](scripts/plot_metrics_diagnostic.py) validates
  request counts, token totals and workload identity before plotting.
  `run_python_remaining.sh` is the exact script for the last six Python
  starts; `run_rust_abba.sh` is the exact Rust run script.

To regenerate a chart, install matplotlib and run, for example:

```bash
python scripts/plot_metrics_diagnostic.py python/manifest.csv \
  --out python/charts --mode preliminary --frontend python --commit 2c8ffde8f7
```

The plots show throughput, paired percent change, time to first text token,
and median inter-token latency. Diamonds are means, bars are observed ranges,
and dots are individual starts or pairs.
