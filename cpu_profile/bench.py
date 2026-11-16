import argparse
import asyncio
import json
import os
import statistics
import time

from benchmark.base import ModelType, RequestType
from benchmark.runner import Benchmark, BenchmarkConfig, DatasetType, InferenceSystemType, ProfilingType


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    q = (len(xs) - 1) * p / 100
    i = int(q)
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (q - i)


def main():
    a = argparse.ArgumentParser()
    for name in ("arm", "head", "output", "prompts"):
        a.add_argument(f"--{name}", required=True)
    a.add_argument("--concurrency", type=int, required=True)
    a.add_argument("--requests", type=int, required=True)
    a.add_argument("--port", type=int, default=8620)
    args = a.parse_args()
    cfg = BenchmarkConfig(
        url=f"http://127.0.0.1:{args.port}",
        model=ModelType.QWEN3_5.inst(model_id="/home/bench/praxis/checkpoints/qwen3.5-0.8b"),
        dataset=DatasetType.TEXT,
        num_requests=args.requests,
        request_type=RequestType.T2T,
        local_cache_dir="/dev/shm/pr582-cache",
        num_warmup=2,
        profiling_type=ProfilingType.CLOSED_LOOP,
        inference_system=InferenceSystemType.OURS,
        max_concurrency=args.concurrency,
        request_txt_file=args.prompts,
        ignore_eos=True,
        output_len_min=64,
        output_len_max=64,
    )
    start = time.monotonic()
    metrics, agg = asyncio.run(Benchmark(cfg).run())
    wall_with_warmup = time.monotonic() - start
    ok = [m for m in metrics if m.error is None and m.e2e_latency is not None]
    times = [m.e2e_latency for m in ok]
    tokens = [m.output_text_tokens for m in ok]
    report = {
        "arm": args.arm, "head": args.head, "concurrency": args.concurrency,
        "requested": args.requests, "completed": len(ok), "errors": [{"id":m.request_id,"error":m.error} for m in metrics if m.error],
        "tokens_total": sum(tokens), "tokens_per_request": tokens,
        "matched_exactly_64_tokens": len(ok)==args.requests and all(t==64 for t in tokens),
        "request_throughput": agg.request_throughput,
        "text_token_throughput": agg.text_token_throughput,
        "latency_p50_s": pct(times,50), "latency_p95_s": pct(times,95),
        "per_request": [{"id":m.request_id,"latency_s":m.e2e_latency,"tokens":m.output_text_tokens,"ttft_s":m.ttft.get("text")} for m in metrics],
        "wall_with_warmup_s": wall_with_warmup,
        "dataset_sha256": os.popen(f"sha256sum {args.prompts}").read().split()[0],
    }
    with open(args.output,"w") as f:
        json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ("per_request","tokens_per_request")},indent=2))


if __name__ == "__main__":
    main()
