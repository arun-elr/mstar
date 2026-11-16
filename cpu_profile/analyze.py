#!/usr/bin/env python3
"""Summarize the measured #582 worker CPU profile, without summing nested phases."""
import gzip
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phases import parse_log, segment

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
ARMS = ("main1", "pr1", "pr2", "main2")
KEYS = ("iter_total.cpu", "worker.gpu_thread.total.cpu", "worker.plan_thread.pre_plan.cpu", "worker.msg.new_request.runtime_add.cpu")
rows = []
for arm in ARMS:
    path = RAW / f"{arm}-server.log.gz"
    recs = parse_log(gzip.open(path, "rt").read())
    for line in (RAW / f"{arm}-windows.tsv").read_text().splitlines():
        c, t0, t1 = line.split()
        c = int(c)
        in_window = [r for r in recs if r.ts is not None and float(t0) <= r.ts <= float(t1)]
        candidates = segment(in_window, skip_warmup=3, bs_tolerance=.15, min_records=3)
        if not candidates:
            raise RuntimeError(f"{arm} c{c}: no steady segment")
        selected = max(candidates, key=lambda s: len(s.records))
        summaries = selected.summary()
        metrics = {}
        for k in KEYS:
            if k not in summaries:
                raise RuntimeError(f"{arm} c{c}: missing {k}")
            stats = summaries[k]
            metrics[k] = {
                "mean_cpu_ms_per_phase_call": stats["mean_ms"],
                "cpu_us_per_request_step": stats["us_per_1k"] / 1000,
                "samples": stats["samples"],
                "off_batch": stats["off_batch"],
            }
        client = json.loads((RAW / f"{arm}-c{c}.json").read_text())
        if not client["matched_exactly_64_tokens"]:
            raise RuntimeError(f"{arm} c{c}: mismatched output work")
        rows.append({
            "arm": arm, "source_head": json.loads((RAW / f"{arm}-metadata.json").read_text())["source_head"],
            "concurrency": c, "window_start_epoch": float(t0), "window_end_epoch": float(t1),
            "window_record_count": len(in_window),
            "steady_record_count": len(selected.records),
            "steady_bs_mean": selected.mean_bs,
            "steady_bs_range": list(selected.bs_range),
            "request_steps": selected.request_steps,
            "segments": [{"record_count": len(s.records), "bs_mean": s.mean_bs, "bs_range": list(s.bs_range)} for s in candidates],
            "completed": client["completed"], "tokens_total": client["tokens_total"],
            "request_throughput": client["request_throughput"],
            "metrics": metrics,
        })
result = {
    "workload": {"model": "Qwen3.5-0.8B", "output_tokens_per_request": 64, "warmup_waves": 2,
                 "order": list(ARMS), "phase_flush_interval": 100, "frontend": "Python", "rust_graph": True},
    "selection": "Per client window, drop first 3 phase flushes; segment by 15% stable actual batch; select longest segment (min 3 records).",
    "units": "cpu_us_per_request_step is microseconds of measured thread CPU for 1 output request-step. Different thread phases are not added. runtime_add.mean_cpu_ms_per_phase_call is cost per new request, not per output step.",
    "rows": rows,
}
comparisons = []
for c in (16, 64):
    group = [r for r in rows if r["concurrency"] == c]
    m = [r for r in group if r["arm"].startswith("main")]
    p = [r for r in group if r["arm"].startswith("pr")]
    if len(m) != 2 or len(p) != 2:
        raise RuntimeError(f"c{c}: expected two runs per source")
    for k in KEYS:
        field = "mean_cpu_ms_per_phase_call" if k.endswith("runtime_add.cpu") else "cpu_us_per_request_step"
        mv = [r["metrics"][k][field] for r in m]
        pv = [r["metrics"][k][field] for r in p]
        comparisons.append({"concurrency": c, "phase": k, "field": field, "main_values": mv, "pr_values": pv,
                            "main_mean": statistics.fmean(mv), "pr_mean": statistics.fmean(pv),
                            "delta_pct": 100*(statistics.fmean(pv)/statistics.fmean(mv)-1)})
result["comparisons"] = comparisons
(ROOT/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
for x in comparisons:
    print(f"c{x['concurrency']} {x['phase']}: main {x['main_mean']:.4f} PR {x['pr_mean']:.4f} delta {x['delta_pct']:+.2f}% ({x['field']})")
for c in (16,64):
    b = [r for r in rows if r["concurrency"] == c]
    print(f"c{c} actual batches: "+", ".join(f"{r['arm']} {r['steady_bs_mean']:.2f} n={r['steady_record_count']}" for r in b))

plt.rcParams.update({"font.size": 10})
colors={"main":"#3778bf","pr":"#e47b32"}
labels=[("iter_total.cpu","Worker main thread"),("worker.gpu_thread.total.cpu","GPU launch thread"),("worker.plan_thread.pre_plan.cpu","Plan pre-plan thread")]
fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=False)
for ax,c in zip(axes,(16,64)):
    grp=[r for r in rows if r["concurrency"]==c]
    for i,(k,label) in enumerate(labels):
        for source,dx in (("main",-.17),("pr",.17)):
            values=[r["metrics"][k]["cpu_us_per_request_step"] for r in grp if r["arm"].startswith(source)]
            xs=[i+dx-.035,i+dx+.035]
            ax.scatter(xs, values, marker="o", s=43, color=colors[source], zorder=3, label=source.upper() if i==0 else None)
            ax.hlines(statistics.fmean(values),i+dx-.095,i+dx+.095,color=colors[source],lw=2)
    ax.set_xticks(range(len(labels)),[x[1] for x in labels],rotation=18,ha="right")
    ax.set_title(f"Client c{c}, actual batch ~{statistics.fmean(r['steady_bs_mean'] for r in grp):.1f}")
    ax.set_ylabel("CPU µs per output request-step")
    ax.grid(axis="y",alpha=.22)
    ax.set_ylim(bottom=0)
axes[0].legend(frameon=False)
fig.suptitle("#582: worker thread CPU, matched 64-token Qwen requests\nTwo server restarts per source; phase timing enabled in both")
fig.tight_layout()
fig.savefig(ROOT/"worker_cpu_per_step.png",dpi=180,bbox_inches="tight")
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=False)
for ax,c in zip(axes,(16,64)):
    grp=[r for r in rows if r["concurrency"]==c]
    for source,ix in (("main",0),("pr",1)):
        values=[r["metrics"]["worker.msg.new_request.runtime_add.cpu"]["mean_cpu_ms_per_phase_call"]*1000 for r in grp if r["arm"].startswith(source)]
        ax.scatter([ix-.04,ix+.04],values,s=48,color=colors[source],zorder=3)
        ax.hlines(statistics.fmean(values),ix-.14,ix+.14,color=colors[source],lw=2)
    ax.set_xticks([0,1],["main","PR"])
    ax.set_xlim(-.4,1.4)
    ax.set_title(f"Client c{c}")
    ax.set_ylabel("CPU µs per new-request insertion")
    ax.grid(axis="y",alpha=.22)
    ax.set_ylim(45,60)
fig.suptitle("#582: new-request insertion CPU\nPer-call worker main-thread phase; two restarts per source; zoomed 45–60 µs axis")
fig.tight_layout()
fig.savefig(ROOT/"request_insert_cpu.png",dpi=180,bbox_inches="tight")
plt.close(fig)
