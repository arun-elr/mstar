"""Rebuild the PR #1556 H200 figures from the raw per-request reports."""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "raw"
ARMS = {"main": ("main1", "main2"), "pr": ("pr1", "pr2")}
HEADS = {"main": "2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175",
         "pr": "258b81371bc4626797177a5c93a14dad23405961"}
COUNTS = {1: 320, 8: 1280, 16: 2560}
COLORS = {"main": "#2563eb", "pr": "#dc2626"}
LABELS = {"main": "Upstream main", "pr": "PR #1556"}


def load():
    out = {}
    for branch, arms in ARMS.items():
        for arm in arms:
            meta = json.loads((RAW / f"{arm}-metadata.json").read_text())
            assert meta["source_head"] == HEADS[branch]
            out[arm] = {}
            for c, n in COUNTS.items():
                r = json.loads((RAW / f"{arm}-long-c{c}.json").read_text())
                assert r["head"] == HEADS[branch] and r["concurrency"] == c
                assert r["requested"] == r["completed"] == n and not r["errors"]
                assert r["matched_exactly_64_tokens"] and r["tokens_total"] == n * 64
                assert all(x == 64 for x in r["tokens_per_request"])
                assert r["dataset_sha256"] == meta["prompts_sha256"]
                r["measurement_wall_s"] = n / r["request_throughput"]
                out[arm][c] = r
    refs = [json.loads((RAW / f"{arm}-metadata.json").read_text()) for arms in ARMS.values() for arm in arms]
    for key in ("config_sha256", "prompts_sha256", "checkpoint_revision", "checkpoint_safetensors_sha256", "hardware"):
        assert len({m[key] for m in refs}) == 1, key
    return out


def value(r, metric):
    if metric == "request_throughput":
        return r[metric]
    if metric == "text_token_throughput":
        return r[metric]
    return r[metric] * 1000


def panel(ax, data, metric, ylabel):
    for branch, arms in ARMS.items():
        xs = np.array(list(COUNTS))
        ys = np.array([[value(data[arm][c], metric) for arm in arms] for c in xs])
        avg = ys.mean(axis=1)
        low = avg - ys.min(axis=1)
        high = ys.max(axis=1) - avg
        ax.errorbar(xs, avg, yerr=[low, high], marker="o", linewidth=2.2, capsize=4,
                    color=COLORS[branch], label=LABELS[branch])
        for j in range(2):
            ax.scatter(xs + (-0.17 if j == 0 else 0.17), ys[:, j], s=20,
                       color=COLORS[branch], alpha=0.45)
    ax.set_xticks(list(COUNTS))
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def figure(data, metrics, labels, outfile, title):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    for ax, metric, label in zip(axes, metrics, labels):
        panel(ax, data, metric, label)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(title + "\nH200 · Qwen3.5-0.8B · 64 tokens/request · 2 restarts/arm", fontsize=13)
    fig.savefig(ROOT / outfile, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    data = load()
    summary = {"order": ["main1", "pr1", "pr2", "main2"], "concurrency": {}}
    for c in COUNTS:
        cells = {}
        for branch, arms in ARMS.items():
            cells[branch] = [{k: data[a][c][k] for k in (
                "requested", "completed", "tokens_total", "request_throughput",
                "text_token_throughput", "latency_p50_s", "latency_p95_s", "measurement_wall_s")}
                for a in arms]
        main_mean = np.mean([v["request_throughput"] for v in cells["main"]])
        pr_mean = np.mean([v["request_throughput"] for v in cells["pr"]])
        summary["concurrency"][str(c)] = {"main": cells["main"], "pr": cells["pr"],
                                           "pr_vs_main_request_throughput_pct": 100 * (pr_mean / main_mean - 1)}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure(data, ("request_throughput", "text_token_throughput"),
           ("Completed requests/s", "Generated text tokens/s"), "throughput.png", "PR #1556 H200 served throughput")
    figure(data, ("latency_p50_s", "latency_p95_s"),
           ("Median latency (ms)", "P95 latency (ms)"), "latency.png", "PR #1556 H200 served latency")
    print(json.dumps({c: x["pr_vs_main_request_throughput_pct"] for c, x in summary["concurrency"].items()}, indent=2))


if __name__ == "__main__":
    main()
