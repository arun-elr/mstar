import json
import statistics
import sys

import matplotlib.pyplot as plt

main = json.load(open(sys.argv[1]))["runs"]
pr = json.load(open(sys.argv[2]))["runs"]
fig, (latency, outcomes) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
colors = ["#596579", "#117e72"]
for x, rows, label, color in ((0, main, "main", colors[0]), (1, pr, "PR #1551", colors[1])):
    vals = [row["plain_chat_ms"] for row in rows]
    latency.scatter([x - 0.07, x, x + 0.07], vals, color=color, s=45, zorder=3)
    med = statistics.median(vals)
    latency.hlines(med, x - 0.2, x + 0.2, color=color, linewidth=2)
    latency.annotate(f"median {med:g} ms", (x, med), xytext=(0, 11),
                     textcoords="offset points", ha="center", color=color, fontsize=9)
latency.set_xticks([0, 1], ["main", "PR #1551"])
latency.set_xlim(-0.5, 1.5)
latency.set_yscale("log")
latency.set_ylim(1, 10000)
latency.set_ylabel("Plain chat latency (ms, log scale)")
latency.set_title("Plain chat while remote fetches wait")
latency.grid(axis="y", alpha=0.25)

upstream = [main[0]["remote_upstream_hits"], pr[0]["remote_upstream_hits"]]
busy = [main[0]["remote_busy_refusals"], pr[0]["remote_busy_refusals"]]
outcomes.bar([0, 1], upstream, color=colors[0], label="Upstream requests")
outcomes.bar([0, 1], busy, bottom=upstream, color=colors[1], label="Immediate cap refusal")
outcomes.set_xticks([0, 1], ["main", "PR #1551"])
outcomes.set_ylim(0, 54)
outcomes.set_ylabel("Remote URL requests (of 48)")
outcomes.set_title("Remote URL outcomes")
outcomes.legend(loc="upper right", frameon=False, fontsize=8)
for x in range(2):
    outcomes.text(x, upstream[x] / 2, str(upstream[x]), ha="center", va="center", color="white")
    if busy[x]:
        outcomes.text(x, upstream[x] + busy[x] / 2, str(busy[x]), ha="center", va="center", color="white")

fig.suptitle("48 slow URLs, 2 s upstream hold, local CPU frontend test", fontsize=12)
fig.text(0.5, 0.01, "Three runs per arm; Bagel stub; main 2c8ffde8f7 vs PR 505acbf6f2; no GPU model work",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.045, 1, 0.95])
fig.savefig(sys.argv[3], dpi=180)
