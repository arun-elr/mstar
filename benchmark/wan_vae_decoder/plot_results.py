"""Charts for the Wan VAE decoder results, from the committed summary CSVs.

  python plot_results.py [--results results/b200]   # writes <results>/charts/*.png

Needs only matplotlib. Inputs: variance.csv, runs.csv, profile.csv, comparison.csv,
intree.csv, quality_frames.csv.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BEFORE = "#b5473f"
STEP = "#5b8db8"
AFTER = "#2f9e5b"
STOCK = "#8f8f8f"
CATEGORY_COLORS = {
    "convolution": "#4c72b0",
    "layout conversion": "#dd8452",
    "norm / SiLU / concat / pad": "#55a868",
    "other": "#c4c4c4",
}
PEAK_TFLOPS = 2250
SETUP = "Wan VAE decode, one B200, bf16, 3-latent chunk (12 frames at 480×832)"
FRAMES_PER_CHUNK = 12

PROFILED = [
    ("stock diffusers\n(eager)", "diffusers_bf16", STOCK),
    ("M* before\n(compiled)", "mstar_ar_ctx8", BEFORE),
    ("M* this change\n(prototype)", "px_nopad_bench", AFTER),
]


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def num(x):
    return float(x) if x not in ("", None) else None


def ms1(v: float) -> str:
    """One decimal, halves rounded up (335.25 -> 335.3, as in the tables)."""
    return f"{v + 1e-9:.1f}"


class Results:
    def __init__(self, d: Path):
        self.variance = {r["variant"]: r for r in read(d / "variance.csv")}
        self.runs = read(d / "runs.csv")
        self.profile = {(r["vae"], r["variant"]): r for r in read(d / "profile.csv")}
        self.comparison = read(d / "comparison.csv")
        self.intree = {(r["vae"], r["variant"]): float(r["mean_ms"])
                       for r in read(d / "intree.csv") if r["session"] == "in-tree"}
        self.ladder = {vae: read(d / f"ladder_{vae}.csv") for vae in ("wan21", "wan22")
                       if (d / f"ladder_{vae}.csv").exists()}

    def ms(self, variant: str, vae: str = "wan21") -> tuple[float, bool]:
        """(ms per chunk, is a 3-run mean): the 3-run mean if there is one, else the
        ablation run, else the run_all suite."""
        if vae == "wan21" and variant in self.variance:
            return float(self.variance[variant]["mean_ms"]), True
        rows = [r for r in self.runs if r["vae"] == vae and r["variant"] == variant]
        rows.sort(key=lambda r: not r["group"].startswith("ablation"))
        return float(rows[0]["median_ms"]), False


def style(ax, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def footnote(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8, color="#555555", ha="left", va="bottom")


LADDER_LABELS = {
    0: "M* before\n(stateless\nre-decode)",
    1: "1. warm\nconv cache",
    2: "2 + 4. channels-last,\nconv-internal\npadding",
    3: "3. one decoder\ncall per chunk",
    5: "5. cudnn.\nbenchmark",
}


def ladder(res: Results, out: Path):
    """The change ladder through M*'s decoder node, compiled and eager, one panel per VAE."""
    vaes = [v for v in ("wan21", "wan22") if v in res.ladder]
    if not vaes:
        return
    fig, axes = plt.subplots(1, len(vaes), figsize=(7.2 * len(vaes), 5.8), squeeze=False)
    w = 0.38
    for ax, vae in zip(axes[0], vaes, strict=True):
        rows = res.ladder[vae]
        steps = [int(r["step"]) for r in rows if r["mode"] == "compiled"]
        by = {(r["mode"], int(r["step"])): r for r in rows}
        xs = list(range(len(steps)))
        base = float(by[("compiled", steps[0])]["mean_ms"])
        for off, mode, color in ((-w / 2, "compiled", None), (w / 2, "eager", "#c9d7e6")):
            vals = [float(by[(mode, st)]["mean_ms"]) if (mode, st) in by else 0.0 for st in steps]
            colors = color or [BEFORE] + [STEP] * (len(steps) - 2) + [AFTER]
            ax.bar([x + off for x in xs], vals, w, color=colors, label=mode)
            for x, v in zip(xs, vals, strict=True):
                if not v:
                    continue
                txt = f"{ms1(v)}" if mode == "eager" else f"{ms1(v)} ms\n{base / v:.1f}×"
                ax.text(x + off, v + base * 0.012, txt, ha="center", va="bottom",
                        fontsize=8 if mode == "eager" else 9, color="#555555" if mode == "eager" else "black")
        ax.set_xticks(xs, [LADDER_LABELS[st] for st in steps], fontsize=8.5)
        top = max(float(r["mean_ms"]) for r in rows)
        ax.set_ylim(0, top * 1.15)
        style(ax, "ms per chunk (lower is better)")
        ax.set_title("Wan 2.1 VAE" if vae == "wan21" else "Wan 2.2 VAE", fontsize=11)
    axes[0][0].legend(handles=[Patch(color=BEFORE, label="M* before: stateless re-decode (compiled)"),
                               Patch(color=STEP, label="intermediate step (compiled)"),
                               Patch(color=AFTER, label="all changes on (compiled)"),
                               Patch(color="#c9d7e6", label="eager, same step")],
                      frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(f"One change at a time, through M*'s decoder node\n{SETUP}", fontsize=11)
    footnote(fig, "n = 3 fresh-process runs per bar (fresh compile each). Speedup vs M* before (compiled). "
                  "Each step adds its switch to the one before; step 4 (conv-internal padding) comes with step 2.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out / "ladder.png", dpi=150)
    plt.close(fig)


def breakdown(res: Results, out: Path):
    cats = {name: [] for name in CATEGORY_COLORS}
    totals = []
    for _, v, _ in PROFILED:
        p = res.profile[("wan21", v)]
        conv, layout = float(p["ms_convolution"]), float(p["ms_layout_conversion"])
        norm = float(p["ms_norm_SiLU"]) + float(p["ms_concat_pad"])
        gpu = float(p["gpu_ms_per_chunk"])
        cats["convolution"].append(conv)
        cats["layout conversion"].append(layout)
        cats["norm / SiLU / concat / pad"].append(norm)
        cats["other"].append(gpu - conv - layout - norm)
        totals.append(gpu)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    xs = list(range(len(PROFILED)))
    bottom = [0.0] * len(xs)
    for name, vals in cats.items():
        ax.bar(xs, vals, bottom=bottom, color=CATEGORY_COLORS[name], width=0.55, label=name)
        for x, (b, v) in enumerate(zip(bottom, vals, strict=True)):
            if v >= 12:
                ax.text(x, b + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=9, color="white")
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    for x, t in enumerate(totals):
        ax.text(x, t + 4, f"{t:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(xs, [label for label, _, _ in PROFILED])
    ax.set_ylim(0, max(totals) * 1.12)
    style(ax, "GPU kernel time per chunk (ms)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Where the decode time goes (torch.profiler, Wan 2.1)\n{SETUP}", fontsize=11)
    footnote(fig, "Kernel categories from kernel names. 'other' = copies, elementwise, upsample, attention.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "profile_breakdown.png", dpi=150)
    plt.close(fig)


def kernels_tflops(res: Results, out: Path):
    rows = [res.profile[("wan21", v)] for _, v, _ in PROFILED]
    labels = [label for label, _, _ in PROFILED]
    colors = [c for _, _, c in PROFILED]
    xs = list(range(len(rows)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    kernels = [float(r["kernels_per_chunk"]) for r in rows]
    a1.bar(xs, kernels, color=colors, width=0.55)
    for x, k in zip(xs, kernels, strict=True):
        a1.text(x, k + max(kernels) * 0.01, f"{k:,.0f}", ha="center", va="bottom", fontsize=10)
    a1.set_xticks(xs, labels, fontsize=9)
    a1.set_ylim(0, max(kernels) * 1.12)
    style(a1, "GPU kernels per chunk")
    a1.set_title("Kernel launches per chunk", fontsize=11)
    tflops = [float(r["achieved_tflops"]) for r in rows]
    a2.bar(xs, tflops, color=colors, width=0.55)
    for x, t in zip(xs, tflops, strict=True):
        a2.text(x, t + 25, f"{t:.0f}\n({t / PEAK_TFLOPS:.0%})", ha="center", va="bottom", fontsize=10)
    a2.axhline(PEAK_TFLOPS, color="#555555", linestyle="--", linewidth=1)
    a2.text(len(xs) - 0.5, PEAK_TFLOPS - 40, f"dense bf16 peak {PEAK_TFLOPS:,}", ha="right", va="top", fontsize=9)
    a2.set_xticks(xs, labels, fontsize=9)
    a2.set_ylim(0, PEAK_TFLOPS * 1.08)
    style(a2, "achieved TFLOP/s")
    a2.set_title("Useful TFLOP/s (40.65 TFLOP per chunk / wall time)", fontsize=11)
    fig.suptitle(f"Kernel count and compute efficiency, Wan 2.1. {SETUP}", fontsize=10)
    footnote(fig, "M* before executes 139.9 TFLOP per chunk (3.4× the useful work) because it re-decodes "
                  "8 context latents; achieved counts useful work only.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "kernels_tflops.png", dpi=150)
    plt.close(fig)


def wan21_vs_wan22(res: Results, out: Path):
    groups = [("Wan 2.1 VAE\n(16 ch, 8× spatial)", "wan21"), ("Wan 2.2 VAE\n(48 ch, 16× spatial)", "wan22")]
    def step(vae, n):
        if vae not in res.ladder:
            return res.intree[(vae, "mstar_ar_ctx8" if n == 0 else "mstar_warm")]
        return next(float(r["mean_ms"]) for r in res.ladder[vae] if r["mode"] == "compiled" and int(r["step"]) == n)

    before = [step(vae, 0) for _, vae in groups]
    after = [step(vae, 5) for _, vae in groups]
    xs = list(range(len(groups)))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.bar([x - w / 2 for x in xs], before, w, color=BEFORE, label="M* before")
    ax.bar([x + w / 2 for x in xs], after, w, color=AFTER, label="M* this change (in-tree node)")
    for x, b, a in zip(xs, before, after, strict=True):
        ax.text(x - w / 2, b + 4, f"{ms1(b)} ms", ha="center", va="bottom", fontsize=10)
        ax.text(x + w / 2, a + 4, f"{ms1(a)} ms\n{b / a:.1f}× faster", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs, [g for g, _ in groups])
    ax.set_ylim(0, max(before) * 1.15)
    style(ax, "ms per chunk (lower is better)")
    ax.legend(frameon=False)
    ax.set_title(f"Wan 2.1 and Wan 2.2 VAE, before and after\n{SETUP}", fontsize=11)
    footnote(fig, "3-run means, fresh process and compile each, through M*'s decoder node.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "wan21_vs_wan22.png", dpi=150)
    plt.close(fig)


FAMILY_COLORS = {"M* this PR": AFTER, "M* before": BEFORE, "vLLM-Omni": "#4c72b0", "SGLang": "#dd8452",
                 "diffusers": STOCK}


def comparison(res: Results, out: Path):
    """Every engine's streaming decode on the same latents, one panel per VAE (``chart`` rows)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, vae, name in ((axes[0], "wan21", "Wan 2.1 VAE"), (axes[1], "wan22", "Wan 2.2 VAE")):
        rows = [(f"{r['engine']}: {r['setting']}", float(r[f"{vae}_ms"]), r["engine"])
                for r in res.comparison if vae in r.get("chart", "").split("+") and r[f"{vae}_ms"]]
        rows.sort(key=lambda t: t[1])
        ys = list(range(len(rows)))
        for y, (_, v, engine) in zip(ys, rows, strict=True):
            ax.barh(y, v, color=FAMILY_COLORS[engine], height=0.62)
            ax.text(v + 4, y, f"{ms1(v)} ms", va="center", fontsize=9)
        ax.set_yticks(ys, [r[0] for r in rows], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(r[1] for r in rows) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("ms per 3-latent chunk (lower is better)")
        ax.set_title(name, fontsize=11)
    fig.suptitle("Wan VAE streaming decode across engines, one B200\n"
                 "same latents, same B200, measured with this benchmark; "
                 "each engine via its own loader and documented flags", fontsize=11)
    footnote(fig, "3-run means. vLLM-Omni and SGLang ran on torch 2.13 / cuDNN 9.20, M* and stock diffusers on "
                  "torch 2.12.1 / cuDNN 9.20; M*'s prototype harness runs 53.45 ms (Wan 2.1) on the torch 2.13 stack. "
                  "SGLang's Wan 2.2 default decodes fp32 as served: 171.0 ms.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)


QUALITY_LINES = [  # (label, variant, color, linestyle)
    ("M* before", "mstar_ar_ctx8", BEFORE, "-"),
    ("stock diffusers bf16", "diffusers_bf16", STOCK, "-"),
    ("prototype harness, all five changes", "px_nopad_bench", STEP, "-"),
    ("M* this change (in-tree node)", "mstar_warm", AFTER, "-"),
]


def per_frame_psnr(res_dir: Path, out: Path):
    path = res_dir / "quality_frames.csv"
    if not path.exists():
        return
    rows = read(path)
    vaes = [v for v in ("wan21", "wan22") if any(r["vae"] == v for r in rows)]
    fig, axes = plt.subplots(len(vaes), 2, figsize=(13, 3.6 * len(vaes)), squeeze=False)
    cap = 80.0
    for i, vae in enumerate(vaes):
        for j, (col, title) in enumerate((("psnr_u8_vs_fp32", "vs fp32 reference (TF32 off)"),
                                          ("psnr_u8_vs_stock", "vs stock diffusers bf16"))):
            ax = axes[i][j]
            n = 0
            for label, v, color, ls in QUALITY_LINES:
                pts = [r for r in rows if r["vae"] == vae and r["variant"] == v]
                if not pts or (j == 1 and v == "diffusers_bf16"):
                    continue
                ys = [min(float(r[col]), cap) for r in pts]
                n = len(ys)
                ax.plot(range(n), ys, color=color, linestyle=ls, linewidth=0.9, label=label)
            for b in range(1, n, FRAMES_PER_CHUNK):
                ax.axvline(b, color="#000000", alpha=0.06, linewidth=0.5)
            style(ax, "PSNR uint8 (dB)" if j == 0 else None)
            ax.set_xlim(0, n)
            ax.set_ylim(bottom=40)
            name = "Wan 2.1" if vae == "wan21" else "Wan 2.2"
            ax.set_title(f"{name} VAE, {title}", fontsize=10)
            if j == 1:
                ax.set_ylim(top=cap + 2)
                ax.text(n, cap, f"≥{cap:.0f} dB = identical", ha="right", va="bottom", fontsize=8, color="#555555")
            if i == len(vaes) - 1:
                ax.set_xlabel("output frame (faint lines: chunk boundaries, every 12 frames)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.suptitle(f"Per-frame PSNR over the whole 85-chunk stream. {SETUP}", fontsize=10)
    footnote(fig, "Axis cut at 40 dB: M* before reaches 34.8 dB on Wan 2.2; the prototype harness's frames 1-20 on "
                  "Wan 2.2 (its compiled 2nd decoder call) drop to 12-23 dB. The in-tree node runs that call eagerly.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out / "per_frame_psnr.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results" / "b200"))
    args = ap.parse_args()
    d = Path(args.results)
    out = d / "charts"
    out.mkdir(exist_ok=True)
    res = Results(d)
    for chart in (ladder, breakdown, kernels_tflops, wan21_vs_wan22, comparison):
        chart(res, out)
    per_frame_psnr(d, out)
    print(f"wrote {', '.join(sorted(p.name for p in out.glob('*.png')))} to {out}")


if __name__ == "__main__":
    main()
