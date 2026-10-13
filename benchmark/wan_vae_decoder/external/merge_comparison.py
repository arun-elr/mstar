"""comparison.csv: every engine's streaming decode on the same latents, from the
committed summaries.

  python external/merge_comparison.py --results results/b200

M* rows come from ``ladder_<vae>.csv`` (the change ladder through M*'s node;
``--mstar intree``: ``intree.csv`` and ``variance.csv`` instead),
stock diffusers from ``variance.csv`` / ``runs.csv`` / ``profile.csv``, the
frame-stream comparison from ``quality.csv``; vLLM-Omni and SGLang from
``external/{vllm_omni,sglang}/*.csv`` (their summarize scripts' output). The
uint8 column compares each engine with stock diffusers bf16 run on that
engine's own stack. ``chart`` marks the rows ``plot_results.py`` draws.
"""

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAES = ("wan21", "wan22")
STACK_MSTAR = "torch 2.12.1, cuDNN 9.20"
STACK_213 = "torch 2.13.0, cuDNN 9.20"


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get("vae") != "vae"]


def pick(rows, vae, variant, key):
    for r in rows:
        if r["vae"] == vae and r["variant"] == variant and r.get(key) not in (None, ""):
            return r[key]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE.parent / "results" / "b200"))
    ap.add_argument("--mstar", choices=["ladder", "intree"], default="ladder")
    ap.add_argument("--runs", default=str(HERE.parent / "runs" / "b200"), help="raw runs (in-tree kernel profiles)")
    args = ap.parse_args()
    d = Path(args.results)
    ladder = {vae: read(d / f"ladder_{vae}.csv") for vae in VAES if (d / f"ladder_{vae}.csv").exists()}
    variance, runs, profile, quality = (
        read(d / f) for f in ("variance.csv", "runs.csv", "profile.csv", "quality.csv"))
    useful = {r["vae"]: float(r["useful_tflop"]) for r in profile}
    vo = {f: read(d / "external" / "vllm_omni" / f"{f}.csv") for f in ("variance", "profile")}
    sg = {f: read(d / "external" / "sglang" / f"{f}.csv") for f in ("variance", "profile")}
    sg_md5 = [dict(r, variant=r["stream"]) for r in read(d / "external" / "sglang" / "frame_md5.csv")]

    intree = read(d / "intree.csv")

    def mstar_step(vae, step, key):
        if args.mstar == "ladder":
            r = next((r for r in ladder[vae] if r["mode"] == "compiled" and int(r["step"]) == step), None)
            return r[key] if r else None
        if step == 0:
            src = variance if vae == "wan21" else intree
            return pick(src, vae, "mstar_ar_ctx8", "mean_ms") if key == "mean_ms" else \
                pick(profile, vae, "mstar_ar_ctx8", "kernels_per_chunk")
        if key == "mean_ms":
            return next(r["mean_ms"] for r in intree
                        if r["vae"] == vae and r["variant"] == "mstar_warm" and r["session"] == "in-tree")
        kp = Path(args.runs) / "ablation" / ("profile_intree" if vae == "wan21" else "profile_intree_wan22") / \
            "mstar_warm.kernels.json"
        return json.loads(kp.read_text())["kernels_per_chunk"] if kp.exists() else None

    def stock_ms(vae, variant):
        return pick(variance, vae, variant, "mean_ms") or pick(runs, vae, variant, "median_ms")

    def u8(vae, variant):
        pct = pick(quality, vae, variant, "pct_u8_diff_ge1_vs_stock")
        return None if pct is None else ("identical" if float(pct) == 0 else f"{float(pct):.1f}% ≠")

    def sg_u8(vae, variant):
        pct = pick(sg_md5, vae, variant, "vs_diffusers_bf16_identical_bytes_pct")
        return None if pct is None else f"{100 - float(pct):.1f}% ≠"

    rows = []

    def add(engine, version, stack, setting, ms, kernels, u8s, source, chart=(True, True)):
        row = {"engine": engine, "version": version, "stack": stack, "setting": setting}
        for i, vae in enumerate(VAES):
            m = ms[i]
            row[f"{vae}_ms"] = f"{float(m):.2f}" if m is not None else ""
            k = kernels[i]
            row[f"{vae}_kernels"] = f"{float(k):.0f}" if k not in (None, "") else ""
            row[f"{vae}_tflops"] = f"{useful[vae] / (float(m) / 1e3):.0f}" if m is not None else ""
            row[f"{vae}_u8_vs_stock"] = u8s[i] or ""
        row["source"] = source
        row["chart"] = "+".join(v for v, c in zip(VAES, chart, strict=True) if c and row[f"{v}_ms"])
        rows.append(row)

    add("M* before", "upstream main", STACK_MSTAR, "stateless context re-decode, compiled",
        [mstar_step(v, 0, "mean_ms") for v in VAES], [mstar_step(v, 0, "kernels_per_chunk") for v in VAES],
        [u8(v, "mstar_ar_ctx8") for v in VAES],
        "ladder step 0" if args.mstar == "ladder" else "variance.csv / intree.csv")
    add("diffusers", "0.39.0", STACK_MSTAR, "stock, eager bf16",
        [stock_ms(v, "diffusers_bf16") for v in VAES],
        [pick(profile, v, "diffusers_bf16", "kernels_per_chunk") for v in VAES],
        ["identical", "identical"], "variance.csv / runs.csv")
    add("diffusers", "0.39.0", STACK_MSTAR, "stock + torch.compile, warm cache",
        [stock_ms(v, "diffusers_bf16_compile") for v in VAES], [None, None],
        [u8(v, "diffusers_bf16_compile") for v in VAES], "variance.csv / runs.csv", chart=(True, False))
    for variant, setting, chart in (("vllm_omni_default", "default (vae_fast_path=lossless)", True),
                                    ("vllm_omni_best", "best (--vae-fast-path channels_last)", True),
                                    ("vllm_omni_best_cudnnbench", "best + cudnn.benchmark", False)):
        add("vLLM-Omni", "f69b1f2b", STACK_213, setting,
            [pick(vo["variance"], v, variant, "mean_ms") for v in VAES],
            [pick(vo["profile"], v, variant, "kernels_per_chunk") for v in VAES],
            ["not hashed", "not hashed"], "external/vllm_omni", chart=(chart, chart))
    for wan21, wan22, setting, chart in (
            ("sglang_default", None, "default (quality=lossless, bf16)", (True, False)),
            (None, "sglang_default", "default as served (fp32 decode)", (False, False)),
            (None, "sglang_default_bf16", "default, bf16 decode", (False, True)),
            ("sglang_best", "sglang_best", "best (--quality extra-high, bf16)", (True, True)),
            ("sglang_compile", "sglang_compile_bf16", "--enable-torch-compile, bf16", (False, False))):
        names = (wan21, wan22)
        add("SGLang", "v0.5.21", STACK_213, setting,
            [pick(sg["variance"], v, n, "mean_ms") if n else None for v, n in zip(VAES, names, strict=True)],
            [pick(sg["profile"], v, n, "kernels_per_chunk") if n else None for v, n in zip(VAES, names, strict=True)],
            [sg_u8(v, n) if n else None for v, n in zip(VAES, names, strict=True)], "external/sglang", chart=chart)
    add("M* this PR", "in-tree node", STACK_MSTAR, "Cosmos3VAEDecoderARSubmodule defaults",
        [mstar_step(v, 5, "mean_ms") for v in VAES], [mstar_step(v, 5, "kernels_per_chunk") for v in VAES],
        [u8(v, "mstar_warm") for v in VAES], "ladder step 5" if args.mstar == "ladder" else "intree.csv")
    out = d / "comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
