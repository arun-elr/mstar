"""vLLM-Omni run dir -> summary CSVs in the column layout of M*'s summarize.py.

  python summarize_vo.py RUNS_DIR --flops-dir DIR --out OUT_DIR

RUNS_DIR layout (main.sh): main/<vae>/run{1,2,3}/, screen/wan21/, controls/<vae>/,
oneshot/<vae>/, profile/<vae>/*.kernels.json; M*'s FLOP counts (flops_<vae>.json from
profile_kernels.py --flops) from --flops-dir.
Writes variance.csv, runs.csv, profile.csv.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # benchmark/wan_vae_decoder: M*'s summarize.py

from summarize import PEAK_TBS, PEAK_TFLOPS, load, r2, run_row, three_runs, write_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--flops-dir", required=True, help="flops_wan21.json / flops_wan22.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    var_rows, means = [], {}
    for vae in ("wan21", "wan22"):
        rows = three_runs(runs / "main" / vae, vae)
        var_rows += rows
        means.update({(vae, r["variant"]): r["mean_ms"] for r in rows})
    write_csv(out / "variance.csv", var_rows)

    single = []
    for vae in ("wan21", "wan22"):
        for group in ("screen", "controls", "oneshot"):
            single += [run_row(vae, group, r) for r in load(runs / group / vae).values()]
        for p in sorted((runs / "main" / vae).glob("run*")):
            single += [run_row(vae, f"main/{p.name}", r) for r in load(p).values()]
    write_csv(out / "runs.csv", single)

    prof = []
    for vae in ("wan21", "wan22"):
        fl = json.loads((Path(args.flops_dir) / f"flops_{vae}.json").read_text())
        useful = fl["chunk_flops"] / 1e12
        intensity = fl["chunk_flops"] / fl["chunk_min_conv_bytes"]
        roof = min(PEAK_TFLOPS, intensity * PEAK_TBS)
        for kp in sorted((runs / "profile" / vae).glob("*.kernels.json")):
            k = json.loads(kp.read_text())
            v = k["variant"]
            ms = means.get((vae, v))
            achieved = useful / (ms / 1e3) if ms else None
            row = {"vae": vae, "variant": v, "kernels_per_chunk": k["kernels_per_chunk"],
                   "gpu_ms_per_chunk": k["gpu_ms_per_chunk"], "wall_ms_per_chunk": r2(ms),
                   "wall_source": "3-run mean" if ms else ""}
            for prefix, cats in (("ms_", k["category_ms_per_chunk"]),
                                 ("refined_ms_", k.get("category_ms_per_chunk_refined", {}))):
                row.update({prefix + c.replace(" / ", "_").replace(" ", "_"): v2 for c, v2 in cats.items()})
            row.update({
                "useful_tflop": round(useful, 2), "conv_tflop": round(fl["chunk_conv_flops"] / 1e12, 2),
                "intensity_flop_per_byte": round(intensity), "roof_tflops": roof,
                "achieved_tflops": round(achieved) if achieved else None,
                "pct_of_roof": round(achieved / roof * 100, 1) if achieved else None,
                "conv_kernel_tflops": round(
                    fl["chunk_conv_flops"] / 1e12 / (k["category_ms_per_chunk"]["convolution"] / 1e3)),
            })
            prof.append(row)
    write_csv(out / "profile.csv", prof)


if __name__ == "__main__":
    main()
