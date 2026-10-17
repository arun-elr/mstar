"""Base vs head correctness facts for one model (limits are applied by ``report``, not here).

Per case and session: identical-output share and closeness for text, PSNR for images, video and frames, the
largest action difference, audio seconds, WER and transcript identity from the eval suites, and status codes for
request-path cases. Written to ``<model dir>/compare.json``.
"""

from __future__ import annotations

import difflib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

PCM_RATE = 24000


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * float(np.log10(255.0**2 / mse))


def load(p: Path) -> Any:
    if p.suffix == ".png":
        from PIL import Image

        return np.asarray(Image.open(p).convert("RGB"))
    if p.suffix == ".mp4":
        import av

        with av.open(str(p)) as c:
            return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    if p.suffix == ".u8":
        return np.frombuffer(p.read_bytes(), dtype=np.uint8)
    if p.suffix == ".pcm":
        return np.frombuffer(p.read_bytes(), dtype=np.int16)
    if p.suffix == ".f32":
        return np.frombuffer(p.read_bytes(), dtype=np.float32)
    return p.read_text(encoding="utf-8")


def text_facts(pairs: list[tuple[str, str]]) -> dict:
    """Identity plus how close the texts are: greedy decoding forks at near-ties across server processes, so a
    late fork is not a regression, while an empty, truncated or unrelated reply is."""
    sims, prefix, ratio = [], [], []
    for x, y in pairs:
        sims.append(difflib.SequenceMatcher(None, x, y, autojunk=False).ratio() if x or y else 1.0)
        k = next((i for i, (u, v) in enumerate(zip(x, y, strict=False)) if u != v), min(len(x), len(y)))
        prefix.append(k / max(len(x), len(y), 1))
        ratio.append(len(y) / len(x) if x else (1.0 if not y else 0.0))
    return {"text_identical": sum(x == y for x, y in pairs), "text_n": len(pairs),
            "text_sim_median": statistics.median(sims), "text_prefix_median": statistics.median(prefix),
            "text_len_ratio_median": statistics.median(ratio), "text_len_ratio_min": min(ratio)}


def saved(a: Path | None, b: Path | None) -> dict:
    """Facts from two directories of saved outputs, matched by file name."""
    if a is None or b is None or not a.is_dir() or not b.is_dir():
        return {"missing": [arm for arm, x in (("base", a), ("head", b)) if x is None or not x.is_dir()]}
    na, nb = {p.name for p in a.iterdir()}, {p.name for p in b.iterdir()}
    texts, ps, act, dur = [], [], [], []
    for n in sorted(na & nb):
        x, y = load(a / n), load(b / n)
        if isinstance(x, str):
            texts.append((x, y))
        elif n.endswith(".pcm"):
            dur.append((len(x) / PCM_RATE, len(y) / PCM_RATE))
        elif n.endswith(".f32"):
            act.append(float(np.max(np.abs(x - y))) if x.shape == y.shape else float("inf"))
        else:
            ps.append(psnr(x, y))
    out: dict = {"n_base": len({n.split(".")[0] for n in na}), "n_head": len({n.split(".")[0] for n in nb})}
    if texts:
        out.update(text_facts(texts))
    if ps:
        finite = [p for p in ps if not math.isnan(p)]
        out["psnr_min_db"] = min(finite) if len(finite) == len(ps) else None
        out["psnr_median_db"] = float(np.median(finite)) if finite else None
        out["shape_mismatch"] = len(ps) - len(finite)
    if dur:
        out["audio_s_base"] = round(sum(x for x, _ in dur), 2)
        out["audio_s_head"] = round(sum(y for _, y in dur), 2)
    if act:
        out["action_max_abs_diff"] = max(act)
    return out


def saved_dir(case_dir: Path, conc: int) -> Path | None:
    """Where the harness kept a point's outputs: ``audio/<label>-c<N>/r<repeat>/``."""
    dirs = sorted(case_dir.glob(f"audio/*-c{conc}/r*"))
    return dirs[-1] if dirs else None


def eval_file(d: Path, conc: int) -> dict | None:
    for f in sorted(d.glob(f"*-c{conc}.json")):
        js = json.loads(f.read_text())
        if js.get("schema") == "mstar-eval/1":
            return js
    return None


def suite_facts(a: Path, b: Path, conc: int) -> dict:
    ja, jb = eval_file(a, conc), eval_file(b, conc)
    if not ja or not jb:
        return {"missing": [arm for arm, j in (("base", ja), ("head", jb)) if not j]}
    out: dict = {}
    for k in ("wer", "sim_mean", "take_turn_rate", "yield_rate", "audio_s_mean"):
        va, vb = (ja["summary"].get(k) or {}).get("mean"), (jb["summary"].get(k) or {}).get("mean")
        if va is not None or vb is not None:
            out[f"{k}_base"], out[f"{k}_head"] = va, vb
    ha = {r["id"]: r.get("hyp") for r in ja["repeats"][0]["requests"] if r.get("ok")}
    hb = {r["id"]: r.get("hyp") for r in jb["repeats"][0]["requests"] if r.get("ok")}
    ids = sorted(set(ha) & set(hb))
    if ids and any(ha[i] is not None for i in ids):
        out.update(text_facts([(ha[i] or "", hb[i] or "") for i in ids]))
    return out


def status_of(v: dict) -> str:
    if "status" in v:
        return str(v["status"])
    if "status_line" in v:
        return v["status_line"].split(" ")[1] if " " in v["status_line"] else v["status_line"]
    if "close_code" in v:
        return f"ws{v['close_code']}"
    if "exception" in v:
        return f"exc:{v['exception']}"
    return "?"


def rp_facts(a: Path, b: Path) -> dict:
    fa, fb = a / "rp.json", b / "rp.json"
    if not fa.exists() or not fb.exists():
        return {"missing": [arm for arm, f in (("base", fa), ("head", fb)) if not f.exists()]}
    ja, jb = json.loads(fa.read_text()), json.loads(fb.read_text())
    diffs = []
    keys = sorted(k for k in set(ja) | set(jb) if not k.startswith("_"))
    for k in keys:
        va, vb = ja.get(k), jb.get(k)
        sa = status_of(va) if isinstance(va, dict) else "absent"
        sb = status_of(vb) if isinstance(vb, dict) else "absent"
        flags = [f for f in ("identical", "ok") if isinstance(va, dict) and f in va]
        if sa != sb or any(va.get(f) != (vb or {}).get(f) for f in flags):
            diffs.append({"case": k, "base": sa, "head": sb, **{f"{f}_base": va.get(f) for f in flags},
                          **{f"{f}_head": (vb or {}).get(f) for f in flags}})
    return {"n": len(keys), "diffs": diffs}


def compare_model(model_dir: Path, cases: list[dict], sessions: int) -> dict:
    """compare.json for one model's results directory (``<arm>/s<k>/<case>/``)."""
    res: dict = {}
    for case in cases:
        conc = min(case["conc"])
        per_session = []
        for s in range(sessions):
            a, b = (model_dir / arm / f"s{s}" / case["id"] for arm in ("base", "head"))
            if case["driver"] == "client":
                facts = saved(saved_dir(a, conc), saved_dir(b, conc))
            elif case["driver"] == "suite":
                facts = suite_facts(a, b, conc)
            else:
                facts = rp_facts(a, b)
            per_session.append(facts)
        if sessions > 1 and case["driver"] == "client":
            # Base against itself across two server processes: the noise floor for the identity checks.
            s0, s1 = (model_dir / "base" / f"s{s}" / case["id"] for s in (0, 1))
            per_session.append({"self_base": saved(saved_dir(s0, conc), saved_dir(s1, conc))})
        res[case["id"]] = per_session
    (model_dir / "compare.json").write_text(json.dumps(res, indent=1, default=str))
    return res
