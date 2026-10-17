"""benchmark.eval compare on CPU: the e2e manifest, base-vs-head checks, verdicts (perf noise, A/A floors and
correctness limits, length-matched latency, ABBA reruns), calibration, the e2e suite against a fake server, the
GPU-slot queue, and one model run end to end through the SSH code path with a fake transport and fake servers."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import threading
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

pytest.importorskip("aiohttp")

from benchmark.eval.e2e import calibrate, checks, compare, hosts, report, trees  # noqa: E402
from benchmark.eval.e2e import manifest as mf  # noqa: E402
from benchmark.eval.e2e.manifest import Case, Model  # noqa: E402
from benchmark.eval.e2e.report import Perf, Stat  # noqa: E402

RUN = {"id": "r", "base": {"label": "main", "sha": "a" * 40}, "head": {"label": "PR #1", "sha": "b" * 40},
       "gpu": "h200", "perf_pct": 5, "noise_sigma": 2, "gpu_hours": 1.0}

# ---------------------------------------------------------------- manifest

BASE = {
    "defaults": {"conc": [1, 8], "repeats": 3, "thresholds": {"perf_pct": 5}, "env": {"A": 1}},
    "datasets": {"tts": "tts/manifest.jsonl", "clone": "clone/manifest.jsonl"},
    "models": {
        "kokoro": {"serve": "kokoro", "family": "TTS", "tags": ["tts", "small"],
                   "cases": [{"id": "tts", "suite": "tts", "data": "tts", "check": {"kind": "wer", "max_increase": 1},
                              "perf": {"latency": "e2e_s_p50", "throughput": "audio_s_per_s"}},
                             {"id": "inv", "invariants": "speech", "optional": True,
                              "check": {"kind": "status_match"}}]},
        "z": {"serve": "z_image_turbo", "family": "Image", "tags": ["image-gen"], "est_min": 3,
              "thresholds": {"perf_pct": 8},
              "cases": [{"id": "t2i", "client": "image", "conc": [1], "args": ["--size", 512],
                         "check": {"kind": "psnr", "min_db": 30}, "perf": {"latency": "lat_p50_s"}}]},
    },
}


def test_manifest_defaults_and_overrides():
    m = mf.parse(copy.deepcopy(BASE))
    k, z = m.models["kokoro"], m.models["z"]
    assert k.cases[0].conc == [1, 8] and k.cases[0].repeats == 3
    assert z.cases[0].conc == [1] and z.cases[0].args == ["--size", "512"]
    assert z.est_min == 3 and k.est_min == 10 and k.env == {"A": "1"}
    assert z.thresholds["perf_pct"] == 8 and k.thresholds["perf_pct"] == 5
    assert Model.from_dict(json.loads(json.dumps(z.to_dict()))) == z


def test_manifest_select_tags_names_exclusions_optional():
    m = mf.parse(copy.deepcopy(BASE))
    assert [x.name for x in m.select("all")] == ["kokoro", "z"]
    assert [x.name for x in m.select("tag:tts")] == ["kokoro"]
    assert [x.name for x in m.select("tag:image")] == ["z"]          # family match
    assert [x.name for x in m.select("all,-tag:small")] == ["z"]
    assert [c.id for c in m.select("kokoro")[0].cases] == ["tts"]
    assert [c.id for c in m.select("kokoro", include_optional=True)[0].cases] == ["tts", "inv"]
    with pytest.raises(mf.ManifestError, match="unknown model"):
        m.select("nope")
    raw = copy.deepcopy(BASE)
    raw["models"]["kokoro_mc"] = {**copy.deepcopy(raw["models"]["kokoro"]), "optional": True,
                                  "tags": ["tts", "mooncake"]}
    m = mf.parse(raw)
    assert [x.name for x in m.select("all")] == ["kokoro", "z"]
    assert [x.name for x in m.select("tag:tts")] == ["kokoro"]           # a tag regular models share
    assert [x.name for x in m.select("tag:mooncake")] == ["kokoro_mc"]


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["models"]["kokoro"]["cases"][0].pop("check"), "missing 'check'"),
    (lambda r: r["models"]["kokoro"]["cases"][0].update(client="image"), "exactly one of"),
    (lambda r: r["models"]["kokoro"]["cases"][0].update(data="nope"), "unknown dataset"),
    (lambda r: r["models"]["kokoro"]["cases"][0]["check"].update(kind="vibes"), "check.kind"),
    (lambda r: r["models"]["kokoro"]["cases"][0]["check"].pop("max_increase"), "needs max_increase"),
    (lambda r: r["models"]["z"]["cases"][0].update(client="teleport"), "client must be one of"),
    (lambda r: r["models"]["z"]["cases"].append(dict(r["models"]["z"]["cases"][0])), "duplicate case ids"),
    (lambda r: r["models"]["z"]["cases"][0].update(conc=[0]), "positive ints"),
    (lambda r: r["models"]["z"]["cases"][0].update(perf={}), "perf needs"),
    (lambda r: r["models"]["z"]["cases"][0].update(data="nope"), "unknown dataset"),
])
def test_manifest_validation_errors(mutate, msg):
    raw = copy.deepcopy(BASE)
    mutate(raw)
    with pytest.raises(mf.ManifestError, match=msg):
        mf.parse(raw)


def test_shipped_manifest():
    m = mf.load()
    regular = [x for x in m.models.values() if not x.optional]
    assert len(regular) == 16
    assert {c.kind for x in m.models.values() for c in x.cases} >= {"tts", "asr", "duplex", *mf.CLIENT_KINDS}
    assert {c.check["kind"] for x in m.models.values() for c in x.cases} == set(mf.CHECKS)
    mc = m.select("tag:mooncake")
    assert {x.name for x in mc} == {"qwen3_omni_mooncake", "orpheus_mooncake"}
    assert all(x.optional and x.serve_args[:2] == ["--tensor-comm-protocol", "TCP"] for x in mc)
    for x in m.models.values():
        for c in x.cases:
            assert not c.data or c.data in m.datasets
    # every calibration entry names a shipped model and case
    cases = {(x.name, c.id) for x in m.models.values() for c in x.cases}
    for key in (k for k in calibrate.load() if not k.startswith("_")):
        model, case = key.split(":", 1)[1].split("/")[:2]
        assert (model, case) in cases, key


# ---------------------------------------------------------------- verdicts

def perf(base, head, lower=True, metric="lat_p50_s"):
    return report.judge_perf(Perf(metric, 1, Stat.of(base), Stat.of(head), lower), perf_pct=5, noise_sigma=2)


def test_judge_perf_same_regression_better_noisy():
    assert perf([1.0, 1.0, 1.0], [1.02, 1.02, 1.02]).verdict == "same"
    p = perf([1.0, 1.0, 1.0], [1.10, 1.10, 1.10])
    assert p.verdict == "regression" and p.delta_pct == pytest.approx(-10)
    assert perf([1.0, 1.0, 1.0], [0.8, 0.8, 0.8]).verdict == "better"
    assert perf([1.0, 1.4, 0.6], [1.1, 1.5, 0.7]).verdict == "noisy"     # +10% but the spread is wider
    t = perf([10, 10, 10], [9, 9, 9], lower=False, metric="req_per_s")
    assert t.verdict == "regression" and t.delta_pct == pytest.approx(-10)
    assert perf([], [1.0]).verdict == "n/a"
    assert report.lower_is_better("lat_p50_s") and report.lower_is_better("ttfa_ms_p50")
    assert not any(map(report.lower_is_better, ("req_per_s", "audio_s_per_s", "take_turn_rate", "rtfx")))


def case(kind, **ck):
    return Case(id="c", modality="m", driver="client", kind="image", check={"kind": kind, **ck})


@pytest.mark.parametrize("c,facts,errors,want", [
    (case("text_match", min_match=0.5), [{"text_n": 16, "text_identical": 15}], {}, "pass"),
    (case("text_match", min_match=0.5), [{"text_n": 16, "text_identical": 4}], {}, "FAIL"),
    (case("text_match", min_match=0.5), [{"text_n": 16, "text_identical": 16}], {"base": 0, "head": 2}, "FAIL"),
    (case("wer", max_increase=1.0), [{"wer_base": 0.020, "wer_head": 0.025}], {}, "pass"),
    (case("wer", max_increase=1.0), [{"wer_base": 0.020, "wer_head": 0.050}], {}, "FAIL"),
    (case("psnr", min_db=30), [{"psnr_min_db": math.inf, "psnr_median_db": math.inf}], {}, "pass"),
    (case("psnr", min_db=30), [{"psnr_min_db": 22.0, "psnr_median_db": 35.0}], {}, "FAIL"),
    (case("psnr", min_db=30), [{"psnr_min_db": None, "psnr_median_db": 40.0, "shape_mismatch": 1}], {}, "FAIL"),
    (case("action_diff", tol=0.05), [{"action_max_abs_diff": 0.01}], {}, "pass"),
    (case("action_diff", tol=0.05), [{"action_max_abs_diff": 0.3}], {}, "FAIL"),
    (case("audio_len", pct=10), [{"audio_s_base": 10.0, "audio_s_head": 10.5}], {}, "pass"),
    (case("audio_len", pct=10), [{"audio_s_base": 10.0, "audio_s_head": 5.0}], {}, "FAIL"),
    (case("duplex", tol=0.1), [{"take_turn_rate_base": 0.9, "take_turn_rate_head": 0.85}], {}, "pass"),
    (case("duplex", tol=0.1), [{"take_turn_rate_base": 0.9, "take_turn_rate_head": 0.5}], {}, "FAIL"),
    (case("status_match"), [{"n": 5, "diffs": []}], {}, "pass"),
    (case("status_match"), [{"n": 5, "diffs": [{"case": "x", "base": "200", "head": "500"}]}], {}, "FAIL"),
    (case("psnr", min_db=30), [{"missing": ["head"]}], {}, "incomplete"),
    (case("psnr", min_db=30), [], {}, "incomplete"),
])
def test_judge_correct(c, facts, errors, want):
    assert report.judge_correct(c, facts, errors)[0] == want


def test_text_match_accepts_a_late_fork_but_not_truncation_and_reports_base_noise():
    c = case("text_match", min_match=0.5)
    fork = [{"text_n": 16, "text_identical": 0, "text_sim_median": 0.93, "text_len_ratio_median": 0.98}]
    cut = [{"text_n": 16, "text_identical": 0, "text_sim_median": 0.62, "text_len_ratio_median": 0.5}]
    assert report.judge_correct(c, fork, {})[0] == "pass"
    assert report.judge_correct(c, cut, {})[0] == "FAIL"
    _, detail = report.judge_correct(c, [{"text_n": 8, "text_identical": 6},
                                         {"self_base": {"text_n": 8, "text_identical": 7}}], {})
    assert detail == "6/8 identical (base vs base 7/8)"


def test_correct_limit_widens_from_aa_swings_with_the_manifest_as_floor():
    """Qwen3-TTS: main vs main moved WER 3.62% to 1.09% (2.53 points) against a 2-point limit."""
    wer = {"kind": "wer", "max_increase": 2.0}
    ck, note = report.correct_limit(wer, [2.53])
    assert ck["max_increase"] == pytest.approx(3.795) and "from A/A" in note
    assert report.correct_limit(wer, [0.4]) == (wer, "")
    assert report.correct_limit({"kind": "text_match", "min_match": 0.5}, [1.0])[0]["min_match"] == 0.1
    assert report.correct_limit({"kind": "psnr", "min_db": 30}, [32.0, 40.0])[0]["min_db"] == 29.0
    assert report.correct_limit({"kind": "psnr", "min_db": 30}, [12.0])[0]["min_db"] == report.PSNR_FLOOR_DB
    c = Case(id="tts", modality="tts", driver="suite", kind="tts", check=wer)
    facts = [{"wer_base": 0.0109, "wer_head": 0.0362}]
    assert report.judge_correct(c, facts, {})[0] == "FAIL"
    v, detail = report.judge_correct(c, facts, {}, swings=[2.53])
    assert v == "pass" and "limit 3.8 points from A/A (manifest 2 points)" in detail


def test_mixed_directions_are_not_called_faster():
    better = Perf("lat_p50_s", 1, Stat.of([1.0] * 3), Stat.of([0.9] * 3), True)
    worse_noisy = Perf("req_per_s", 8, Stat.of([10, 14, 6]), Stat.of([9, 13, 5]), False)
    r = report.Row(model="m", family="F", case="c", modality="x", starts={"base": True, "head": True}, correct="pass",
                   correct_detail="", latency=report.judge_perf(better, 5, 2),
                   throughput=report.judge_perf(worse_noisy, 5, 2))
    assert r.verdict() == "mixed, within noise"
    assert report.fmt_val("lat_p50_s", 0.4123) == "412 ms" and report.fmt_val("lat_p50_s", 5.18) == "5.18 s"
    assert report.fmt_val("ttfa_ms_p50", 19.73) == "19.7 ms" and report.fmt_val("req_per_s", 4.961) == "4.96 req/s"


# Result directories as compare writes them: results/<model>/<arm>/s<k>/<case>/*-c<N>.json

def chat_model(name="m", kind="chat_text", check=None, conc=(1, 8)):
    tts = kind == "tts"
    p = {"latency": "e2e_s_p50", "throughput": "audio_s_per_s"} if tts else {"latency": "lat_p50_s",
                                                                              "throughput": "req_per_s"}
    return Model(name=name, serve=name, family="Omni", tags=[], thresholds={"perf_pct": 5, "noise_sigma": 2},
                 cases=[Case(id="c", modality="text", driver="suite" if tts else "client", kind=kind, conc=list(conc),
                             perf=p, check=check or {"kind": "text_match", "min_match": 0.5})])


def write_arm(md: Path, arm: str, reqs1, reqsn, wall_n: float, audio=False, session=0, passes=3, errors=0):
    """reqs: (e2e_s, output length) per request; the pass metrics are what the suites compute."""
    d = md / arm / f"s{session}" / "c"
    d.mkdir(parents=True, exist_ok=True)
    (md / arm / f"s{session}" / "startup.json").write_text(json.dumps({"healthy": True, "startup_s": 10}))
    for conc, reqs, wall in ((1, reqs1, sum(e for e, _ in reqs1)), (8, reqsn, wall_n)):
        rows = []
        for i, (e, ln) in enumerate(reqs):
            r = {"id": f"{i:03d}", "ok": True, "e2e_s": e}
            r.update({"audio_s": ln} if audio else {"text": "x" * int(ln), "n_chunks": int(ln), "audio_s": 0.0})
            rows.append(r)
        lat = sorted(r["e2e_s"] for r in rows)
        p50 = (lat[len(lat) // 2 - 1] + lat[len(lat) // 2]) / 2
        metrics = {"lat_p50_s": p50, "e2e_s_p50": p50, "req_per_s": len(rows) / wall,
                   "audio_s_per_s": sum(ln for _, ln in reqs) / wall, "errors": errors}
        reps = [{"index": k, "wall_s": wall, "metrics": metrics, "requests": rows} for k in range(passes)]
        (d / f"x-{arm}-c{conc}.json").write_text(json.dumps({"schema": "mstar-eval/1", "repeats": reps, "summary": {}}))


def finish(rd: Path, md: Path, facts=None, run=None):
    (md / "done.json").write_text(json.dumps({"status": "done"}))
    (md / "compare.json").write_text(json.dumps({"c": facts or [{"text_n": 8, "text_identical": 8}]}))
    (rd / "run.json").write_text(json.dumps({**RUN, "models": ["m"], **(run or {})}))


def test_rows_markdown_chart_and_summary(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.01, 96)] * 8, [(1.01, 96)] * 8, wall_n=2.02)
    finish(rd, md)
    path, rows = report.write(rd, [chat_model()], chart_url="https://example/x.png", calibration={})
    r = rows[0]
    assert r.correct == "pass" and r.latency.verdict == "same" and not r.flagged and r.no_baseline
    text = path.read_text()
    assert "No regressions" in text and "https://example/x.png" in text and "No A/A noise baseline on H200" in text
    assert "\u2014" not in text and "\u2013" not in text
    assert json.loads((rd / "report" / "summary.json").read_text())[0]["verdict"] == "same"
    pytest.importorskip("matplotlib")
    assert (rd / "report" / "chart.png").stat().st_size > 1000


def test_regression_correctness_and_a_head_that_does_not_start_are_flagged(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.2, 96)] * 8, [(1.2, 96)] * 8, wall_n=2.4)
    finish(rd, md, [{"text_n": 8, "text_identical": 1, "text_sim_median": 0.3, "text_len_ratio_median": 0.4}])
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.flagged and r.correct == "FAIL"
    assert "regression" in r.verdict() and "correctness changed" in r.verdict()
    assert report.flagged_perf_cases([r]) == ["c"]
    assert "1 of 1 rows flagged" in report.render_md([r], json.loads((rd / "run.json").read_text()))
    (md / "head" / "s0" / "startup.json").write_text(json.dumps({"healthy": False}))
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.correct == "FAIL" and r.flagged and "head does not start" in r.verdict()


def test_aa_floor_turns_a_drift_into_noise_and_calibrate_folds_each_run_once(tmp_path):
    rd, md = tmp_path / "aa", tmp_path / "aa" / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.08, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)       # +8% at c=1, tight passes
    finish(rd, md)
    assert report.build_rows(rd, [chat_model()], calibration={})[0].latency.verdict == "regression"
    cal = {"h200:m/c/lat_p50_s@c1": {"deltas": [-6.0, 2.0]}}                  # A/A drift up to 6%: floor 9%
    p = report.build_rows(rd, [chat_model()], calibration=cal)[0].latency
    assert p.floor_pct == 9.0 and p.verdict == "noisy"
    path = tmp_path / "cal.json"
    with pytest.raises(ValueError, match="not an A/A run"):
        calibrate.add_run(rd, [chat_model()], path)
    run = json.loads((rd / "run.json").read_text())
    run["head"]["sha"] = run["base"]["sha"]
    (rd / "run.json").write_text(json.dumps(run))
    assert calibrate.add_run(rd, [chat_model()], path) > 0
    assert calibrate.add_run(rd, [chat_model()], path) == 0
    assert calibrate.floor_pct(calibrate.load(path), "h200:m/c/lat_p50_s@c1") == pytest.approx(12.0)


def test_calibrate_folds_correctness_swings_into_a_run_seen_before(tmp_path):
    rd, md = tmp_path / "aa", tmp_path / "aa" / "results" / "m"
    write_arm(md, "base", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    write_arm(md, "head", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    finish(rd, md, [{"wer_base": 0.0362, "wer_head": 0.0109}], run={"head": RUN["base"], "id": "aa"})
    m = chat_model(kind="tts", check={"kind": "wer", "max_increase": 2})
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"_runs": ["aa"]}))                         # perf folded in before swings existed
    assert calibrate.add_run(rd, [m], path) == 1
    cal = calibrate.load(path)
    assert cal["h200:m/c/correct"] == {"kind": "wer", "swings": [2.53]} and "h200:m/c/e2e_s_p50@c1" not in cal
    rd2, md2 = tmp_path / "pr", tmp_path / "pr" / "results" / "m"
    write_arm(md2, "base", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    write_arm(md2, "head", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    finish(rd2, md2, [{"wer_base": 0.0109, "wer_head": 0.0362}])
    assert report.build_rows(rd2, [m], calibration={})[0].correct == "FAIL"
    assert report.build_rows(rd2, [m], calibration=cal)[0].correct == "pass"


def test_diverged_lengths_compare_matched_requests_at_c1_and_tokens_per_s_at_cmax(tmp_path):
    """Head's greedy text forks into replies twice as long on half the prompts: per pass p50 latency and req/s say
    -50%, while every request of equal length is exactly as fast."""
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 4 + [(2.0, 192)] * 4, [(1.0, 96)] * 4 + [(2.0, 192)] * 4, wall_n=3.0)
    finish(rd, md, [{"text_n": 8, "text_identical": 4, "text_sim_median": 0.7, "text_len_ratio_median": 1.5}])
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.latency.basis == "matched length 4/8" and r.latency.verdict == "same"
    assert r.throughput.metric == "tok_per_s" and r.throughput.verdict == "same"
    assert "[matched length 4/8]" in report.fmt_perf(r.latency)


def test_matched_set_is_fixed_across_sessions(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    lats = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    for s in (0, 1):
        write_arm(md, "base", [(x, 96) for x in lats], [(1.0, 96)] * 8, wall_n=2.0, session=s)
    write_arm(md, "head", [(x, 96) for x in lats], [(1.0, 96)] * 8, wall_n=2.0, session=0)
    write_arm(md, "head", [(x, 96) for x in lats[:6]] + [(1.5, 140), (1.6, 150)], [(1.0, 96)] * 8, wall_n=2.0,
              session=1)
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.latency.basis == "matched length 6/8" and r.latency.base.n == 6
    assert r.latency.base.sd == 0 and r.latency.head.sd == 0 and r.latency.verdict == "same"


def test_too_few_matched_lengths_fall_back_to_per_token_and_tts_per_audio_second(tmp_path):
    rd, md = tmp_path / "a", tmp_path / "a" / "results" / "m"
    write_arm(md, "base", [(1.0, 100)] * 8, [(1.0, 100)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.3, 130)] * 7 + [(1.0, 100)], [(1.3, 130)] * 8, wall_n=2.6)
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.latency.metric == "s_per_tok_p50" and r.latency.basis.startswith("per token")
    assert r.latency.verdict == "same" and report.fmt_val("s_per_tok_p50", 0.0104) == "10.4 ms/token"
    rd, md = tmp_path / "b", tmp_path / "b" / "results" / "m"
    write_arm(md, "base", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    write_arm(md, "head", [(0.4, 4.0)] * 8, [(0.8, 4.0)] * 8, wall_n=1.6, audio=True)   # 50% slower per audio s
    finish(rd, md, [{"wer_base": 0.02, "wer_head": 0.02}])
    r = report.build_rows(rd, [chat_model(kind="tts", check={"kind": "wer", "max_increase": 2})], calibration={})[0]
    assert r.latency.metric == "s_per_audio_s_p50" and r.latency.verdict == "regression"
    assert r.throughput.metric == "audio_s_per_s" and not r.throughput.basis


def test_matching_lengths_and_fixed_size_outputs_are_left_alone(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 97)] * 8, [(1.0, 97)] * 8, wall_n=2.0)                # within one token
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.latency.metric == "lat_p50_s" and not r.latency.basis and r.throughput.metric == "req_per_s"
    lat = Perf("lat_p50_s", 1, Stat.of([1.0]), Stat.of([2.0]), True)
    assert report.length_aware(chat_model(kind="image").cases[0], {}, {}, lat, None) == (lat, None)


def test_abba_rerun_replaces_a_one_session_flag(tmp_path):
    """A server process that settles at one of two speeds: one session flags -11%, two sessions ABBA do not."""
    rd, md, rr = tmp_path, tmp_path / "results" / "m", tmp_path / "results" / "m__abba"
    write_arm(md, "base", [(1.25, 96)] * 8, [(1.25, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.385, 96)] * 8, [(1.385, 96)] * 8, wall_n=2.0)
    finish(rd, md)
    assert report.build_rows(rd, [chat_model()], calibration={})[0].flagged
    for s, (b, h) in enumerate(((1.25, 1.385), (1.385, 1.25))):           # the speed level follows the process
        write_arm(rr, "base", [(b, 96)] * 8, [(b, 96)] * 8, wall_n=2.0, session=s)
        write_arm(rr, "head", [(h, 96)] * 8, [(h, 96)] * 8, wall_n=2.0, session=s)
    (rr / "done.json").write_text(json.dumps({"status": "done"}))
    (rr / "compare.json").write_text(json.dumps({"c": [{"text_n": 8, "text_identical": 8}] * 2}))
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert not r.flagged and r.latency.verdict in ("same", "noisy")
    assert r.rerun.startswith("flagged with 1 session (c=1 -11%") and r.latency.base.n == 6
    text = report.render_md([r], json.loads((rd / "run.json").read_text()))
    assert "rerun with two (ABBA)" in text and "(2-session rerun)" in text


# ---------------------------------------------------------------- checks

def test_saved_outputs_text_psnr_actions_audio(tmp_path):
    from PIL import Image

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "000.txt").write_text("same"), (b / "000.txt").write_text("same")
    (a / "001.txt").write_text("x"), (b / "001.txt").write_text("y")
    img = np.zeros((8, 8, 3), np.uint8)
    Image.fromarray(img).save(a / "002.png")
    Image.fromarray(img).save(b / "002.png")
    img2 = img.copy()
    img2[0, 0] = 255
    Image.fromarray(img).save(a / "003.png")
    Image.fromarray(img2).save(b / "003.png")
    np.array([0.0, 1.0], np.float32).tofile(a / "004.f32")
    np.array([0.0, 1.25], np.float32).tofile(b / "004.f32")
    np.zeros(24000, np.int16).tofile(a / "005.pcm")
    np.zeros(12000, np.int16).tofile(b / "005.pcm")
    f = checks.saved(a, b)
    assert f["text_identical"] == 1 and f["text_n"] == 2 and 0 <= f["text_sim_median"] <= 1
    assert f["psnr_median_db"] > 30 and f["psnr_min_db"] < float("inf")
    assert f["action_max_abs_diff"] == 0.25 and f["audio_s_base"] == 1.0 and f["audio_s_head"] == 0.5
    Image.fromarray(np.zeros((4, 8, 3), np.uint8)).save(b / "002.png")
    assert checks.saved(a, b)["shape_mismatch"] == 1
    assert checks.saved(a, tmp_path / "nope")["missing"] == ["head"]


def test_text_facts_fork_and_truncation():
    fox = "the quick brown fox jumps over the lazy dog"
    late = checks.text_facts([(fox, fox.replace("the lazy", "a lazy"))])
    assert late["text_identical"] == 0 and late["text_sim_median"] > 0.9 and late["text_prefix_median"] > 0.7
    cut = checks.text_facts([(fox, "the quick")])
    assert cut["text_len_ratio_min"] < 0.3


def test_rp_and_suite_facts(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "rp.json").write_text(json.dumps({"x": {"status": 200}, "y": {"status_line": "HTTP/1.1 400 Bad"},
                                           "z": {"status": 200, "identical": True}, "_elapsed_s": 1}))
    (b / "rp.json").write_text(json.dumps({"x": {"status": 200}, "y": {"status_line": "HTTP/1.1 200 OK"},
                                           "z": {"status": 200, "identical": False}}))
    f = checks.rp_facts(a, b)
    assert f["n"] == 3 and {d["case"] for d in f["diffs"]} == {"y", "z"}

    def write(d, wer, hyps):
        reqs = [{"id": f"i{k}", "ok": True, "hyp": h} for k, h in enumerate(hyps)]
        (d / "x-c1.json").write_text(json.dumps({"schema": "mstar-eval/1", "summary": {"wer": {"mean": wer}},
                                                 "repeats": [{"requests": reqs}]}))
    write(a, 0.02, ["a", "b", "c"])
    write(b, 0.03, ["a", "b", "d"])
    f = checks.suite_facts(a, b, 1)
    assert f["wer_base"] == 0.02 and f["wer_head"] == 0.03 and f["text_identical"] == 2 and f["text_n"] == 3


# ---------------------------------------------------------------- the e2e suite and the CLI

FAKE_SERVE = r'''
import argparse, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DELAY, SUFFIX = {delay}, {suffix!r}
p = argparse.ArgumentParser()
p.add_argument("cmd"); p.add_argument("model"); p.add_argument("--port", type=int)
a, _ = p.parse_known_args()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        words = (req["messages"][0]["content"] + SUFFIX).split()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close"); self.end_headers()
        for w in words:
            time.sleep(DELAY)
            msg = {{"choices": [{{"delta": {{"content": w + " "}}}}]}}
            self.wfile.write(b"data: " + json.dumps(msg).encode() + b"\n\n"); self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        self.close_connection = True


ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
'''


def fake_tree(d: Path, delay: float = 0.0, suffix: str = "") -> Path:
    """A checkout whose `python -m mstar.cli.main serve` is a stdlib chat server streaming the prompt back."""
    (d / "mstar" / "cli").mkdir(parents=True)
    (d / "mstar" / "__init__.py").write_text("")
    (d / "mstar" / "cli" / "__init__.py").write_text("")
    (d / "mstar" / "cli" / "main.py").write_text(FAKE_SERVE.format(delay=delay, suffix=suffix))
    (d / ".complete").touch()
    return d


def test_e2e_suite_runs_without_data_and_saves_outputs(tmp_path):
    from benchmark.eval.cli import main
    from benchmark.eval.schema import RunResult

    tree = fake_tree(tmp_path / "tree")
    out = tmp_path / "out"
    main(["run", "e2e", "--kind", "chat_text", "--serve-cmd",
          f"{sys.executable} {tree}/mstar/cli/main.py serve fake --port {{port}}", "--limit", "4",
          "--concurrency", "1", "--repeats", "2", "--warmup", "1", "--label", "fake", "--out", str(out)])
    res = RunResult.load(out / "e2e-fake-c1.json")
    assert res.config["data"]["path"] is None and res.config["suite"]["kind"] == "chat_text"
    assert res.summary["errors"]["mean"] == 0 and res.summary["lat_p50_s"]["n"] == 2
    row = res.repeats[0].requests[0]
    assert row["ok"] and row["n_chunks"] == len(row["text"].split()) and "_audio" not in row
    saved = checks.saved_dir(out, 1)
    assert sorted(p.name for p in saved.iterdir()) == ["000.txt", "001.txt", "002.txt", "003.txt"]
    with pytest.raises(SystemExit, match="needs --data"):
        main(["run", "tts", "--url", "http://127.0.0.1:1"])
    with pytest.raises(SystemExit, match="needs --media"):
        main(["run", "e2e", "--kind", "chat_image", "--url", "http://127.0.0.1:1"])


# ---------------------------------------------------------------- slots and the queue

def test_parse_hosts_and_provider_hook(monkeypatch):
    assert hosts.parse_hosts("u@a, u@b:1,u@b:0") == [("u@a", 0), ("u@b", 1), ("u@b", 0)]
    with pytest.raises(ValueError, match="same GPU twice"):
        hosts.parse_hosts("a,a:0")
    mod = type(sys)("fake_launcher")
    released = []

    class Launcher(hosts.HostProvider):
        def hosts(self):
            return "u@vm1,u@vm2:1"

        def release(self):
            released.append(True)

    mod.make = lambda args: Launcher()
    monkeypatch.setitem(sys.modules, "fake_launcher", mod)
    args = Namespace(hosts=None, host_provider="fake_launcher:make")
    slots, provider = compare.make_slots(args)
    assert [s.name for s in slots] == ["u@vm1:0", "u@vm2:1"]
    assert all(isinstance(s.transport, hosts.SSHTransport) for s in slots)
    provider.release()
    assert released == [True]


def slot(name: str) -> hosts.Slot:
    return hosts.Slot(Namespace(name=name), 0)


def test_queue_longest_first_reruns_first_and_idle_slots_take_queued_work():
    q = hosts.WorkQueue(["small", "big", "mid"], {"small": 1, "big": 30, "mid": 10})
    assert q.items == ["big", "mid", "small"]
    done: dict[str, list[str]] = {"x:0": [], "y:0": []}
    release = threading.Event()

    def run_one(s, name):
        if name == "big":
            release.wait(10)        # whichever slot took "big" holds it until the other has drained the queue
        done[s.name].append(name)
        if name == "small":
            release.set()
        return "mid__abba" if name == "mid" else None

    hosts.run_slots([slot("x"), slot("y")], q, run_one)
    assert sorted(done.values()) == [["big"], ["mid", "mid__abba", "small"]]     # the rerun jumps the queue


def test_a_lost_host_requeues_its_model_and_gives_up_when_every_slot_is_gone():
    q = hosts.WorkQueue(["a", "b"], {"a": 2, "b": 1})
    dead, alive = slot("dead"), slot("alive")
    ran, lost = [], []
    took = threading.Event()

    def run_one(s, name):
        if s is dead:
            took.set()
            raise hosts.HostLostError("ssh: connection refused")
        took.wait(10)               # the dead slot has taken one model before the live one finishes any
        ran.append(name)

    hosts.run_slots([dead, alive], q, run_one, on_lost=lambda s, n, e: lost.append(n))
    assert sorted(ran) == ["a", "b"] and len(lost) == 1
    q2, gave_up = hosts.WorkQueue(["a", "b"], {}), []

    def always_lost(s, name):
        raise hosts.HostLostError("gone")

    hosts.run_slots([slot("only")], q2, always_lost, on_given_up=gave_up.append)
    assert sorted(gave_up) == ["a", "b"]


# ---------------------------------------------------------------- one model, end to end, through the SSH path

class FakeRemote:
    """An "SSH host" on this machine: commands run in its own home, files are copied there, never shared."""

    def __init__(self, home: Path):
        self.name, self.home = "bench@fake-host", home

    def sh(self, script, timeout=None):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout,
                              cwd=self.home, check=False)

    def push(self, local, remote):
        hosts._rsync([f"{local}/" if local.is_dir() else str(local), remote])

    def pull(self, remote, local, exclude=()):
        hosts._rsync([*[f"--exclude={e}" for e in exclude], f"{remote}/", f"{local}/"])


def test_one_model_runs_both_arms_on_a_remote_slot_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "POLL_S", 0.2)
    home = tmp_path / "home"
    home.mkdir()
    man = mf.parse({"defaults": {"conc": [1, 2], "repeats": 2, "warmup": 1, "startup_timeout_s": 60}, "models": {
        "fake": {"serve": "fake", "family": "LLM", "cases": [
            {"id": "text", "client": "chat_text", "n": 4, "check": {"kind": "text_match", "min_match": 0.5},
             "perf": {"latency": "lat_p50_s", "throughput": "req_per_s"}}]}}})
    models = man.select("all")
    base, head = trees.Ref("main", "a" * 40, "main"), trees.Ref("x", "b" * 40, "PR #1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({**RUN, "base": base.__dict__, "head": head.__dict__,
                                                  "models": ["fake"]}))
    (tmp_path / "data" / "media").mkdir(parents=True)
    args = Namespace(python=sys.executable, host_python=sys.executable, host_dir="mstar-eval",
                     data_dir=str(tmp_path / "data"), sessions=1, repeats=None, no_rerun=True,
                     calibration=str(tmp_path / "cal.json"))
    slot = hosts.Slot(FakeRemote(home), None)
    runner = compare.Runner(args, man, models, base, head, run_dir, [slot])
    runner.cache = tmp_path / "cache"
    fake_tree(runner.cache / "src" / base.sha)
    fake_tree(runner.cache / "src" / head.sha, suffix=" indeed")
    assert runner.run_one(slot, "fake") is None
    host_dir = home / "mstar-eval"
    assert slot.host_dir == str(host_dir) and (host_dir / "evalfw" / runner.evalfw / "benchmark" / "eval").is_dir()
    res = run_dir / "results" / "fake"
    assert json.loads((res / "done.json").read_text())["status"] == "done"
    assert not list(res.rglob("audio"))                       # saved outputs stay on the host
    assert list((host_dir / "runs" / "run" / "fake").rglob("audio/*-c1/r1/000.txt"))
    for arm in ("base", "head"):
        info = json.loads((res / arm / "s0" / "startup.json").read_text())
        assert info["healthy"] and info["mstar_from_arm_src"] and info["cases"]["text"]["rc"] == 0
    facts = json.loads((res / "compare.json").read_text())["text"][0]
    assert facts["text_n"] == 4 and facts["text_identical"] == 0 and facts["text_sim_median"] > 0.6
    job = json.loads((run_dir / "jobs" / "fake.json").read_text())
    assert job["order"] == ["base", "head"] and job["arms"]["head"]["src"] == f"{host_dir}/src/{head.sha}"
    row = report.build_rows(run_dir, models, calibration={})[0]
    assert row.starts == {"base": True, "head": True} and row.correct == "pass"
    assert row.latency.base.n == 2 and row.latency.head.n == 2
    assert runner.state["models"]["fake"]["status"] == "done"
    assert runner.state["models"]["fake"]["elapsed_s"] > 1          # polled: the launch did not wait for the job


def test_stopping_compare_stops_the_job_and_its_servers(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "POLL_S", 0.2)
    man = mf.parse({"defaults": {"conc": [1], "repeats": 50, "warmup": 0, "startup_timeout_s": 60}, "models": {
        "fake": {"serve": "fake", "family": "LLM", "cases": [
            {"id": "text", "client": "chat_text", "n": 4, "check": {"kind": "text_match", "min_match": 0.5},
             "perf": {"latency": "lat_p50_s"}}]}}})
    base, head = trees.Ref("main", "a" * 40, "main"), trees.Ref("x", "b" * 40, "PR #1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({**RUN, "models": ["fake"]}))
    (tmp_path / "data").mkdir()
    args = Namespace(python=sys.executable, host_python=sys.executable, host_dir="mstar-eval",
                     data_dir=str(tmp_path / "data"), sessions=1, repeats=None, no_rerun=True,
                     calibration=str(tmp_path / "cal.json"))
    slot = hosts.Slot(hosts.LocalTransport(), None)
    runner = compare.Runner(args, man, man.select("all"), base, head, run_dir, [slot])
    runner.cache = tmp_path / "cache"
    for ref in (base, head):
        fake_tree(runner.cache / "src" / ref.sha, delay=0.05)
    out = run_dir / "results" / "fake"
    th = threading.Thread(target=runner.run_one, args=(slot, "fake"))
    th.start()
    for _ in range(300):
        if '"phase": "case"' in ((out / "status.json").read_text() if (out / "status.json").exists() else ""):
            break
        threading.Event().wait(0.1)
    pid = int((out / "pid").read_text())
    assert procs_under(runner.cache / "src")              # the base server is up
    runner.stop.set()
    th.join(60)
    assert not th.is_alive() and runner.state["models"]["fake"]["status"] == "interrupted"
    assert json.loads((out / "done.json").read_text())["status"] == "interrupted"
    assert not Path(f"/proc/{pid}").exists()
    assert not procs_under(runner.cache / "src")          # servers run with their checkout as working directory


def procs_under(root: Path) -> list[int]:
    found = []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            if Path(os.readlink(d / "cwd")).is_relative_to(root.resolve()):
                found.append(int(d.name))
        except OSError:
            continue
    return found


def test_a_flagged_model_queues_an_abba_rerun_of_its_flagged_cases(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.2, 96)] * 8, [(1.2, 96)] * 8, wall_n=2.4)
    finish(rd, md)
    man = mf.Manifest(models={"m": chat_model()}, datasets={}, defaults={})
    args = Namespace(sessions=1, no_rerun=False, repeats=None, calibration=str(tmp_path / "none.json"),
                     data_dir=str(tmp_path))
    runner = compare.Runner(args, man, [chat_model()], trees.Ref("a", "a" * 40, "a"), trees.Ref("b", "b" * 40, "b"),
                            rd, [])
    assert runner.maybe_rerun("m") == "m__abba" and runner.rerun_cases["m__abba"] == ["c"]
    assert runner.maybe_rerun("m__abba") is None                   # never a rerun of a rerun
    assert [c.id for c in runner.model_for("m__abba").cases] == ["c"]
    args.no_rerun = True
    assert runner.maybe_rerun("m") is None


def test_trees_resolve_and_export(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    (repo / "rust").mkdir()
    (repo / "rust" / "Cargo.toml").write_text("[package]\n")
    (repo / "a.txt").write_text("one")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "one"], check=True)
    ref = trees.resolve(repo, "main", remote="origin")
    assert ref.label == "main" and len(ref.sha) == 40 and ref.rust_tree
    by_sha = trees.resolve(repo, ref.sha[:10], remote="origin")
    assert by_sha.sha == ref.sha and by_sha.label == ref.sha[:9]
    with pytest.raises(trees.TreeError, match="cannot resolve"):
        trees.resolve(repo, "no-such-branch", remote="origin")
    out = trees.export(repo, ref.sha, tmp_path / "src" / ref.sha)
    assert (out / "a.txt").read_text() == "one" and (out / ".complete").exists()
    assert compare.gpu_type("NVIDIA H200") == "h200" and compare.gpu_type("NVIDIA B200") == "b200"
