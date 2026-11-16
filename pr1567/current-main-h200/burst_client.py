"""Reproducible Chatterbox mixed-CFG burst for #486 / #1567.

One long CFG=0 request begins first. As soon as its first PCM chunk arrives,
eight short streaming requests are submitted together: even IDs with CFG=0,
odd IDs with the model default. Every run uses the same prompts and seeds.
Requires aiohttp. Intended for a fresh server per run and matched branch arms.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


LONG_TEXT = (
    "The old observatory stood above the harbor where the road curved between "
    "the pines and the sea. Each morning the caretaker opened the shutters, "
    "checked the instruments, and recorded the weather in a worn notebook. "
    "On this particular morning the sky was clear, but the tide was higher "
    "than anyone expected. The fishing boats returned early, and their crews "
    "gathered near the pier to compare what they had seen beyond the headland. "
    "A visitor arrived carrying a map that showed a small island, a narrow "
    "channel, and a lighthouse that had not appeared on older charts. The "
    "caretaker listened carefully, then followed the visitor up the path to "
    "the observatory. Together they watched the horizon until the afternoon "
    "light turned silver and the first stars appeared."
)
SHORT_TEXTS = [
    "Please confirm your appointment for Thursday afternoon.",
    "The train leaves from platform four in ten minutes.",
    "Turn left at the next street and stop by the library.",
    "The weather will be clear tomorrow morning.",
    "We will send the revised report before noon.",
    "The children found a small shell near the beach.",
    "Please remember to bring your keys and your phone.",
    "The meeting starts at nine in the main conference room.",
]
SR = 24000


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


async def stream(
    session: aiohttp.ClientSession,
    url: str,
    *,
    label: str,
    text: str,
    seed: int,
    cfg_weight: float | None,
    first_audio: asyncio.Event | None = None,
) -> dict:
    body = {
        "model": "chatterbox",
        "input": text,
        "voice": "default",
        "stream": True,
        "response_format": "pcm",
        "seed": seed,
    }
    if cfg_weight is not None:
        body["cfg_weight"] = cfg_weight
    start = time.perf_counter()
    nbytes = 0
    first_byte = None
    status = None
    error = None
    try:
        async with session.post(url, json=body) as response:
            status = response.status
            async for chunk in response.content.iter_any():
                if chunk:
                    nbytes += len(chunk)
                    if first_byte is None:
                        first_byte = time.perf_counter()
                        if first_audio is not None:
                            first_audio.set()
    except Exception as exc:  # record the failed request alongside the others
        error = f"{type(exc).__name__}: {exc}"
        if first_audio is not None:
            first_audio.set()
    end = time.perf_counter()
    return {
        "label": label,
        "cfg_weight": cfg_weight,
        "seed": seed,
        "prompt": text,
        "status": status,
        "error": error,
        "submit_at_s": start,
        "first_byte_at_s": first_byte,
        "finish_at_s": end,
        "ttfa_s": first_byte - start if first_byte is not None else None,
        "e2e_s": end - start,
        "audio_s": nbytes / (2 * SR),
        "audio_bytes": nbytes,
    }


async def run(url: str, delay_ms: int, timeout_s: int) -> dict:
    event = asyncio.Event()
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        long = asyncio.create_task(
            stream(
                session,
                url,
                label="long",
                text=LONG_TEXT,
                seed=48600,
                cfg_weight=0.0,
                first_audio=event,
            )
        )
        await asyncio.wait_for(event.wait(), timeout=timeout_s)
        await asyncio.sleep(delay_ms / 1000)
        long_active_at_burst = not long.done()
        shorts = await asyncio.gather(
            *(
                stream(
                    session,
                    url,
                    label=f"short-{i}",
                    text=text,
                    seed=48601 + i,
                    cfg_weight=0.0 if i % 2 == 0 else None,
                )
                for i, text in enumerate(SHORT_TEXTS)
            )
        )
        long_row = await long
    rows = [long_row, *shorts]
    start = min(r["submit_at_s"] for r in rows)
    for row in rows:
        for key in ("submit_at_s", "first_byte_at_s", "finish_at_s"):
            if row[key] is not None:
                row[key] -= start
    good = [r for r in rows if r["status"] == 200 and r["audio_bytes"] > 0]
    short_good = [r for r in good if r["label"].startswith("short-")]
    wall_s = max(r["finish_at_s"] for r in rows)
    summary = {
        "ok": len(good),
        "total": len(rows),
        "wall_s": wall_s,
        "audio_s_per_s": sum(r["audio_s"] for r in good) / wall_s,
        "short_ttfa_p50_s": quantile([r["ttfa_s"] for r in short_good], 0.5),
        "short_ttfa_p95_s": quantile([r["ttfa_s"] for r in short_good], 0.95),
        "short_ttfa_mean_s": statistics.mean(r["ttfa_s"] for r in short_good) if short_good else None,
        "short_e2e_p50_s": quantile([r["e2e_s"] for r in short_good], 0.5),
        "short_e2e_p95_s": quantile([r["e2e_s"] for r in short_good], 0.95),
        "long_e2e_s": long_row["e2e_s"],
        "long_audio_s": long_row["audio_s"],
        "long_active_at_burst": long_active_at_burst,
        "long_finish_after_short_submit_s": long_row["finish_at_s"] - shorts[0]["submit_at_s"],
        "short_submit_after_long_first_byte_s": shorts[0]["submit_at_s"] - long_row["first_byte_at_s"]
        if long_row["first_byte_at_s"] is not None else None,
    }
    return {"protocol": "pcm_s16le_24000", "delay_ms": delay_ms, "summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8486/v1/audio/speech")
    parser.add_argument("--delay-ms", type=int, default=50)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run(args.url, args.delay_ms, args.timeout_s))
    report["arm"] = args.arm
    report["start"] = args.start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    if report["summary"]["ok"] != 9:
        raise SystemExit("expected 9 successful PCM streams")
    if not report["summary"]["long_active_at_burst"]:
        raise SystemExit("long request finished before the short burst")


if __name__ == "__main__":
    main()
