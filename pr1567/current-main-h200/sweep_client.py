"""Closed-loop Chatterbox audio throughput sweep for #448 / #449.

For one server and one (mode, concurrency) cell: keep `c` streaming
`/v1/audio/speech` requests in flight until `n` requests have finished
(after `warmup` unmeasured ones). Request i uses sentence i % len(SENTENCES)
and seed SEED_BASE + i, so main and the PR get the same work. In `mixed`
mode even i send cfg_weight=0 and odd i the default; in `default` mode all
send the default.

Metrics per cell:
- audio_s_per_s: seconds of audio returned / wall time from the first
  measured submit to the last measured finish;
- ttfa p50/p95: submit to first PCM byte;
- stall after first chunk p50/p95: if a client plays from the first byte,
  how long playback waits right after the first chunk runs out (0 when the
  next chunk arrived in time). Arrivals within 5 ms are one chunk;
- total stall p50/p95: the same, summed over the whole stream;
- e2e p50: submit to last byte.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp

SENTENCES = [
    "The quiet harbor town woke slowly, with fishermen mending nets along the pier while gulls circled overhead.",
    "Please confirm your appointment for Thursday at three in the afternoon.",
    "Scientists have discovered a new species of deep sea fish that glows in the dark and feeds on tiny shrimp.",
    "I can't believe we actually made it to the top of the mountain before sunrise.",
    "The library will be closed on Monday for maintenance, and will reopen on Tuesday morning at nine.",
    "Turn left at the next intersection, then continue straight for about two miles.",
    "Every great journey begins with a single step, but it is the thousand steps after that which matter most.",
    "Our quarterly results exceeded expectations, driven by strong demand in the European and Asian markets.",
    "She opened the old wooden box and found a stack of letters tied together with a faded blue ribbon.",
    "Thank you for calling. All of our representatives are currently busy, please stay on the line.",
    "The recipe calls for two cups of flour, a pinch of salt, three eggs, and a generous spoonful of honey.",
    "After the storm passed, the children ran outside to splash in the puddles and look for rainbows.",
    "The train to the airport departs every fifteen minutes from platform four.",
    "Learning a new language takes patience, practice, and the courage to make mistakes in front of others.",
    "He tuned his guitar, took a deep breath, and began to play the song he had written for her.",
    "Remember to back up your files regularly so that nothing important is lost if your computer fails.",
]
SEED_BASE = 4480
SR = 24000
MERGE_S = 0.005


def stalls(chunks):
    """(stall right after the first chunk, total stall), played from the first byte."""
    merged = []
    for t, cum in chunks:
        if merged and t - merged[-1][0] <= MERGE_S:
            merged[-1] = (merged[-1][0], cum)
        else:
            merged.append((t, cum))
    if not merged:
        return None, None
    play_end = merged[0][0] + merged[0][1] / 2 / SR
    prev = merged[0][1]
    first = 0.0 if len(merged) > 1 else None
    total = 0.0
    for k, (t, cum) in enumerate(merged[1:]):
        gap = max(0.0, t - play_end)
        if k == 0:
            first = gap
        total += gap
        play_end = max(play_end, t) + (cum - prev) / 2 / SR
        prev = cum
    return first, total


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


async def one(session, url, i, mode, rows, measured):
    cfg = 0.0 if (mode == "mixed" and i % 2 == 0) else None
    body = {
        "model": "chatterbox", "input": SENTENCES[i % len(SENTENCES)], "voice": "default",
        "stream": True, "response_format": "pcm", "seed": SEED_BASE + i,
    }
    if cfg is not None:
        body["cfg_weight"] = cfg
    nbytes = 0
    chunks = []
    start = time.perf_counter()
    async with session.post(url, json=body) as resp:
        status = resp.status
        async for chunk in resp.content.iter_any():
            if chunk:
                nbytes += len(chunk)
                chunks.append((time.perf_counter() - start, nbytes))
    end = time.perf_counter()
    first, total = stalls(chunks)
    if measured:
        rows.append({
            "i": i, "cfg_weight": cfg, "status": status, "start": start, "end": end,
            "ttfa_s": chunks[0][0] if chunks else None, "e2e_s": end - start,
            "audio_s": nbytes / 2 / SR, "n_chunks": len(chunks),
            "stall_first_s": first, "stall_total_s": total,
        })


async def run_cell(url, mode, c, n, warmup):
    rows: list = []
    timeout = aiohttp.ClientTimeout(total=900)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        # warmup at the same concurrency, not measured
        await asyncio.gather(*(one(session, url, 10_000 + k, mode, rows, False) for k in range(warmup)))
        queue = asyncio.Queue()
        for i in range(n):
            queue.put_nowait(i)

        async def lane():
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await one(session, url, i, mode, rows, True)

        await asyncio.gather(*(lane() for _ in range(c)))
    ok = [r for r in rows if r["status"] == 200 and r["audio_s"] > 0]
    wall = max(r["end"] for r in rows) - min(r["start"] for r in rows)
    audio = sum(r["audio_s"] for r in ok)

    def stat(key, rs):
        return {"p50": pct([r[key] for r in rs], 0.5), "p95": pct([r[key] for r in rs], 0.95)}

    summary = {
        "mode": mode, "c": c, "n": len(rows), "ok": len(ok), "wall_s": wall, "audio_total_s": audio,
        "audio_s_per_s": audio / wall, "req_per_s": len(ok) / wall,
        "ttfa_s": stat("ttfa_s", ok), "stall_first_s": stat("stall_first_s", ok),
        "stall_total_s": stat("stall_total_s", ok), "e2e_s": stat("e2e_s", ok),
        "mean_audio_s": statistics.mean(r["audio_s"] for r in ok) if ok else None,
    }
    if mode == "mixed":
        for tag, sel in (("cfg0", lambda r: r["cfg_weight"] == 0.0), ("default", lambda r: r["cfg_weight"] is None)):
            rs = [r for r in ok if sel(r)]
            summary[tag] = {"n": len(rs), "ttfa_s": stat("ttfa_s", rs), "stall_first_s": stat("stall_first_s", rs),
                            "stall_total_s": stat("stall_total_s", rs), "e2e_s": stat("e2e_s", rs)}
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/v1/audio/speech")
    ap.add_argument("--modes", default="default,mixed")
    ap.add_argument("--conc", default="4,8,16,32")
    ap.add_argument("--n-per-lane", type=int, default=6)
    ap.add_argument("--min-n", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in [int(x) for x in args.conc.split(",")]:
        for mode in args.modes.split(","):
            n = max(args.min_n, args.n_per_lane * c)
            summary, rows = asyncio.run(run_cell(args.url, mode, c, n, warmup=min(c, 8)))
            json.dump({"summary": summary, "rows": rows}, open(out / f"{mode}_c{c}.json", "w"), indent=1)
            print(json.dumps({k: summary[k] for k in ("mode", "c", "n", "ok", "audio_s_per_s", "ttfa_s",
                                                      "stall_first_s", "stall_total_s", "e2e_s")}), flush=True)


if __name__ == "__main__":
    main()
