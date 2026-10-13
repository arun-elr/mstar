"""Streaming rollout client for the e2e Cosmos3 runs, adapted from
benchmark/cosmos3/bench_stream_video.py (same /generate request, same prompt),
plus concurrency, every chunk's arrival time and frame count, and the mp4
chunks saved per stream.

Each round fires --concurrency identical-shape requests at once (stream i uses
seed + i) and waits for all. Writes one JSON with every request's arrivals.
"""
import argparse
import base64
import io
import json
import statistics
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--mode", choices=["kv", "chained"], default="kv")
ap.add_argument("--size", default="832x480")
ap.add_argument("--frames", type=int, default=241)
ap.add_argument("--window-frames", type=int, default=29)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--gs", type=float, default=6.0)
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--image", default="")
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--save-dir", default="", help="mp4 chunks of the timed rounds: <dir>/r<round>_s<stream>_<i>.mp4")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
args = ap.parse_args()

PROMPT = "A robot arm is cleaning a plate in the kitchen, smooth natural motion."


def count_frames(mp4: bytes):
    import av

    with av.open(io.BytesIO(mp4)) as c:
        s = c.streams.video[0]
        return int(s.frames) if s.frames else sum(1 for _ in c.decode(s))


def multipart(fields, files):
    b = uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for name, fn, data in files:
        body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{fn}\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{b}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={b}"


def one(seed, t_start, save_prefix, res):
    mk = {"size": args.size, "num_frames": args.frames, "num_inference_steps": args.steps,
          "guidance_scale": args.gs, "fps": args.fps, "seed": seed,
          "window_mode": args.mode, "window_frames": args.window_frames, "stream_video": True}
    fields = {"text": PROMPT, "output_modalities": "video", "streaming": "true", "model_kwargs": json.dumps(mk)}
    files = []
    if args.image:
        files.append(("files", Path(args.image).name, Path(args.image).read_bytes()))
        fields["input_modalities"] = "image,text"
    body, ctype = multipart(fields, files)
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/generate", data=body,
                                 headers={"Content-Type": ctype})
    rec = {"seed": seed, "arrivals": [], "frames": [], "bytes": [], "error": None}
    try:
        t0 = time.perf_counter()
        rec["send_offset_s"] = t0 - t_start
        rec["send_wall"] = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("modality") == "error":
                    raise RuntimeError(base64.b64decode(msg["data"]).decode(errors="replace"))
                if msg.get("modality") != "video":
                    continue
                data = base64.b64decode(msg["data"])
                rec["arrivals"].append(time.perf_counter() - t0)
                rec["bytes"].append(len(data))
                rec["frames"].append(count_frames(data))
                if save_prefix:
                    Path(f"{save_prefix}_{len(rec['bytes']) - 1:02d}.mp4").write_bytes(data)
        if not rec["arrivals"]:
            raise RuntimeError("no video chunk received")
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
    res.append(rec)


def run_round(r, timed):
    res, ths = [], []
    t_start = time.perf_counter()
    wall0 = time.time()
    for i in range(args.concurrency):
        sp = f"{args.save_dir}/r{r}_s{i}" if (timed and args.save_dir) else ""
        th = threading.Thread(target=one, args=(args.seed + i, t_start, sp, res))
        th.start()
        ths.append(th)
    for th in ths:
        th.join()
    wall = time.perf_counter() - t_start
    res.sort(key=lambda x: x["seed"])
    return {"round": r, "timed": timed, "wall_s": wall, "wall_start": wall0, "wall_end": time.time(), "requests": res}


if args.save_dir:
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
print(f"=== {args.tag} mode={args.mode} size={args.size} frames={args.frames} window={args.window_frames} "
      f"steps={args.steps} gs={args.gs} seed={args.seed} c={args.concurrency} "
      f"{'i2v ' + args.image if args.image else 't2v'} ===", flush=True)
rounds = []
for w in range(args.warmup):
    rounds.append(run_round(-1 - w, False))
for r in range(args.rounds):
    rd = run_round(r, True)
    rounds.append(rd)
    for q in rd["requests"]:
        if q["error"]:
            print(f"  round {r} seed {q['seed']}: ERROR {q['error'][:300]}", flush=True)
            continue
        a = q["arrivals"]
        gaps = [y - x for x, y in zip(a, a[1:])]
        print(f"  round {r} seed {q['seed']}: chunks={len(a)} frames={sum(q['frames'])} TTFF={a[0]:.2f}s "
              f"gap_med={statistics.median(gaps) if gaps else float('nan'):.2f}s total={a[-1]:.2f}s "
              f"fps={sum(q['frames']) / a[-1]:.2f}", flush=True)
    print(f"  round {r} wall={rd['wall_s']:.2f}s", flush=True)
Path(args.out).write_text(json.dumps({"args": vars(args), "prompt": PROMPT, "rounds": rounds}, indent=1))
print("DONE", flush=True)
