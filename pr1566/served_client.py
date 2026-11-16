#!/usr/bin/env python3
"""Bounded fixed-work CosyVoice3 served check for #530/#1566."""
import argparse, asyncio, base64, hashlib, json, statistics, time
from pathlib import Path
import aiohttp
import numpy as np

PROMPT_TEXT="希望你以后能够做的比我还好呦。"
TEXTS=(
  "The quick brown fox jumps over the lazy dog near the riverbank.",
  "Please remember to bring your umbrella, because rain is expected this afternoon.",
  "Scientists have discovered a new species of frog in the mountains of Peru.",
  "She opened the old wooden box and found a stack of letters tied with ribbon.",
)
RATE=24000

def payload(text, seed, ref):
    return {
      "model":"cosyvoice3","input":text,"seed":seed,
      "ref_audio":ref,"ref_text":PROMPT_TEXT,
      "response_format":"pcm","stream":True,
      "max_new_tokens":32,"ignore_eos":True,
    }

async def speak(client, port, p, keep_audio=None):
    t0=time.perf_counter(); first=None; parts=[]
    async with client.post(f"http://127.0.0.1:{port}/v1/audio/speech",json=p) as r:
      if r.status!=200:
        raise RuntimeError(f"HTTP {r.status}: {(await r.read())[:500]!r}")
      async for part in r.content.iter_chunked(65536):
        if part:
          if first is None: first=time.perf_counter()-t0
          parts.append(part)
    total=time.perf_counter()-t0
    raw=b"".join(parts)
    if len(raw)%2: raise AssertionError("odd PCM byte count")
    samples=np.frombuffer(raw,dtype="<i2")
    if keep_audio is not None: Path(keep_audio).write_bytes(raw)
    return {
      "pcm_sha256":hashlib.sha256(raw).hexdigest(),
      "pcm_bytes":len(raw),"pcm_samples":len(samples),
      "audio_s":len(samples)/RATE,"ttfa_s":first,"total_s":total,
      "rms":float(np.sqrt(np.mean((samples.astype(np.float32)/32768)**2))) if len(samples) else 0,
    }

async def run(a):
    Path(a.out).mkdir(parents=True,exist_ok=True)
    ref="data:audio/wav;base64,"+base64.b64encode(Path(a.ref).read_bytes()).decode()
    result={"arm":a.arm,"ref_wav_sha256":hashlib.sha256(Path(a.ref).read_bytes()).hexdigest(),
            "mode":"32 generated tokens requested; ignore_eos=true; fixed ref audio, text and seed",
            "device":"server H200","tests":{}}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600),connector=aiohttp.TCPConnector(limit=16)) as client:
      await speak(client,a.port,payload("Hello there.",999,ref))
      parity=[]
      for i in range(3):
        q=payload(TEXTS[i],10+i,ref)
        x=await speak(client,a.port,q,Path(a.out)/f"parity-{i}.pcm")
        parity.append({"text":TEXTS[i],"seed":10+i,**x})
        print("parity",i,x["pcm_sha256"],round(x["total_s"],3),flush=True)
      result["tests"]["parity"]=parity
      for c,n in ((1,160),(4,256)):
        # Warm the exact concurrency without counting those requests in throughput.
        for j in range(0,2*c,c):
          await asyncio.gather(*(speak(client,a.port,payload(TEXTS[(j+i)%len(TEXTS)],5000+j+i,ref)) for i in range(c)))
        work=[(i%len(TEXTS),TEXTS[i%len(TEXTS)],1000+i) for i in range(n)]
        t0=time.perf_counter(); cpu0=time.process_time(); rows=[]
        for j in range(0,len(work),c):
          batch=work[j:j+c]
          got=await asyncio.gather(*(speak(client,a.port,payload(t,seed,ref)) for _,t,seed in batch))
          rows.extend({"text_id":text_id,"seed":seed,**x} for (text_id,_,seed),x in zip(batch,got,strict=True))
        wall=time.perf_counter()-t0
        ordered=sorted(x["total_s"] for x in rows)
        result["tests"][f"c{c}"]={"requests":len(rows),"warmup_requests":2*c,"wall_s":wall,"client_cpu_s":time.process_time()-cpu0,
          "requests_per_s":len(rows)/wall,"audio_s_total":sum(x["audio_s"] for x in rows),
          "audio_s_per_s":sum(x["audio_s"] for x in rows)/wall,
          "ttfa_p50_s":statistics.median(x["ttfa_s"] for x in rows),
          "total_p50_s":statistics.median(ordered),"total_p95_s":ordered[int(.95*(len(ordered)-1))],
          "rows":rows}
        (Path(a.out)/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        print("bench",c,len(rows),round(wall,3),flush=True)
    (Path(a.out)/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print("DONE",a.arm,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--arm",required=True);p.add_argument("--port",type=int,default=8760)
    p.add_argument("--out",required=True);p.add_argument("--ref",required=True)
    asyncio.run(run(p.parse_args()))
