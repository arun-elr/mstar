import asyncio
import json
import time

import aiohttp

URL = "http://127.0.0.1:8620/generate"


async def call(session, rid, n=64, temperature=0.0):
    form = aiohttp.FormData()
    form.add_field("request_id", rid)
    form.add_field("text", "Write a short paragraph about the planets.")
    form.add_field("output_modalities", "text")
    form.add_field("streaming", "false")
    form.add_field("model_kwargs", json.dumps({"max_output_tokens": n, "ignore_eos": True, "temperature": temperature, "enable_thinking": False}))
    t = time.monotonic()
    async with session.post(URL, data=form) as resp:
        raw = await resp.text()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw[:1000]}
        return {"status": resp.status, "latency_s": time.monotonic() - t, "body": parsed}


async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        first = await call(session, "same-client-id", n=32, temperature=0.7)
        second = await call(session, "same-client-id", n=32, temperature=0.7)
        ongoing = asyncio.create_task(call(session, "duplicate-in-flight", n=2048))
        await asyncio.sleep(0.05)
        duplicate = await call(session, "duplicate-in-flight", n=32)
        ongoing_result = await ongoing
        separate = await call(session, "other-client-id", n=32, temperature=0.7)
    out = {"reuse_first": first, "reuse_second": second,
           "ongoing": ongoing_result, "duplicate": duplicate, "separate": separate}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
