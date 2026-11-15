#!/usr/bin/env python3
"""Matched Qwen3.5-2B streaming A/B client with saved per-request samples."""
import argparse
import base64
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

p = argparse.ArgumentParser()
p.add_argument('--url', required=True)
p.add_argument('--label', required=True)
p.add_argument('--out', required=True, type=Path)
p.add_argument('--samples', type=int, default=50)
p.add_argument('--max-tokens', type=int, default=32)
a = p.parse_args()
PROMPTS = [
    'Explain how a GPU warp schedules a matrix multiplication tile.',
    'Describe a reliable retry protocol for a distributed task queue.',
    'List three causes of numerical drift in bfloat16 inference.',
    'Explain why a cached KV tensor needs a stable position index.',
]

def request(index):
    start = time.perf_counter()
    first = None
    tokens = 0
    output_parts = []
    data = {'text': PROMPTS[index % len(PROMPTS)], 'streaming': 'true',
            'output_modalities': 'text',
            'model_kwargs': json.dumps({'max_output_tokens': a.max_tokens,
                                        'temperature': 0, 'ignore_eos': True})}
    with requests.post(a.url.rstrip('/')+'/generate', data=data, stream=True, timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            item = json.loads(line)
            if item['modality'] == 'error':
                raise RuntimeError(base64.b64decode(item['data']).decode())
            if item['modality'] == 'text':
                tokens += 1
                first = first or time.perf_counter()
                output_parts.append(base64.b64decode(item['data']))
    end = time.perf_counter()
    if tokens != a.max_tokens:
        raise RuntimeError(f'expected {a.max_tokens} output tokens; got {tokens}')
    output = b''.join(output_parts)
    return {'ttft_ms': 1000*((first or start)-start),
            'total_ms': 1000*(end-start), 'tokens': tokens,
            'prompt_index': index % len(PROMPTS),
            'output_sha256': hashlib.sha256(output).hexdigest(),
            'output_text': output.decode('utf-8', errors='replace')}

def measure(concurrency, count, offset):
    start = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        samples = list(pool.map(request, range(offset, offset+count)))
    wall = time.perf_counter()-start
    def stat(key):
        values=sorted(x[key] for x in samples)
        return {'p50': statistics.median(values), 'p95': values[min(len(values)-1, int(.95*len(values)))]}
    return {'concurrency': concurrency, 'requests': count, 'wall_s': wall,
            'throughput_tok_s': sum(x['tokens'] for x in samples)/wall,
            'ttft_ms': stat('ttft_ms'), 'total_ms': stat('total_ms'), 'samples': samples}

rows=[]
for concurrency in (1,4,16):
    measure(concurrency, 8, 0)
    row=measure(concurrency, a.samples, 1000)
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='samples'}), flush=True)
a.out.write_text(json.dumps({'label':a.label,'model':'Qwen3.5-2B',
                             'max_tokens':a.max_tokens,'rows':rows}, indent=2)+'\n')
