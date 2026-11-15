#!/usr/bin/env python3
"""Regenerate the Qwen aggregate A/B chart from six pinned per-server runs."""

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from html import escape
from pathlib import Path

ROOT = Path('/tmp/mstar-e5-abba-3h/results/mstar-abba-ccff-attempt4')
PLAN = Path('/tmp/mstar-e5-abba-3h/README.md')
OUT = Path('/tmp/mstar-qwen-abba-charts-20261115')
OUT.mkdir(exist_ok=True)

MAIN = 'df256450f18530ac981cc26cf5cca5a2cc6a6a0e'
E5 = 'ccff51938b81ec3d02ecd473c69e9c2323e5f6da'
ORDER = ('main', 'e5', 'e5', 'main', 'main', 'e5')
NAVY = '#22354c'
TEAL = '#0b8f82'
INK = '#172638'
MUTED = '#536575'
GRID = '#dce4e8'
MID = '#738695'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p95(values):
    values = sorted(values)
    return values[math.ceil(.95 * len(values)) - 1]


def text(x, y, value, size=16, color=INK, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')


def line(x1, y1, x2, y2, color=GRID, width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'


def rect(x, y, w, h, color, radius=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{color}"/>'


def dot(x, y, color, radius=7):
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="#ffffff" stroke-width="2"/>'


def svg(parts, width=1160, height=920):
    return ('<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
            '<style>text{font-family:Inter,DejaVu Sans,Arial,sans-serif}</style>'
            + ''.join(parts) + '</svg>\n')


def load_and_validate():
    aggregate = json.loads((ROOT / 'aggregate-summary.json').read_text())
    assert tuple(aggregate['order']) == ORDER
    assert 'aggregate stack, not one PR' in aggregate['semantics']
    plan = PLAN.read_text()
    assert MAIN in plan and E5 in plan
    gpu = (ROOT / 'gpu.txt').read_text().strip()
    assert 'NVIDIA H200' in gpu
    assert 'source matched' in (ROOT / 'main-source-match.log').read_text()
    assert 'source matched' in (ROOT / 'e5-source-match.log').read_text()
    assert all(x.endswith('OK') for x in (ROOT / 'qwen-files-verified.log').read_text().splitlines())
    assert 'ABBA scheduled runs complete' in (ROOT / 'complete.txt').read_text()
    assert (ROOT / 'qwen-bench.sha256').read_text().split()[0] == digest(ROOT / 'qwen-bench.yaml')
    order_log = (ROOT / 'order.log').read_text()
    runs = {}
    for i, arm in enumerate(ORDER, 1):
        assert f'run={i} arm={arm} start=' in order_log
        assert f'run={i} arm={arm} end=' in order_log
        path = ROOT / f'run-{i}-{arm}.json'
        assert digest(path) == aggregate['runs'][str(i)]['sha256']
        assert aggregate['runs'][str(i)]['arm'] == arm
        run = json.loads(path.read_text())
        assert run['label'] == arm and run['model'] == 'Qwen3.5-2B' and run['max_tokens'] == 32
        assert {r['concurrency'] for r in run['rows']} == {1, 4, 16}
        server_log = (ROOT / f'run-{i}-{arm}-server.log').read_text()
        graph_mib = [int(x) for x in re.findall(r'LLM: capture took (\d+) MiB', server_log)]
        assert graph_mib and max(graph_mib) > 0
        assert 'prefix cache open for main on LLM' in server_log
        for row in run['rows']:
            assert row['requests'] == len(row['samples']) == 50
            assert all(s['tokens'] == 32 for s in row['samples'])
            assert all(s['prompt_index'] in (0, 1, 2, 3) for s in row['samples'])
            assert math.isclose(sum(s['tokens'] for s in row['samples']) / row['wall_s'], row['throughput_tok_s'], rel_tol=1e-9)
            for field, raw in (('ttft_ms', 'ttft_ms'), ('total_ms', 'total_ms')):
                values = [s[raw] for s in row['samples']]
                assert math.isclose(statistics.median(values), row[field]['p50'], abs_tol=1e-8)
                assert math.isclose(p95(values), row[field]['p95'], abs_tol=1e-8)
        runs[i] = run
    metrics = {'throughput_tok_s': ('throughput_tok_s', None),
               'ttft_p50_ms': ('ttft_ms', 'p50'), 'ttft_p95_ms': ('ttft_ms', 'p95'),
               'total_p50_ms': ('total_ms', 'p50'), 'total_p95_ms': ('total_ms', 'p95')}
    derived = {}
    for concurrency in (1, 4, 16):
        by_arm = {}
        hashes_by_arm = {}
        for arm in ('main', 'e5'):
            arm_runs = [runs[i] for i, a in enumerate(ORDER, 1) if a == arm]
            rows = [next(x for x in run['rows'] if x['concurrency'] == concurrency) for run in arm_runs]
            by_arm[arm] = {}
            hash_sets = {}
            for name, (field, subfield) in metrics.items():
                values = [row[field] if subfield is None else row[field][subfield] for row in rows]
                summary = aggregate['by_concurrency'][str(concurrency)][arm][name]
                assert len(values) == len(summary['per_run']) == 3
                assert all(math.isclose(v, s, rel_tol=1e-10) for v, s in zip(values, summary['per_run']))
                assert math.isclose(statistics.median(values), summary['median'], rel_tol=1e-10)
                assert math.isclose(min(values), summary['min'], rel_tol=1e-10)
                assert math.isclose(max(values), summary['max'], rel_tol=1e-10)
                by_arm[arm][name] = {'median': statistics.median(values), 'per_run': values,
                                     'min': min(values), 'max': max(values)}
            for row in rows:
                for s in row['samples']:
                    hash_sets.setdefault(s['prompt_index'], set()).add(s['output_sha256'])
            for prompt in range(4):
                assert sorted(hash_sets[prompt]) == aggregate['by_concurrency'][str(concurrency)][arm]['output_hashes_by_prompt'][str(prompt)]
            hashes_by_arm[arm] = hash_sets
        for name in metrics:
            ratio = by_arm['e5'][name]['median'] / by_arm['main'][name]['median']
            assert math.isclose(ratio, aggregate['by_concurrency'][str(concurrency)]['e5_to_main_ratio'][name], rel_tol=1e-10)
        for prompt in range(4):
            assert hashes_by_arm['main'][prompt] == hashes_by_arm['e5'][prompt]
            assert aggregate['by_concurrency'][str(concurrency)]['shared_output_hash_sets_by_prompt'][str(prompt)]
        derived[concurrency] = by_arm
    return aggregate, runs, derived, gpu


def chart(derived, gpu):
    p = [text(56, 58, 'Qwen3.5-2B aggregate stack A/B', 29, INK, 650),
         text(56, 90, 'One H200; six fresh servers in main / E5 / E5 / main / main / E5 order', 15, MUTED),
         text(56, 128, 'Streaming throughput', 21, INK, 600),
         text(56, 150, '32 output chunks per request; positive delta means more chunks per second', 14, MUTED)]
    x0, scale = 800, 72
    for tick in (-3, -2, -1, 0, 1, 2, 3):
        x = x0 + tick * scale
        p += [line(x, 175, x, 338, MID if tick == 0 else GRID, 2 if tick == 0 else 1),
              text(x, 356, f'{tick:+d}%' if tick else '0', 12, MUTED, anchor='middle')]
    for c, y in ((1, 213), (4, 262), (16, 311)):
        a = derived[c]
        main = a['main']['throughput_tok_s']['median']
        e5 = a['e5']['throughput_tok_s']['median']
        delta = (e5/main - 1)*100
        x = x0 + delta*scale
        p += [text(56, y+5, f'C{c}: main {main:,.0f}, E5 {e5:,.0f} chunks/s', 15, INK, 500),
              line(x0, y, x, y, TEAL, 5), dot(x, y, TEAL),
              text(1075, y+5, f'{delta:+.2f}%', 15, TEAL, 600, anchor='end')]
    p += [text(56, 410, 'Total request latency', 21, INK, 600),
          text(56, 432, 'Median of three fresh-server p50 or p95 values; positive delta means slower', 14, MUTED)]
    for tick in (-3, -2, -1, 0, 1, 2, 3):
        x = x0 + tick*scale
        p += [line(x, 453, x, 745, MID if tick == 0 else GRID, 2 if tick == 0 else 1),
              text(x, 765, f'{tick:+d}%' if tick else '0', 12, MUTED, anchor='middle')]
    for i, (c, metric, label) in enumerate((
        (1, 'total_p50_ms', 'p50'), (1, 'total_p95_ms', 'p95'),
        (4, 'total_p50_ms', 'p50'), (4, 'total_p95_ms', 'p95'),
        (16, 'total_p50_ms', 'p50'), (16, 'total_p95_ms', 'p95'))):
        y = 477 + i*49
        main = derived[c]['main'][metric]['median']; e5 = derived[c]['e5'][metric]['median']
        delta = (e5/main - 1)*100
        x = x0 + delta*scale
        color = NAVY if label == 'p50' else TEAL
        p += [text(56, y+5, f'C{c} {label}: main {main:.1f}, E5 {e5:.1f} ms', 15, INK, 500),
              line(x0, y, x, y, color, 5), dot(x, y, color),
              text(1075, y+5, f'{delta:+.2f}%', 15, color, 600, anchor='end')]
    p += [text(56, 811, '3 runs/arm, 8 warmups + 50 measured requests at C1/C4/C16; 32 streamed output chunks/request.', 12, MUTED),
          text(56, 830, 'Main df256450, E5 ccff519; both use -O0 and captured Qwen graphs. Aggregate stack, no per-PR attribution.', 12, MUTED),
          text(56, 849, 'Three runs/arm; observed ranges are in the JSON. This repeated-prompt workload limits generalization.', 12, MUTED),
          text(56, 868, 'Output hash sets match per prompt; batched runs can select different completions. No gain/regression claim.', 12, MUTED),
          text(56, 887, 'Run ranges, TTFT, output hash sets and source/config pins are in the aggregate JSON and provenance.', 12, MUTED)]
    (OUT / 'qwen-abba-aggregate.svg').write_text(svg(p))


def main():
    aggregate, runs, derived, gpu = load_and_validate()
    chart(derived, gpu)
    metrics = ('throughput_tok_s','ttft_p50_ms','ttft_p95_ms','total_p50_ms','total_p95_ms')
    prov = {
        'main_commit': MAIN, 'e5_commit': E5, 'hardware': gpu,
        'order': list(ORDER), 'model': 'Qwen3.5-2B', 'config_sha256': digest(ROOT / 'qwen-bench.yaml'),
        'run_protocol': 'six fresh servers, 3 per arm, 8 warmup + 50 measured requests at each concurrency 1/4/16, 32 streamed output chunks per request',
        'summary_statistic': 'median of three per-run statistics; p50 is sample median, p95 is nearest-rank, throughput is measured chunks divided by wall time',
        'graph_capture': 'both arms used -O0; each server log reports nonzero LLM capture MiB',
        'scope': 'aggregate stacked diff from current main to E5; no per-PR attribution',
        'interpretation': 'three runs per arm on a repeated-prompt workload; no general performance gain or regression is claimed',
        'workload_limit': 'four short prompts repeat with prefix caching; this is not mixed traffic',
        'output_limit': 'matching output hash sets per prompt, not identical output frequencies in batched runs',
        'raw_samples_in_assets': False,
        'png_render': 'rsvg-convert version 2.58.0; rsvg-convert qwen-abba-aggregate.svg -o qwen-abba-aggregate.png',
        'generator_sha256': digest(Path(__file__)),
        'source_input_sha256': {f'run-{i}-{arm}.json': digest(ROOT / f'run-{i}-{arm}.json') for i,arm in enumerate(ORDER,1)}
            | {name: digest(ROOT / name) for name in ('aggregate-summary.json','qwen-bench.yaml','order.log','gpu.txt',
                'main-source-match.log','e5-source-match.log','qwen-files-verified.log')}
            | {'plan-README.md': digest(PLAN)},
        'by_concurrency': {str(c): {
            arm: {metric: derived[c][arm][metric] for metric in metrics} for arm in ('main','e5')}
            for c in (1,4,16)},
        'all_shared_output_hash_sets_by_prompt': all(all(aggregate['by_concurrency'][str(c)]['shared_output_hash_sets_by_prompt'].values()) for c in (1,4,16)),
    }
    (OUT / 'provenance.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n')
    (OUT / 'aggregate-summary.json').write_bytes((ROOT / 'aggregate-summary.json').read_bytes())
    (OUT / 'qwen-bench.yaml').write_bytes((ROOT / 'qwen-bench.yaml').read_bytes())
    print(json.dumps({
        'output':str(OUT),
        'deltas_percent':{str(c):{metric:round((derived[c]['e5'][metric]['median']/derived[c]['main'][metric]['median']-1)*100,3) for metric in metrics}
                          for c in (1,4,16)}
    },indent=2))


if __name__=='__main__':
    main()
