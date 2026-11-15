#!/usr/bin/env python3
"""Build local, pinned Praxis DROID validation charts from raw JSON."""

import hashlib
import json
import math
import re as regex
import statistics
import tarfile
from collections import defaultdict
from html import escape
from pathlib import Path

ROOT = Path('/tmp/mstar-e5-3h/results-partial/e5-3h-ccff519-attempt2')
REF_ARCHIVE = Path('/tmp/mstar-e5-3h/prior-reference-56-episodes.tar.gz')
OUT = Path('/tmp/mstar-e5-current-head-charts-20261115')
OUT.mkdir(exist_ok=True)

NAVY = '#22354c'
TEAL = '#0b8f82'
INK = '#172638'
MUTED = '#536575'
GRID = '#dce4e8'
BG = '#ffffff'
PALE = '#f4f8f9'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pctl(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo) if lo != hi else values[lo]


def line(x1, y1, x2, y2, color=GRID, width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'


def text(x, y, value, size=16, color=INK, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')


def rect(x, y, w, h, color, radius=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{color}"/>'


def svg(parts, width, height):
    return ('<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" fill="{BG}"/>'
            '<style>text{font-family:Inter,DejaVu Sans,Arial,sans-serif}</style>'
            + ''.join(parts) + '</svg>\n')


def load():
    m = json.loads((ROOT / 'mstar-matched.json').read_text())
    r = json.loads((ROOT / 'reference-matched.json').read_text())
    c = json.loads((ROOT / 'paired-commands.json').read_text())
    me = [json.loads(s) for s in (ROOT / 'mstar-56/suite/episodes.jsonl').read_text().splitlines()]
    with tarfile.open(REF_ARCHIVE, 'r:gz') as t:
        f = t.extractfile('praxis1-h200/droid56-simulation/reference/suite/episodes.jsonl')
        re = [json.loads(s) for s in f.read().decode().splitlines()]
    assert m['mode'] == 'mstar' and r['mode'] == 'reference'
    assert m['gpu'] == r['gpu'] and 'H200' in m['gpu']
    assert m['reference_client_commit'] == r['reference_client_commit'] == '68f4d7a254b4bf41951d2e97f44b07f6cac2bf69'
    assert m['client_dirty'] and r['client_dirty']
    assert m['script_sha256'] == r['script_sha256']
    assert m['observation_sha256'] == r['observation_sha256']
    assert m['sessions'] == r['sessions'] == 5
    assert m['warmup_sessions'] == r['warmup_sessions'] == 1
    assert c['same_observation_bytes'] and c['compared_command_values'] == 4485
    assert sum(x['rows'] * 13 for x in m['calls']) == 4485
    assert sum(x['rows'] * 13 for x in r['calls']) == 4485
    assert 'action_cuda_graphs: false' in (ROOT / 'ref_serve.yaml').read_text()
    assert 'nvfp4_moe_cuda_graphs: false' in (ROOT / 'ref_serve.yaml').read_text()
    assert '--port "$PORT" -O0' in Path('/tmp/mstar-e5-3h/mstar_serve_wrapper.sh').read_text()
    server_log = (ROOT / 'e5-droid-openpi-rust-server-8127.log').read_text()
    captured_mib = [int(x) for x in regex.findall(r'CapturePlan\[cuda:0\].*capture took (\d+) MiB', server_log)]
    assert len(captured_mib) >= 5 and all(x == 0 for x in captured_mib)
    assert len(me) == len(re) == 56
    key = lambda a: (a['task'], a['seed'], a['sampler_seed'])
    assert len({key(a) for a in me}) == len({key(a) for a in re}) == 56
    assert {key(a) for a in me} == {key(a) for a in re}
    assert {a['server_meta']['code_version'] for a in me} == {'ccff51938b'}
    prior = json.loads((ROOT / 'prior-weight-comparison.json').read_text())
    assert prior['missing'] == prior['mismatched'] == []
    assert prior['current_substantive_file_count'] == 39
    assert prior['reference_worktree_diff_byte_equal']
    assert prior['reference_status_byte_equal']
    assert sha(ROOT / 'sim_droid.yaml') == sha(Path('/tmp/mstar-e5-3h/reference-served-candidate.yaml'))
    assert (ROOT / 'reference-56-provenance.txt').read_text().startswith('Reused audited 55/56 reference arm:')
    for a, name in ((m, 'M*'), (r, 'reference')):
        for k in (0, 1, 2):
            calls = [x['round_trip_ms'] for x in a['calls'] if x['session'] >= 1 and x['call'] == k]
            assert len(calls) == 4, (name, k, len(calls))
            assert abs(statistics.median(calls) - a['summary'][str(k)]['p50_ms']) < 1e-5
            assert abs(pctl(calls, 95) - a['summary'][str(k)]['p95_ms']) < 1e-5
    return m, r, c, me, re


def latency_chart(m, r):
    p = [rect(0, 0, 1000, 630, BG), text(60, 58, 'Praxis DROID session latency', 28, INK, 650),
         text(60, 90, 'Same H200, identical observation bytes; 4 measured sessions after 1 warmup', 15, MUTED),
         rect(60, 111, 13, 13, NAVY, 2), text(80, 123, 'Reference', 14, MUTED),
         rect(205, 111, 13, 13, TEAL, 2), text(225, 123, 'M*', 14, MUTED),
         text(932, 123, 'p50 round trip', 14, MUTED, anchor='end')]
    x0, scale = 275, 590 / 600
    for tick in (0, 100, 200, 300, 400, 500, 600):
        x = x0 + tick * scale
        p += [line(x, 160, x, 475), text(x, 499, tick, 12, MUTED, anchor='middle')]
    p.append(text(570, 533, 'milliseconds per call', 14, MUTED, anchor='middle'))
    for k, y, label in ((0, 199, 'Cold, 21 rows'), (1, 304, 'Warm 1, 24 rows'), (2, 409, 'Warm 2, 24 rows')):
        rv = r['summary'][str(k)]['p50_ms']
        mv = m['summary'][str(k)]['p50_ms']
        p += [text(60, y + 9, label, 16, INK, 550),
              rect(x0, y - 16, rv * scale, 23, NAVY, 4),
              rect(x0, y + 16, mv * scale, 23, TEAL, 4),
              text(x0 + rv * scale + 9, y + 1, f'{rv:.0f} ms', 14, NAVY, 600),
              text(x0 + mv * scale + 9, y + 33, f'{mv:.0f} ms  ({mv/rv:.2f}x ref)', 14, TEAL, 600)]
    p.append(text(60, 565, 'Reference 68f4d7a (recorded dirty harness)  |  M* ccff51938b  |  synthetic 1/22/25-frame calls', 12, MUTED))
    p.append(text(60, 585, 'Bring-up settings: M* served with -O0; server logged 0 MiB capture across Praxis nodes.', 12, MUTED))
    p.append(text(60, 605, 'Reference action and NVFP4 MoE CUDA graphs off.', 12, MUTED))
    (OUT / 'droid-session-latency.svg').write_text(svg(p, 1000, 630))


def simulation_chart(me, re):
    task_order = ['lift_cube', 'cube_on_plate', 'block_in_bowl', 'block_out_of_bowl', 'move_block', 'stack_block', 'press_button']
    mby = defaultdict(list)
    rby = defaultdict(list)
    for a in me: mby[a['task']].append(a)
    for a in re: rby[a['task']].append(a)
    mtotal = sum(a['success'] for a in me)
    rtotal = sum(a['success'] for a in re)
    assert (mtotal, rtotal) == (54, 55)
    p = [text(60, 56, 'DROID simulation guard', 28, INK, 650),
         text(60, 86, '7 tasks × 8 paired scene seeds; sampler seed 0', 15, MUTED),
         rect(60, 111, 13, 13, NAVY, 2), text(80, 123, f'Reference: {rtotal}/56', 14, MUTED),
         rect(260, 111, 13, 13, TEAL, 2), text(280, 123, f'M*: {mtotal}/56', 14, MUTED),
         text(927, 123, 'successful scenes', 14, MUTED, anchor='end')]
    x0, scale = 285, 565 / 8
    for tick in (0, 2, 4, 6, 8):
        x = x0 + tick * scale
        p += [line(x, 155, x, 680), text(x, 705, tick, 13, MUTED, anchor='middle')]
    for i, task in enumerate(task_order):
        y = 190 + i * 72
        rs = sum(a['success'] for a in rby[task]); ms = sum(a['success'] for a in mby[task])
        assert len(rby[task]) == len(mby[task]) == 8
        p += [text(60, y + 13, task.replace('_', ' '), 15, INK, 550),
              rect(x0, y - 16, rs * scale, 20, NAVY, 4),
              rect(x0, y + 10, ms * scale, 20, TEAL, 4),
              text(x0 + rs * scale + 8, y, f'{rs}/8', 13, NAVY, 600),
              text(x0 + ms * scale + 8, y + 26, f'{ms}/8', 13, TEAL, 600)]
    p += [text(60, 744, 'Outcome baseline: prior audited reference run; source, checkpoint, config and input recipe hashes match.', 12, MUTED),
          text(60, 764, 'Reference and M* simulation outcomes are paired by task and scene seed; latency is not compared across these runs.', 12, MUTED),
          text(60, 784, 'M* misses block_out_of_bowl/1 and lift_cube/6; reference misses stack_block/0.', 12, MUTED)]
    (OUT / 'droid-simulation-56.svg').write_text(svg(p, 1000, 815))


def command_chart(c):
    p = [text(60, 56, 'Composed command differences', 28, INK, 650),
         text(60, 86, 'Same synthetic observation bytes and seed; 4,485 actuator values across 5 sessions (incl. warmup)', 15, MUTED),
         rect(60, 111, 13, 13, NAVY, 2), text(80, 123, '95th percentile absolute difference', 14, MUTED),
         rect(366, 111, 13, 13, TEAL, 2), text(386, 123, 'maximum absolute difference', 14, MUTED)]
    x0, scale = 290, 585 / .09
    for tick in (0, .02, .04, .06, .08):
        x = x0 + tick * scale
        p += [line(x, 165, x, 475), text(x, 502, f'{tick:.2f}', 12, MUTED, anchor='middle')]
    for k, y, label in ((0, 210, 'Cold, 21 rows'), (1, 320, 'Warm 1, 24 rows'), (2, 430, 'Warm 2, 24 rows')):
        d = c['by_call'][str(k)]
        p += [text(60, y + 5, label, 16, INK, 550),
              rect(x0, y - 25, d['p95_abs'] * scale, 21, NAVY, 4),
              rect(x0, y + 8, d['max_abs'] * scale, 21, TEAL, 4),
              text(x0 + d['p95_abs'] * scale + 8, y - 9, f"{d['p95_abs']:.3f}", 13, NAVY, 600),
              text(x0 + d['max_abs'] * scale + 8, y + 24, f"{d['max_abs']:.3f}", 13, TEAL, 600)]
    p += [text(60, 550, 'Pose and grip values only; reference timestamp metadata is excluded.', 12, MUTED),
          text(60, 570, 'This is output difference, not an acceptance threshold or closed-loop parity measure.', 12, MUTED)]
    (OUT / 'droid-command-differences.svg').write_text(svg(p, 1000, 600))


def main():
    m, r, c, me, re = load()
    latency_chart(m, r)
    simulation_chart(me, re)
    command_chart(c)
    mk = {(a['task'], a['seed']): a for a in me}
    rk = {(a['task'], a['seed']): a for a in re}
    discordant = [dict(task=t, scene_seed=s, mstar_success=bool(mk[t,s]['success']), reference_success=bool(rk[t,s]['success']))
                  for t,s in sorted(mk) if mk[t,s]['success'] != rk[t,s]['success']]
    prov = {
        'mstar_commit': 'ccff51938b81ec3d02ecd473c69e9c2323e5f6da',
        'reference_commit': m['reference_client_commit'],
        'reference_worktree_dirty': True,
        'reference_worktree_status_file': str(ROOT / 'reference-status.txt'),
        'hardware': m['gpu'],
        'paired_session_mode': 'same H200 and observation bytes, sequential arms, 5 sessions with first as warmup',
        'paired_session_script_sha256': m['script_sha256'],
        'generator_sha256': sha(Path(__file__)),
        'png_render': 'rsvg-convert version 2.58.0; rsvg-convert input.svg -o output.png',
        'raw_inputs_in_asset_commit': False,
        'matched_observation_sha256': m['observation_sha256'],
        'session_p50_ms': {k: {'reference': r['summary'][k]['p50_ms'], 'mstar': m['summary'][k]['p50_ms']}
                           for k in ('0','1','2')},
        'commands': {'values_compared': c['compared_command_values'], 'sessions_counted': 5,
                     'includes_warmup_session': True, 'max_abs': c['max_abs'], 'p95_abs': c['p95_abs'],
                     'reference_metadata_timestamp_excluded': True},
        'simulation': {'reference_success': sum(a['success'] for a in re), 'mstar_success': sum(a['success'] for a in me),
                       'scenes': 56, 'sampler_seed': 0, 'discordant': discordant,
                       'reference_outcome_source': 'prior audited reference arm, exact source/checkpoint/config/input recipe hash match',
                       'reference_simulation_latency_reused': False},
        'input_files_sha256': {name: sha(ROOT/name) for name in ('mstar-matched.json','reference-matched.json','paired-commands.json',
                                                                'mstar-56/suite/episodes.jsonl',
                                                                'prior-weight-comparison.json',
                                                                'reference-56-provenance.txt',
                                                                'sim_droid.yaml', 'ref_serve.yaml',
                                                                'reference-head.txt', 'reference-status.txt',
                                                                'diagnostic.patch',
                                                                'e5-droid-openpi-rust-server-8127.log')}
                            | {'prior-reference-56-episodes.tar.gz': sha(REF_ARCHIVE),
                               'reference-served-candidate.yaml': sha(Path('/tmp/mstar-e5-3h/reference-served-candidate.yaml')),
                               'mstar_serve_wrapper.sh': sha(Path('/tmp/mstar-e5-3h/mstar_serve_wrapper.sh'))},
    }
    (OUT / 'provenance.json').write_text(json.dumps(prov, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output':str(OUT), 'session_p50_ms':prov['session_p50_ms'],
                      'simulation':prov['simulation'], 'chart_files':[p.name for p in OUT.glob('*.svg')]}, indent=2))


if __name__ == '__main__':
    main()
