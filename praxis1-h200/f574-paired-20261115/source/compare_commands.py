#!/usr/bin/env python3
"""Compare paired served DROID command rows; timestamps are excluded."""
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('reference', type=Path)
p.add_argument('mstar', type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
ref = json.loads(a.reference.read_text())
new = json.loads(a.mstar.read_text())
assert ref['observation_sha256'] == new['observation_sha256'], 'input bytes differ'
assert len(ref['calls']) == len(new['calls']), 'call counts differ'
errors = []
by_call = {0: [], 1: [], 2: []}
command_key_differences = []
for left, right in zip(ref['calls'], new['calls'], strict=True):
    assert (left['session'], left['call'], left['rows']) == (right['session'], right['call'], right['rows'])
    if left['command_keys'] != right['command_keys']:
        command_key_differences.append({'session': left['session'], 'call': left['call'], 'reference': left['command_keys'], 'mstar': right['command_keys']})
    for lrow, rrow in zip(left['numeric_commands'], right['numeric_commands'], strict=True):
        assert len(lrow) == len(rrow) == 13
        for lval, rval in zip(lrow, rrow, strict=True):
            assert math.isfinite(lval) and math.isfinite(rval)
            error = abs(lval - rval)
            errors.append(error)
            by_call[left['call']].append(error)
errors.sort()
def p95(values):
    values = sorted(values)
    return values[min(len(values)-1, math.ceil(.95*len(values))-1)]
result = {
    'same_observation_bytes': True,
    'command_key_differences': command_key_differences,
    'compared_command_values': len(errors),
    'max_abs': max(errors),
    'p95_abs': p95(errors),
    'by_call': {str(k): {'max_abs': max(v), 'p95_abs': p95(v)} for k,v in by_call.items()},
    'reference_summary': ref['summary'],
    'mstar_summary': new['summary'],
    'note': 'Composed commands only; no internal VAE/denoiser stage comparison. No numerical acceptance threshold is inferred.',
}
a.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
