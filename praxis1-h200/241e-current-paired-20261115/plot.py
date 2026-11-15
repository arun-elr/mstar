#!/usr/bin/env python3
"""Validate one paired DROID H200 run and render charts from its raw records."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from html import escape
from pathlib import Path

REF = '#24364b'
MSTAR = '#0b8f82'
INK = '#172638'
MUTED = '#536575'
GRID = '#dce4e8'
REFERENCE_REVISION = '68f4d7a254b4bf41951d2e97f44b07f6cac2bf69'
TASKS = ('lift_cube','cube_on_plate','block_in_bowl','block_out_of_bowl',
         'move_block','stack_block','press_button')

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path: Path):
    return json.loads(path.read_text())

def episodes(source: Path, arm: str) -> list[dict]:
    flat=source/f'{arm}-episodes.jsonl'
    if flat.is_file(): return [json.loads(s) for s in flat.read_text().splitlines() if s.strip()]
    paths=sorted((source/arm/'suite').glob('*/episode_*.json'))
    return [read(p) for p in paths]

def validate(source: Path, rev: str):
    required=['paired-56-report.json','paired-commands.json','mstar-matched.json',
      'reference-matched.json','gpu.txt','source-match.log','check-results.log']
    missing=[name for name in required if not (source/name).is_file()]
    if missing: raise ValueError(f'missing paired results: {missing}')
    if not (source/'complete.txt').is_file():
        raise ValueError('the paired run did not complete its final validator')
    if 'NVIDIA H200' not in (source/'gpu.txt').read_text(): raise ValueError('not H200')
    if 'source matched' not in (source/'source-match.log').read_text():
        raise ValueError('extension/server source match not recorded')
    report=read(source/'paired-56-report.json')
    commands=read(source/'paired-commands.json')
    m=read(source/'mstar-matched.json')
    r=read(source/'reference-matched.json')
    me,re=episodes(source,'mstar'),episodes(source,'reference')
    expected={(task,scene,0) for task in TASKS for scene in range(8)}
    key=lambda row:(row['task'],int(row['seed']),int(row['sampler_seed']))
    if len(me)!=56 or len(re)!=56 or {key(x) for x in me}!={key(x) for x in re} or {key(x) for x in me}!=expected:
        raise ValueError('scene/seed units are not the exact matched 56')
    if report['units']!=56: raise ValueError('report unit count')
    if {row['server_meta']['code_version'] for row in me}!={rev}:
        raise ValueError('M* scene records are not pinned to requested commit')
    if m['reference_client_commit']!=r['reference_client_commit'] or m['reference_client_commit']!=REFERENCE_REVISION:
        raise ValueError('reference client revision mismatch')
    if m['script_sha256']!=r['script_sha256'] or m['observation_sha256']!=r['observation_sha256']:
        raise ValueError('matched latency client or input bytes differ')
    if not commands['same_observation_bytes'] or commands['compared_command_values']<=0:
        raise ValueError('command comparison missing')
    if m['sessions']!=r['sessions'] or m['warmup_sessions']!=r['warmup_sessions'] or m['sessions']<=m['warmup_sessions']:
        raise ValueError('session sample counts differ')
    successes_m=sum(bool(x['success']) for x in me)
    successes_r=sum(bool(x['success']) for x in re)
    if (report['total_mstar_successes'],report['total_reference_successes'])!=(successes_m,successes_r):
        raise ValueError('reported successes differ from raw scenes')
    by_m={key(x):x for x in me}; by_r={key(x):x for x in re}
    for task in TASKS:
        row=report['per_task'][task]
        if row['episodes']!=8 or row['mstar_successes']!=sum(bool(by_m[(task,s,0)]['success']) for s in range(8)) \
          or row['reference_successes']!=sum(bool(by_r[(task,s,0)]['success']) for s in range(8)):
            raise ValueError(f'per-task raw/report mismatch: {task}')
    for arm in (m,r):
        for call in range(3):
            samples=[v['round_trip_ms'] for v in arm['calls'] if v['session']>=arm['warmup_sessions'] and v['call']==call]
            if len(samples)!=arm['summary'][str(call)]['n'] or not samples:
                raise ValueError('matched call sample count mismatch')
            if not math.isclose(statistics.median(samples),arm['summary'][str(call)]['p50_ms'],abs_tol=1e-5):
                raise ValueError('matched p50 is not from raw samples')
    return report,commands,m,r,me,re

def txt(x,y,value,size=15,color=INK,weight=400,anchor='start'):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')
def rect(x,y,w,h,color): return f'<rect x="{x}" y="{y}" width="{w:.2f}" height="{h}" rx="3" fill="{color}"/>'
def line(x1,y1,x2,y2,color=GRID): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"/>'
def save(out,name,width,height,parts):
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
         '<style>text{font-family:Inter,DejaVu Sans,Arial,sans-serif}</style>'+''.join(parts)+'</svg>\n')
    path=out/f'{name}.svg';path.write_text(svg)
    subprocess.run(['rsvg-convert','-o',str(out/f'{name}.png'),str(path)],check=True)

def latency(out,m,r,rev):
    values=[arm['summary'][str(c)]['p50_ms'] for arm in (m,r) for c in range(3)]
    end=math.ceil(max(values)*1.14/100)*100
    scale=530/end;x0=300
    p=[txt(58,55,'Praxis DROID session latency',28,weight=650),
       txt(58,86,'One H200, identical observations, sequential server arms',15,MUTED),
       rect(58,108,14,14,REF),txt(82,121,'Reference',14,MUTED),
       rect(206,108,14,14,MSTAR),txt(230,121,'M*',14,MUTED),
       txt(935,121,'p50 round trip',14,MUTED,anchor='end')]
    for tick in range(0,end+1,max(100,end//6)):
        x=x0+scale*tick;p.extend((line(x,156,x,475),txt(x,496,tick,12,MUTED,anchor='middle')))
    for k,y,label in ((0,194,'Cold, 21 rows'),(1,302,'Warm 1, 24 rows'),(2,410,'Warm 2, 24 rows')):
        a=r['summary'][str(k)]['p50_ms'];b=m['summary'][str(k)]['p50_ms']
        p.extend((txt(58,y+10,label,15,weight=550),rect(x0,y-17,a*scale,23,REF),
          rect(x0,y+15,b*scale,23,MSTAR),txt(x0+a*scale+8,y,f'{a:.1f} ms',13,REF,600),
          txt(x0+b*scale+8,y+32,f'{b:.1f} ms ({b/a:.2f}x)',13,MSTAR,600)))
    n=m['sessions']-m['warmup_sessions']
    p.extend((txt(565,528,'milliseconds per call',14,MUTED,anchor='middle'),
      txt(58,558,f'{n} measured sessions after {m["warmup_sessions"]} warmup; three calls each. M* {rev[:10]}; reference {REFERENCE_REVISION[:10]}.',12,MUTED),
      txt(58,579,'Bring-up settings: M* -O0; reference compile=None, action CUDA graphs off.',12,MUTED)))
    save(out,'droid-session-latency',1000,610,p)

def outcomes(out,report,me,re):
    ms=report['total_mstar_successes'];rs=report['total_reference_successes']
    p=[txt(58,55,'Praxis DROID simulation guard',28,weight=650),
       txt(58,86,'Fresh sequential arms on one H200, 7 tasks x 8 scenes, sampler seed 0',15,MUTED),
       rect(58,108,14,14,REF),txt(82,121,f'Reference {rs}/56',14,MUTED),
       rect(258,108,14,14,MSTAR),txt(282,121,f'M* {ms}/56',14,MUTED)]
    x0,scale=300,535/8
    for tick in (0,2,4,6,8):
        x=x0+tick*scale;p.extend((line(x,155,x,681),txt(x,703,tick,12,MUTED,anchor='middle')))
    for i,task in enumerate(TASKS):
        y=185+72*i;row=report['per_task'][task]
        a,b=row['reference_successes'],row['mstar_successes']
        p.extend((txt(58,y+12,task.replace('_',' '),15,weight=550),
          rect(x0,y-17,a*scale,21,REF),rect(x0,y+9,b*scale,21,MSTAR),
          txt(x0+a*scale+8,y,f'{a}/8',13,REF,600),
          txt(x0+b*scale+8,y+26,f'{b}/8',13,MSTAR,600)))
    discord=report['discordant']
    label=f'{len(discord)} discordant scenes: '+', '.join(f"{x['task']}/{x['scene_seed']}" for x in discord)
    p.extend((txt(58,739,label[:130],12,MUTED),
      txt(58,760,'Same scene seeds; VAEs differ. Scene outcomes do not prove internal parity.',12,MUTED)))
    save(out,'droid-simulation-56',1000,785,p)

def differences(out,commands,m):
    count=commands['compared_command_values']
    vals=[commands['by_call'][str(k)][name] for k in range(3) for name in ('p95_abs','max_abs')]
    end=math.ceil(max(vals)*1.15*100)/100
    if end==0:end=.01
    scale=530/end;x0=300
    p=[txt(58,55,'Composed actuator differences',28,weight=650),
      txt(58,86,f'Identical synthetic observations; {count:,} values across {m["sessions"]} sessions',15,MUTED),
      rect(58,109,14,14,REF),txt(82,122,'95th percentile absolute',14,MUTED),
      rect(317,109,14,14,MSTAR),txt(341,122,'maximum absolute',14,MUTED)]
    for i in range(6):
        tick=end*i/5;x=x0+scale*tick
        p.extend((line(x,160,x,475),txt(x,498,f'{tick:.3f}',12,MUTED,anchor='middle')))
    for k,y,label in ((0,199,'Cold, 21 rows'),(1,307,'Warm 1, 24 rows'),(2,415,'Warm 2, 24 rows')):
        row=commands['by_call'][str(k)];a,b=row['p95_abs'],row['max_abs']
        p.extend((txt(58,y+7,label,15,weight=550),rect(x0,y-17,a*scale,22,REF),
          rect(x0,y+15,b*scale,22,MSTAR),txt(x0+a*scale+8,y,f'{a:.4f}',13,REF,600),
          txt(x0+b*scale+8,y+32,f'{b:.4f}',13,MSTAR,600)))
    p.extend((txt(58,547,f"Overall max {commands['max_abs']:.4f}; p95 {commands['p95_abs']:.4f}. Timestamps excluded.",12,MUTED),
      txt(58,568,'Descriptive only; no action-equivalence threshold was predeclared.',12,MUTED)))
    save(out,'droid-command-differences',1000,595,p)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True,help='remote run log directory after salvage')
    p.add_argument('--out',type=Path,required=True,help='new, empty output directory')
    p.add_argument('--mstar-revision',required=True)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()): raise SystemExit('output directory must be empty')
    report,commands,m,r,me,re=validate(a.raw,a.mstar_revision)
    a.out.mkdir(parents=True,exist_ok=True)
    dst=a.out/'source';dst.mkdir()
    names=('paired-56-report.json','paired-commands.json','mstar-matched.json','reference-matched.json',
      'gpu.txt','source-match.log','check-results.log','complete.txt','mstar-served.yaml',
      'reference-sim-served.yaml','reference-session-served.yaml','input-hashes.sha256',
      'model-files-verified.log','runner-and-config-hashes.sha256','reference-working-tree.actual.diff',
      'reference-working-tree.actual.status')
    for name in names:
        source=a.raw/name
        if source.is_file():shutil.copy2(source,dst/name)
    for arm,rows in (('mstar',me),('reference',re)):
        (dst/f'{arm}-episodes.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
    shutil.copy2(Path(__file__),a.out/'plot.py')
    latency(a.out,m,r,a.mstar_revision)
    outcomes(a.out,report,me,re)
    differences(a.out,commands,m)
    provenance={
      'mstar_source_commit':a.mstar_revision,'reference_source_commit':REFERENCE_REVISION,
      'gpu':(dst/'gpu.txt').read_text().strip(),
      'arms':'M* then fresh reference, sequential on one H200',
      'serve_gate':'committed DROID startup gate; YAM gated',
      'simulation':{'scenes':56,'mstar_successes':report['total_mstar_successes'],
        'reference_successes':report['total_reference_successes'],'discordant':report['discordant']},
      'matched_sessions':{'sessions':m['sessions'],'warmup_sessions':m['warmup_sessions'],
        'script_sha256':m['script_sha256'],'observation_sha256':m['observation_sha256'],
        'mstar_p50_ms':{k:v['p50_ms'] for k,v in m['summary'].items()},
        'reference_p50_ms':{k:v['p50_ms'] for k,v in r['summary'].items()},
        'command_values':commands['compared_command_values'],
        'command_max_abs':commands['max_abs'],'command_p95_abs':commands['p95_abs']},
      'limits':['Bring-up settings, not optimized production latency',
        'different VAE implementations; no internal numerical parity claim',
        f'{m["sessions"]-m["warmup_sessions"]} measured matched sessions in one run'],
      'source_sha256':{str(x.relative_to(a.out)):sha(x) for x in sorted(dst.iterdir()) if x.is_file()},
    }
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    files=sorted(x for x in a.out.rglob('*') if x.is_file() and x.name!='artifact-manifest.sha256')
    (a.out/'artifact-manifest.sha256').write_text(''.join(f'{sha(x)}  {x.relative_to(a.out)}\n' for x in files))
    print(json.dumps({'out':str(a.out),'mstar_successes':report['total_mstar_successes'],
      'reference_successes':report['total_reference_successes'],
      'mstar_p50_ms':provenance['matched_sessions']['mstar_p50_ms'],
      'reference_p50_ms':provenance['matched_sessions']['reference_p50_ms']},sort_keys=True))

if __name__=='__main__':main()
