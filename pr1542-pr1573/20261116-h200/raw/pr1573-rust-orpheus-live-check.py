import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:8613'
KEY = Path('/dev/shm/pr1573-api-key').read_text().strip()
OUT = Path('/dev/shm/pr1573-rust-orpheus-live.json')
rows = []

def call(method, path, body=None, auth=True, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    headers = {'Content-Type': 'application/json'}
    if auth:
        headers['Authorization'] = 'Bearer ' + KEY
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        payload = exc.read()
    try:
        value = json.loads(payload)
    except Exception:
        value = payload.decode(errors='replace')[:300]
    row = {'method': method, 'path': path, 'status': status, 'elapsed_s': round(time.monotonic()-start, 3), 'body': body, 'response': value}
    rows.append(row)
    print(path, status, row['elapsed_s'], flush=True)
    OUT.write_text(json.dumps({'source_head':'5b42f8e906edd29aa16970d95da5ee4a2ad162b7','frontend':'rust','model':'orpheus','checkpoint_revision':'4206a56e5a68cf6cf96900a8a78acd3370c02eb6','checkpoint_shard_sha256':{'1':'8feb9137e2a243d33827c2196ec81270179ec6e7acf8972a37f84b5eb6852450','2':'69df9510f4b3e8d883b51e5d19f5f5b669d99ba9990f9e0e16f4fedd3d73709e','3':'d24f6a0f618bd4aa09a23216fe55d086a6db92ee896f98740972702f265f026b','4':'a1deaa1b3f38d5b9efe22f012dbb5c3f29e15f31b15dcf95e471a85bb03d2d21'},'rows':rows}, indent=2))
    return status, value

s, _ = call('GET', '/health')
assert s == 200
s, _ = call('GET', '/is_paused', auth=False)
assert s == 401
s, state = call('GET', '/is_paused')
assert s == 200 and not state['paused']
payload = json.dumps({'input':'Hello there.','response_format':'wav','seed':7,'voice':'tara','temperature':0}).encode()
req = urllib.request.Request(BASE + '/v1/audio/speech', data=payload, headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'}, method='POST')
start = time.monotonic()
with urllib.request.urlopen(req, timeout=180) as resp:
    before_status = resp.status
    before_audio = resp.read()
import hashlib
before_speech = {'status':before_status,'elapsed_s':round(time.monotonic()-start,3),'size_bytes':len(before_audio),'sha256':hashlib.sha256(before_audio).hexdigest(),'riff_wave':before_audio[:4]==b'RIFF' and before_audio[8:12]==b'WAVE'}
Path('/dev/shm/pr1573-rust-orpheus-before.wav').write_bytes(before_audio)
report = json.loads(OUT.read_text())
report['speech_before_reload'] = before_speech
OUT.write_text(json.dumps(report, indent=2))
print('/v1/audio/speech before', before_speech, flush=True)
assert before_status == 200 and before_speech['riff_wave'] and len(before_audio) > 44
s, _ = call('POST', '/weight_checker', {'mode':'checksum'})
assert s == 409
s, state = call('POST', '/pause_generation', {'mode':'wait'})
assert s == 200 and state['paused']
s, state = call('GET', '/is_paused')
assert s == 200 and state['paused']
s, checks = call('POST', '/weight_checker', {'mode':'checksum'})
assert s == 200 and checks['rank_checksums']
original = checks['rank_checksums']
s, _ = call('POST', '/weight_checker', {'mode':'compare','checksums':original})
assert s == 200
wrong = {k:'0'*64 for k in original}
s, _ = call('POST', '/weight_checker', {'mode':'compare','checksums':wrong})
assert s == 409
s, _ = call('POST', '/update_weights_from_disk', {'model_path':'<outside-reload-root>','weight_version':'bad'})
assert s == 400
s, update = call('POST', '/update_weights_from_disk', {'model_path':'<orpheus-checkpoint>','weight_version':'v2'})
assert s == 200 and update['weight_version'] == 'v2'
s, checks = call('POST', '/weight_checker', {'mode':'checksum'})
assert s == 200 and checks['rank_checksums'] == original
s, state = call('POST', '/continue_generation', {})
assert s == 200 and not state['paused']
s, state = call('GET', '/is_paused')
assert s == 200 and not state['paused'] and state['weight_version'] == 'v2'
s, models = call('GET', '/v1/models')
assert s == 200
payload = json.dumps({'input':'Hello there.','response_format':'wav','seed':7,'voice':'tara','temperature':0}).encode()
req = urllib.request.Request(BASE + '/v1/audio/speech', data=payload, headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'}, method='POST')
start = time.monotonic()
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        speech_status = resp.status
        audio = resp.read()
except urllib.error.HTTPError as exc:
    speech_status = exc.code
    audio = exc.read()
import hashlib
speech = {'status':speech_status,'elapsed_s':round(time.monotonic()-start,3),'size_bytes':len(audio),'sha256':hashlib.sha256(audio).hexdigest(),'riff_wave':audio[:4]==b'RIFF' and audio[8:12]==b'WAVE'}
Path('/dev/shm/pr1573-rust-orpheus-after.wav').write_bytes(audio)
report = json.loads(OUT.read_text())
report['speech_after_reload'] = speech
OUT.write_text(json.dumps(report, indent=2))
print('/v1/audio/speech', speech, flush=True)
assert speech_status == 200 and speech['riff_wave'] and len(audio) > 44
print('PASS', flush=True)
