import av
import hashlib
import json
import statistics
from pathlib import Path
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT=Path('/dev/shm')

def read_video(path):
    with av.open(path) as c:
        frames=[frame.to_ndarray(format='rgb24') for frame in c.decode(video=0)]
    return frames

def compare(a,b,label):
    aa,bb=read_video(a),read_video(b)
    assert len(aa)==len(bb)==33,(len(aa),len(bb))
    assert all(x.shape==y.shape==(480,832,3) for x,y in zip(aa,bb))
    psnr=[float(peak_signal_noise_ratio(x,y,data_range=255)) for x,y in zip(aa,bb)]
    ssim=[float(structural_similarity(x,y,channel_axis=2,data_range=255)) for x,y in zip(aa,bb)]
    return {'label':label,'frames':len(aa),'shape':list(aa[0].shape),
            'file_a':str(a),'file_b':str(b),
            'sha256_a':hashlib.sha256(a.read_bytes()).hexdigest(),
            'sha256_b':hashlib.sha256(b.read_bytes()).hexdigest(),
            'psnr_db_mean':statistics.fmean(psnr),'psnr_db_min':min(psnr),
            'ssim_mean':statistics.fmean(ssim),'ssim_min':min(ssim),
            'per_frame_psnr_db':psnr,'per_frame_ssim':ssim}

out={'source_head':'1546535ace4eb7c4f1b57bf89dc3df4d94b2e0d3',
     'checkpoint_revision':'b8fff7315c768468a5333511427288870b2e9635',
     'decoder':'PyAV '+av.__version__,'ssim_library':'scikit-image '+skimage.__version__,
     'comparisons':[]}
for seed in (41,42,43):
    out['comparisons'].append(compare(ROOT/f'pr1540-bf16/seed{seed}.mp4',ROOT/f'pr1540-fp8/seed{seed}.mp4',f'bf16_vs_fp8_seed{seed}'))
out['comparisons'].append(compare(ROOT/'pr1540-fp8/seed41.mp4',ROOT/'pr1540-fp8/seed41-repeat.mp4','fp8_repeat_seed41'))
out['comparisons'].append(compare(ROOT/'pr1540-bf16/seed41.mp4',ROOT/'pr1540-bf16/seed41-repeat.mp4','bf16_repeat_seed41'))
out['comparisons'].append(compare(ROOT/'pr1540-bf16/seed41.mp4',ROOT/'pr1540-ablate/seed41.mp4','bf16_vs_keep_first_last_seed41'))
out['comparisons'].append(compare(ROOT/'pr1540-fp8/seed41.mp4',ROOT/'pr1540-ablate/seed41.mp4','full_fp8_vs_keep_first_last_seed41'))
Path('/dev/shm/pr1540-quality.json').write_text(json.dumps(out,indent=2))
for row in out['comparisons']:
    print(json.dumps({k:row[k] for k in ('label','frames','psnr_db_mean','psnr_db_min','ssim_mean','ssim_min')}))
