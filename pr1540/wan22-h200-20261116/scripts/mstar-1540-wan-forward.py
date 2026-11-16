import json
import statistics
import torch
from mstar.model.wan22.config import Wan22Config
from mstar.model.wan22.weight_loader import build_wan22_dit
from mstar.model.wan22.wan22_model import Wan22Model
from mstar.model.components.quant_linear import QuantLinear

MODEL='/dev/shm/mstar-pr1540-wan'
torch.set_float32_matmul_precision('high')
base=build_wan22_dit(Wan22Config(), MODEL, device='cuda').eval()
fp8=Wan22Model(MODEL, linear_dtype='dense_fp8').get_submodule('dit', device='cuda').transformer.eval()
report={'source_head':open('/dev/shm/mstar-pr1540-src/MEASURED_HEAD').read().strip(),
        'checkpoint_revision':'b8fff7315c768468a5333511427288870b2e9635',
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'quant_linears':sum(isinstance(m,QuantLinear) for m in fp8.modules()),'seeds':[]}
with torch.inference_mode():
    for seed in (41,42,43):
        g=torch.Generator(device='cpu').manual_seed(seed)
        hidden=torch.randn((2,48,9,30,52),generator=g,dtype=torch.float32).to('cuda',torch.bfloat16)
        text=torch.randn((2,512,4096),generator=g,dtype=torch.float32).to('cuda',torch.bfloat16)
        timestep=torch.full((2,9),999,device='cuda',dtype=torch.int64)
        timestep[:,0]=0
        args={'hidden_states':hidden,'timestep':timestep,'encoder_hidden_states':text}
        def run(module):
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record(); out=module(**args); end.record(); torch.cuda.synchronize()
            return out,start.elapsed_time(end)
        # Warm both kernels before timing. Alternate order in timed samples.
        run(base); run(fp8)
        base_ms=[]; fp8_ms=[]; out_base=None; out_fp8=None
        for i in range(6):
            if i%2:
                out_fp8,ms=run(fp8); fp8_ms.append(ms)
                out_base,ms=run(base); base_ms.append(ms)
            else:
                out_base,ms=run(base); base_ms.append(ms)
                out_fp8,ms=run(fp8); fp8_ms.append(ms)
        delta=(out_fp8.float()-out_base.float())
        row={'seed':seed,'shape':list(out_base.shape),
             'bf16_ms_median':statistics.median(base_ms),'fp8_ms_median':statistics.median(fp8_ms),
             'max_abs':delta.abs().max().item(),'mean_abs':delta.abs().mean().item(),
             'relative_l2':(delta.norm()/out_base.float().norm()).item(),
             'fp8_finite':bool(torch.isfinite(out_fp8).all())}
        report['seeds'].append(row)
        print(json.dumps(row),flush=True)
        del hidden,text,out_base,out_fp8,delta
report['max_allocated_bytes']=torch.cuda.max_memory_allocated()
print('REPORT '+json.dumps(report),flush=True)
