#!/bin/bash
# Full-length runs (10 warm-up + 75 timed chunks x 2 repeats), one GPU job at a time.
cd ~/w22/h
one() {  # one TAG M PY VARIANT VAE [extra args]
  local tag=$1 m=$2 py=$3 v=$4 vae=$5; shift 5
  M=$m PY=$py ./dock.sh run.py --variant $v --vae $vae --latents /vo/inputs/$vae/latents.pt \
    --ref /vo/inputs/$vae/ref_fp32_outputs.pt --out /w22/runs/verify/$tag/$vae "$@" > $HOME/w22/runs/verify/$tag.$v.$vae.log 2>&1 \
    || echo "FAILED $tag $v $vae" >> ~/w22/runs/verify/failures.txt
  tail -1 ~/w22/runs/verify/$tag.$v.$vae.log | python3 -c "import json,sys
try:
  d=json.loads(sys.stdin.read()); print(\"$tag\", \"$vae\", d[\"variant\"], d[\"median_ms\"], d[\"p90_ms\"], \"u8\", d[\"psnr_u8_db_vs_fp32\"], \"f1\", d[\"psnr_float_db_vs_fp32_peak1\"])
except Exception as e: print(\"$tag $vae $v parse-fail\")" >> ~/w22/runs/verify/summary.txt
}
mkdir -p ~/w22/runs/verify
FIX=/w22/fix/mstar; PR=/w22/mstar; P212=/w22/venv212/bin/python
for vae in wan22 wan21; do
  for v in mstar_warm px_nopad_bench; do
    one t213_fix $FIX python $v $vae
    one t213_pr  $PR  python $v $vae
  done
done
one t213_fix $FIX python mstar_warm_eager wan22
for vae in wan22 wan21; do
  one t212_fix $FIX $P212 mstar_warm $vae
  one t212_pr  $PR  $P212 mstar_warm $vae
done
for t in "t213_pr $PR python" "t213_fix $FIX python"; do set -- $t
  one $1 $2 $3 mstar_ar_ctx8 wan22 --chunks 10 --repeats 1
done
echo DONE >> ~/w22/runs/verify/summary.txt
