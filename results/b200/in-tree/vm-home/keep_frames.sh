#!/bin/bash
Q=~/mstar-pr/benchmark/wan_vae_decoder/runs/b200/ablation/quality
for vae in wan21 wan22; do
  until [ -f $Q/$vae/mstar_warm.run2.meta.json ]; do sleep 5; done
  chmod a-w $Q/$vae && echo "$(date +%T) $vae frames protected"
done
