# Synthetic attention resource probes

These scripts reproduce the H200 wrapper timings reported on #1491, #1492 and #1517. They run synthetic BF16 Q/K/V through the pinned M* resource code. They do not load a Praxis checkpoint, capture the whole engine or measure served latency. The measured environment used one H200, PyTorch 2.12.1+cu132 and the Triton Flex backend. Each script verifies the exact source archive and resource file hashes before importing M*.

Run from a checkout with the recorded commits and a Python environment with that PyTorch build and M*'s dependencies. Each `git archive` command below regenerates the SHA checked by its script:

```bash
git archive --format=tar.gz --output=pr1491.tar.gz 5ef239f4b6ef1f4c1557203758d96d8c423631c1
git archive --format=tar.gz --output=pr1492.tar.gz 9c55a0e2691689d15e9611baa8fcda7cd73e3af0
git archive --format=tar.gz --output=pr1517.tar.gz 52c336af9f19cee9c3a2cfebfe2c45078a6ec7af

python attention-perf-pr1491.py --source-archive pr1491.tar.gz --text-tokens 6 --blank-tokens 1 --text-source 'Qwen3.5-2B AutoProcessor on Pick up the red cube. / blank, same padding and mask as #1520' --samples 20 --warmups 5 --output pr1491.json
python attention-perf-pr1492.py --source-archive pr1492.tar.gz --text-tokens 6 --blank-tokens 1 --text-source 'Qwen3.5-2B AutoProcessor on Pick up the red cube. / blank, same padding and mask as #1520' --samples 20 --warmups 5 --output pr1492.json
python attention-perf-pr1517.py --source-archive pr1517.tar.gz --text-tokens 6 --blank-tokens 1 --text-source 'Qwen3.5-2B AutoProcessor on Pick up the red cube. / blank, same padding and mask as #1520' --samples 20 --warmups 5 --output pr1517.json
```

The #1517 run uses a corrected dense-oracle constructor in the disposable benchmark. Its captured expert-cross case uses one 3,584-token ceiling for both Q and KV, padding a warm Q=30 to 3,584. Eager wrapper and captured replay therefore do different work. The chart and its summarized values are in `../attention-1517-52c336/`.
