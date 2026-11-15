# Synthetic attention resource probes

These scripts reproduce the H200 wrapper timings reported on #1491, #1492 and #1517. They run synthetic BF16 Q/K/V through the pinned M* resource code. They do not load a Praxis checkpoint, capture the whole engine or measure served latency. The measured environment used one H200, PyTorch 2.12.1+cu132 and the Triton Flex backend. Each script verifies the exact source archive and resource file hashes before importing M*.

Run from a checkout with the recorded commits and a Python environment with that PyTorch build and M*'s dependencies. These `git archive` commands regenerated the checked SHA values with Git 2.43.0 on the measured machine. A different Git or gzip build may emit different compressed bytes; the scripts fail closed in that case.

```bash
git archive --format=tar.gz --output=pr1491.tar.gz 5ef239f4b6ef1f4c1557203758d96d8c423631c1
git archive --format=tar.gz --output=pr1492.tar.gz 9c55a0e2691689d15e9611baa8fcda7cd73e3af0
git archive --format=tar.gz --output=pr1517.tar.gz 52c336af9f19cee9c3a2cfebfe2c45078a6ec7af

python attention-perf-pr1491.py --source-archive pr1491.tar.gz --text-tokens 6 --blank-tokens 1 --text-source 'Qwen3.5-2B AutoProcessor on Pick up the red cube. / blank, same padding and mask as #1520' --samples 20 --warmups 5 --output pr1491.json
python attention-perf-pr1492.py --source-archive pr1492.tar.gz --text-tokens 6 --blank-tokens 1 --text-source 'Qwen3.5-2B AutoProcessor on Pick up the red cube. / blank, same padding and mask as #1520' --samples 20 --warmups 5 --output pr1492.json
python attention-perf-pr1517.py --source-archive pr1517.tar.gz --text-tokens 6 --blank-tokens 1 --text-source 'Qwen3.5-2B AutoProcessor on Pick up the red cube. / blank, same padding and mask as #1520' --samples 20 --warmups 5 --output pr1517.json
```

The #1517 run uses a corrected dense-oracle constructor in the disposable benchmark. Its captured expert-cross case uses one 3,584-token ceiling for both Q and KV, padding a warm Q=30 to 3,584. Eager wrapper and captured replay therefore do different work. The chart and its summarized values are in `../attention-1517-52c336/`.

The final #1517 resource-sharing head was measured separately with the same synthetic recipe. Its pinned archive SHA is `d9cee2fd77c5263e1d4d59fa51ddc3c9bc0788e4e1220f8cc3833fc64109c5b5`:

```bash
git archive --format=tar.gz --output=pr1517-final.tar.gz 3c904e6ab6ae72af9762f3338f6b416b6da15a47
python attention-perf-pr1517-final.py --source-archive pr1517-final.tar.gz --text-tokens 6 --blank-tokens 1 --text-source 'Qwen3.5-2B AutoProcessor on Pick up the red cube. / blank, same padding and mask as #1520' --samples 20 --warmups 5 --output pr1517-final.json
```

The final chart and summarized values are in `../attention-1517-3c904/`. The script measures resource wrappers; the distinct-label `build_specs` regression is covered by the PR's CPU suite, not by this synthetic timing.
