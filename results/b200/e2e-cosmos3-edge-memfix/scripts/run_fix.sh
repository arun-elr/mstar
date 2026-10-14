#!/bin/bash
# The fix re-run, one server at a time: after1fix (timed phases, then a fresh
# launch for concurrency 8), then the same for after0fix. Each run_all.sh call
# waits until nothing else is on the GPU.
set -u
E=$HOME/e2e-cosmos3
RUNS=runs-fix bash "$E/scripts/run_all.sh" after1fix || exit 1
RUNS=runs-c8-fix PHASES=c8 bash "$E/scripts/run_all.sh" after1fix || exit 1
RUNS=runs-fix bash "$E/scripts/run_all.sh" after0fix || exit 1
RUNS=runs-c8-fix PHASES=c8 bash "$E/scripts/run_all.sh" after0fix || exit 1
echo FIX_ALL_DONE
