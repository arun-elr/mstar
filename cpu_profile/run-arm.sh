#!/usr/bin/env bash
set -euo pipefail

arm=$1
source_arm=$2
case "$source_arm" in main|pr) ;; *) echo "source_arm must be main or pr" >&2; exit 2;; esac
export PATH="/home/bench/praxis/mstar-venv/bin:$PATH"
export PYTHONPATH="/dev/shm/mstar-pr582-${source_arm}-pylib:/dev/shm/mstar-pr582-${source_arm}-src"
head=$(cat "/dev/shm/mstar-pr582-${source_arm}-src/MEASURED_HEAD")
out=/dev/shm/pr582-cpu-results
mkdir -p "$out"
: > "$out/${arm}-windows.tsv"

for item in 16:2560 64:5120; do
    c=${item%%:*}
    n=${item##*:}
    t0=$(date -u +%s.%N)
    echo "START $arm c=$c n=$n at $(date -u +%FT%TZ)"
    timeout 900 /home/bench/praxis/mstar-venv/bin/python /dev/shm/mstar-pr582-bench.py \
        --arm "$arm" --head "$head" --output "$out/${arm}-c${c}.json" \
        --prompts /dev/shm/mstar-pr582-prompts.txt --concurrency "$c" --requests "$n" \
        > "$out/${arm}-c${c}.log" 2>&1 || {
        tail -40 "$out/${arm}-c${c}.log" >&2
        exit 1
    }
    t1=$(date -u +%s.%N)
    printf '%s\t%s\t%s\n' "$c" "$t0" "$t1" >> "$out/${arm}-windows.tsv"
    /home/bench/praxis/mstar-venv/bin/python - "$out/${arm}-c${c}.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print({k: r[k] for k in ("concurrency", "completed", "tokens_total", "matched_exactly_64_tokens", "request_throughput")})
PY
done
echo "DONE $arm at $(date -u +%FT%TZ)"
