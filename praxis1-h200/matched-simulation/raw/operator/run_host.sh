#!/usr/bin/env bash
# Operator entrypoint. Do not run before independent package review.
set -euo pipefail
ROOT=/tmp/mstar-lift-matched-operator
INFRA=/home/aruneli/work/infra
TIMER=/tmp/mstar-praxis-tower-diag/arm_vm_hard_stop.sh
SSH=$INFRA/ssh/praxis-h200.sh
VM_ROOT=/home/bench/praxis/mstar-lift-matched-operator
VM_OUT=/home/bench/praxis/logs/lift-matched-rerun
cd "$ROOT"
sha256sum -c frozen.sha256
python3 "$ROOT/verify_archive.py" /home/aruneli/oss/mstar "$ROOT/mstar-source.tar.gz"
[[ ! -e "$ROOT/results" ]] || { echo 'local results directory already exists; refusing to mix runs' >&2; exit 1; }
state=$($INFRA/gpu-status.sh | awk '$1 == "praxis-h200" {print $2}')
[[ "$state" == STOPPED ]] || { echo "praxis-h200 must be STOPPED, got $state" >&2; exit 1; }

timer_output=$($TIMER)
echo "$timer_output"
timer=$(sed -n 's/^timer=\([^ ]*\) DIAG_DEADLINE_EPOCH=.*/\1/p' <<< "$timer_output")
[[ "$timer" == *.timer ]] || { echo '50-minute hard-stop timer missing' >&2; exit 1; }
deadline=$(cat /tmp/mstar-praxis-tower-diag/vm-deadline-epoch.txt)
[[ "$deadline" =~ ^[0-9]+$ ]] || { echo 'invalid hard-stop deadline' >&2; exit 1; }
scene0=${RUN_SCENE0:-0}
scene1=${RUN_SCENE1:-0}
[[ "$scene0" =~ ^[01]$ && "$scene1" =~ ^[01]$ ]] || { echo 'RUN_SCENE0 and RUN_SCENE1 must be 0 or 1' >&2; exit 1; }
started=0
cleanup() {
  set +e
  if [[ "$started" == 1 ]]; then
    mkdir -p "$ROOT/results"
    "$SSH" "test -d $VM_OUT && tar -C /home/bench/praxis/logs -cf - lift-matched-rerun" \
      | tar -C "$ROOT/results" -xf - || true
    "$INFRA/gpu-down.sh" praxis-h200 --force || true
  fi
  state=$($INFRA/gpu-status.sh | awk '$1 == "praxis-h200" {print $2}')
  if [[ "$state" == STOPPED ]]; then
    systemctl --user stop "$timer" || true
    echo 'praxis-h200 independently verified STOPPED'
  else
    echo "praxis-h200 is $state; hard-stop timer remains armed" >&2
    trap - EXIT INT TERM
    exit 1
  fi
}
trap cleanup EXIT INT TERM
started=1
"$INFRA/gpu-up.sh" h200 praxis-h200
[[ -x "$SSH" ]] || { echo 'SSH helper missing after gpu-up' >&2; exit 1; }
tar -C /tmp -cf - mstar-lift-matched-operator \
  | "$SSH" 'mkdir -p /home/bench/praxis && tar -C /home/bench/praxis -xf -'
"$SSH" "DIAG_DEADLINE_EPOCH=$deadline RUN_SCENE0=$scene0 RUN_SCENE1=$scene1 timeout --signal=TERM --kill-after=30s 2700s $VM_ROOT/run_vm.sh"
