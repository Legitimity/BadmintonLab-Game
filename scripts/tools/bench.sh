#!/usr/bin/env bash
# P0 throughput benchmark (design doc 4.4). Run from the repo root on the remote machine.
# Each setting runs 60 iterations; read steady-state numbers from iterations 20-60 in the logs.
# Must own the GPU: do not run anything else concurrently.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/root/autodl-tmp/env_isaaclab/bin/python}
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/source/badmintonlab_game${PYTHONPATH:+:$PYTHONPATH}"
TASK=${TASK:-BadmintonLab-Game-Serve}
LOG=${LOG:-/root/autodl-tmp/logs/bench}
SIZES=${SIZES:-"2048 4096 8192"}
mkdir -p "$LOG"
# GPU memory sampler (peak per run is read from this file afterwards).
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 5 > "$LOG/gpu_mem.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
run() {  # name num_envs
  echo "$1 start=$(date +%T)"
  timeout 1800 "$PY" scripts/rsl_rl/train_serve.py --headless --task "$TASK" --num_envs "$2" --max_iterations 60 \
    --seed 1 --run_name "bench_$1" --swanlab_mode disabled > "$LOG/$1.log" 2>&1
  echo "$1 exit=$? end=$(date +%T)"
}
for n in $SIZES; do
  run "single_$n" "$n"
  grep -E "Computation|collection|learning|steps/s" "$LOG/single_$n.log" | tail -3 || true
done
# Two processes at once: total throughput decides whether to train two seeds in parallel.
# DUAL_SIZES: per-process court counts for the two-process runs (default 2048).
if [ -z "${SKIP_DUAL:-}" ]; then
  for n in ${DUAL_SIZES:-2048}; do
    run "dual_${n}_a" "$n" & run "dual_${n}_b" "$n" & wait
  done
fi
grep -h -E "PhysX|overflow|capacity" "$LOG"/*.log | sort | uniq -c || true
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
free -g; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true
