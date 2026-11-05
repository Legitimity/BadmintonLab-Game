#!/usr/bin/env bash
# P0 throughput benchmark (design doc 4.4). Run from the repo root on the remote machine.
# Each setting runs 60 iterations; read steady-state numbers from iterations 20-60 in the logs.
set -euo pipefail
TASK=${TASK:-BadmintonLab-Game-Serve}
LOG=${LOG:-/root/autodl-tmp/logs/bench}
mkdir -p "$LOG"
run() {  # name num_envs
  python scripts/rsl_rl/train_serve.py --headless --task "$TASK" --num_envs "$2" --max_iterations 60 \
    --seed 1 --run_name "bench_$1" > "$LOG/$1.log" 2>&1
}
for n in 2048 4096 8192; do
  run "single_$n" "$n"
  grep -E "Computation|collection|learning|steps/s" "$LOG/single_$n.log" | tail -3 || true
done
# Two processes at once: total throughput decides whether to train two seeds in parallel.
run dual_a 2048 & run dual_b 2048 & wait
grep -h -E "PhysX|overflow|capacity" "$LOG"/*.log | sort | uniq -c || true
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
free -g; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true
