#!/usr/bin/env bash
# P0 throughput benchmark (TRAINING_DESIGN_V3 section 4.4) on BadmintonLab-Game-Serve.
# Runs 2048 / 4096 / 8192 courts and two concurrent 2048-court processes, 60 PPO
# iterations each, while sampling GPU memory/utilisation and cgroup memory.
# Summarise afterwards with scripts/tools/summarize_bench.py <out_dir>.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ISAAC_PYTHON:-/root/autodl-tmp/env_isaaclab/bin/python}"
OUT="${1:-/root/autodl-tmp/logs/p0/bench}"
ITERS="${ITERS:-60}"
CONFIGS="${CONFIGS:-2048 4096 8192 dual2048}"
mkdir -p "$OUT"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$REPO"

sample() {  # $1 = output csv; runs until killed
  echo "t,gpu_mem_mib,gpu_util,cgroup_mem_bytes" > "$1"
  while true; do
    g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    m=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    echo "$(date +%s),$g,$m" >> "$1"
    sleep 2
  done
}

train() {  # $1 = num_envs, $2 = run name, $3 = seed
  "$PY" scripts/rsl_rl/train.py --task BadmintonLab-Game-Serve --headless \
    --num_envs "$1" --max_iterations "$ITERS" --seed "$3" --run_name "$2" --swanlab_mode disabled \
    > "$OUT/$2.log" 2>&1
}

for cfg in $CONFIGS; do
  echo "[bench] $cfg start $(date +%T)"
  sample "$OUT/$cfg.mem.csv" & SAMPLER=$!
  start=$(date +%s)
  if [ "$cfg" = "dual2048" ]; then
    train 2048 bench_dual2048_a 1 & A=$!
    train 2048 bench_dual2048_b 2 & B=$!
    wait $A; ea=$?; wait $B; eb=$?
    echo "$cfg exit=$ea,$eb wall=$(( $(date +%s) - start ))" >> "$OUT/status.txt"
  else
    train "$cfg" "bench_$cfg" 1
    echo "$cfg exit=$? wall=$(( $(date +%s) - start ))" >> "$OUT/status.txt"
  fi
  kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
  echo "[bench] $cfg done $(date +%T)"
  sleep 5
done
echo "[bench] all done"
