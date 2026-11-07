#!/usr/bin/env bash
# P5 V2 pilots (P5_RECEIVE_DESIGN_V2 6): M (multi-critic) then S (single critic, same rewards and stages, the
# groups summed), serial; each with its own monitor (serve hold every 250, full play + det eval every 500).
# Budget in iterations (ITERS, default 1800: the 6.1 checks are at 600 / 1200 / 1800 and M and S are compared at
# equal iterations); HOURS is only a safety cap (P5 runs at ~8-11 s per iteration here, not the doc's ~6 s).
#   nohup bash scripts/tools/p5v2_queue.sh > /root/autodl-tmp/logs/v3l/p5/p5v2_queue.log 2>&1 &
# RUNS="S" restricts the queue.
set -u
REPO=/root/autodl-tmp/BadmintonLab-Game-V3L
PY=/root/autodl-tmp/env_isaaclab/bin/python
SWPY=/root/autodl-tmp/env_swanlab/bin/python
OUT=/root/autodl-tmp/logs/v3l/p5
RUNS=${RUNS:-"M S"}
ITERS=${ITERS:-1800}
HOURS=${HOURS:-5.5}
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$REPO/source/badmintonlab_game"  # the installed badmintonlab_game is another checkout
for R in $RUNS; do
  name="p5v2_$R"
  critic=multi
  [ "$R" = S ] && critic=single
  done_file="$OUT/DONE_$name"
  rm -f "$done_file"
  echo "[$(date '+%F %T')] start $name (critic $critic, $ITERS iterations, cap ${HOURS} h, commit $(git rev-parse --short HEAD))"
  nohup "$SWPY" scripts/tools/p5_monitor.py --run_name "$name" --done_file "$done_file" > "$OUT/monitor_$name.log" 2>&1 &
  timeout 7h "$PY" scripts/rsl_rl/train_p5.py --headless --num_envs 2048 --seed 1 \
    --run_name "$name" --critic "$critic" --max_iterations "$ITERS" --time_budget_hours "$HOURS" \
    --swanlab_group v3l-p5v2 > "$OUT/train_$name.log" 2>&1
  code=$?
  echo "exit $code" > "$done_file"
  echo "[$(date '+%F %T')] $name finished (exit $code)"
  if [ -f "$OUT/STOP_REQUESTED_$name" ]; then
    echo "[$(date '+%F %T')] $name was stopped by its monitor: $(cat "$OUT/STOP_REQUESTED_$name")"
  fi
done
echo "[$(date '+%F %T')] queue finished"
