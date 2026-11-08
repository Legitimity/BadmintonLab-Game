#!/usr/bin/env bash
# P3_SERVE_STYLES: one policy per serve style, all from p3_final (p3_c7 model_3000), in parallel at $TRAIN_DT for
# $HOURS. Reward presets in serve/styles.py (--style). Stage C fixed, entropy 0, std in [0.15, 0.35].
# Monitor per run: play eval (1 ms, zero opponent, both sides) every $EVAL_EVERY iterations and at the end; stop when
# play in < 1.0 - $STOP_DROP twice in a row; selection: in >= SELECT_FLOOR, fault <= 1 %, fell_after_hit <= 2 %, then
# the style metric (flat: shortest median flight time; landing styles: median landing closest to the target).
# Status: $OUT/QUEUE_STATUS.txt; per run: $OUT/<run>/STATUS.txt and state.json.
set -u
REPO=${REPO:-/root/autodl-tmp/BadmintonLab-Game-V3L}
PY=${PY:-/root/autodl-tmp/env_isaaclab/bin/python}
SWPY=${SWPY:-/root/autodl-tmp/env_swanlab/bin/python}
OUT=${OUT:-/root/autodl-tmp/logs/v3l/styles}
LOGROOT="$REPO/logs/rsl_rl/badmintonlab_serve"
START_CKPT=${START_CKPT:-$LOGROOT/2026-11-06_22-49-04_p3_c7/model_3000.pt}
START_IT=${START_IT:-3000}
SELECT_FLOOR=${SELECT_FLOOR:-0.97}
STOP_DROP=${STOP_DROP:-0.10}
TRAIN_DT=${TRAIN_DT:-0.002}
HOURS=${HOURS:-3.9}
NUM_ENVS=${NUM_ENVS:-1024}
MINI_BATCHES=${MINI_BATCHES:-8}
EVAL_EVERY=${EVAL_EVERY:-200}
STYLES=${STYLES:-"s1_flat s2_wide_fh s3_wide_bh s4_short"}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
STATUS="$OUT/QUEUE_STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
read -r -a EXTRA_A <<< "$EXTRA"

waitpid() { while kill -0 "$1" 2>/dev/null; do sleep 15; done; }

field() {  # run_name key(best|select) field
  python3 -c "import json; b=json.load(open('$OUT/$1/state.json')).get('$2'); print(b['$3'] if b else '')" 2>/dev/null
}

run_style() {  # style
  local style=$1 name="st_$1" m code log
  log="$OUT/train_$name.log"
  mkdir -p "$OUT/$name"
  rm -f "$OUT/$name/DONE"
  "$SWPY" scripts/tools/dt1ms_monitor.py --run_name "$name" --out "$OUT/$name" --done_file "$OUT/$name/DONE" \
    --mode f2 --start_iter "$START_IT" --eval_every "$EVAL_EVERY" --ref 1.0 --stop_drop "$STOP_DROP" \
    --select_floor "$SELECT_FLOOR" --style "$style" > "$OUT/$name/monitor.log" 2>&1 &
  m=$!
  say "$name: launch dt=$TRAIN_DT hours=$HOURS envs=$NUM_ENVS from $START_CKPT"
  timeout "$(python3 -c "print(int($HOURS*3600+1800))")" "$PY" scripts/rsl_rl/train_serve.py --headless \
    --num_envs "$NUM_ENVS" --mini_batches "$MINI_BATCHES" --seed 1 --stage C --no_curriculum --entropy_coef 0 \
    --std_range 0.15,0.35 --swanlab_group v3l-p3styles --physics_dt "$TRAIN_DT" --resume --checkpoint "$START_CKPT" \
    --max_iterations 100000 --time_budget_hours "$HOURS" --save_interval_override 100 --run_name "$name" \
    --style "$style" "${EXTRA_A[@]}" >> "$log" 2>&1
  code=$?
  say "$name: exit $code $(grep -o 'stopped: .*' "$log" | tail -1)"
  touch "$OUT/$name/DONE"
  waitpid "$m"
  say "$name: done; select $(field "$name" select it) (in $(field "$name" select in)), best-in $(field "$name" best it) ($(field "$name" best in))"
}

say "styles queue start, HEAD $(git rev-parse --short HEAD), styles [$STYLES], dt $TRAIN_DT x ${HOURS} h, envs $NUM_ENVS, eval every $EVAL_EVERY, floor $SELECT_FLOOR, stop_drop $STOP_DROP"
PIDS=()
for s in $STYLES; do
  run_style "$s" &
  PIDS+=($!)
  sleep 60   # stagger the env start-up
done
wait "${PIDS[@]}"
say "styles queue done"
