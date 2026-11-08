#!/usr/bin/env bash
# P3_FINETUNE_DT1MS (revised 2026-11-08 11:45): no F1 (p3_final already serves 1.000 in play at 1 ms), F2 only.
#   p3q_short / p3q_deep run in parallel from p3_c7 model_3000 for $F2_HOURS at $TRAIN_DT (2 ms), then $TAIL_ITERS
#   at 1 ms from the branch's selected checkpoint. Stage C fixed, v_close_max 12, entropy 0, std in [0.15, 0.35].
#   Monitor per run: play eval (1 ms, zero opponent, both sides) every 250 (tail: 100) iterations; branch stop when
#   play in < REF_IN - 0.04 twice in a row; selection: in >= SELECT_FLOOR, fault <= 1 %, fell_after_hit <= 2 %, then
#   the largest landing-to-receiver-racket distance (within 0.2 m: the higher in).
# Status: $OUT/QUEUE_STATUS.txt; per run: $OUT/<run>/STATUS.txt and state.json.
set -u
REPO=${REPO:-/root/autodl-tmp/BadmintonLab-Game-V3L}
PY=${PY:-/root/autodl-tmp/env_isaaclab/bin/python}
SWPY=${SWPY:-/root/autodl-tmp/env_swanlab/bin/python}
OUT=${OUT:-/root/autodl-tmp/logs/v3l/dt1ms}
LOGROOT="$REPO/logs/rsl_rl/badmintonlab_serve"
START_CKPT=${START_CKPT:-$LOGROOT/2026-11-06_22-49-04_p3_c7/model_3000.pt}
START_IT=${START_IT:-3000}
REF_IN=${REF_IN:-1.0}               # E0: p3_final play in at 1 ms (both sides 1.000)
SELECT_FLOOR=${SELECT_FLOOR:-0.97}
TRAIN_DT=${TRAIN_DT:-0.002}
F2_HOURS=${F2_HOURS:-4.0}
TAIL_ITERS=${TAIL_ITERS:-200}
TAIL_HOURS=${TAIL_HOURS:-2.0}
NET_MARGIN=${NET_MARGIN:-0.12}      # E0: net 0.000 in play and in the training env (det) -> 0.12 allowed
BRANCHES=${BRANCHES:-"p3q_short:1.0 p3q_deep:5.6"}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
STATUS="$OUT/QUEUE_STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
read -r -a EXTRA_A <<< "$EXTRA"
COMMON=(--headless --num_envs 2048 --mini_batches 16 --seed 1 --stage C --no_curriculum --entropy_coef 0
        --std_range 0.15,0.35 --swanlab_group v3l-p3dt "${EXTRA_A[@]}")
Q=(env.serve_reward.v_close_max=12.0 env.serve_reward.x_sigma=0.8 env.serve_reward.y_target=1.7
   env.serve_reward.y_sigma=0.5 env.serve_reward.y_safe=2.25 env.serve_reward.net_margin=$NET_MARGIN
   --stage_weights C.flight_time=1.0)

train() {  # run_name dt hours max_iters checkpoint log [extra args...]; returns the trainer's exit code
  local name=$1 dt=$2 hours=$3 iters=$4 ck=$5 log=$6 code
  shift 6
  say "$name: launch dt=$dt hours=$hours iters=$iters from $ck"
  timeout "$(python3 -c "print(int($hours*3600+1800))")" "$PY" scripts/rsl_rl/train_serve.py "${COMMON[@]}" \
    --physics_dt "$dt" --resume --checkpoint "$ck" --max_iterations "$iters" --time_budget_hours "$hours" \
    --run_name "$name" "$@" >> "$log" 2>&1
  code=$?
  say "$name: exit $code $(grep -o 'stopped: .*' "$log" | tail -1)"
  return $code
}

monitor() {  # run_name start_iter eval_every; prints the pid
  local name=$1
  mkdir -p "$OUT/$name"
  rm -f "$OUT/$name/DONE"
  "$SWPY" scripts/tools/dt1ms_monitor.py --run_name "$name" --out "$OUT/$name" --done_file "$OUT/$name/DONE" \
    --mode f2 --start_iter "$2" --eval_every "$3" --ref "$REF_IN" --select_floor "$SELECT_FLOOR" \
    > "$OUT/$name/monitor.log" 2>&1 &
  echo $!
}

waitpid() { while kill -0 "$1" 2>/dev/null; do sleep 15; done; }

field() {  # run_name key(best|select) field
  python3 -c "import json; b=json.load(open('$OUT/$1/state.json')).get('$2'); print(b['$3'] if b else '')" 2>/dev/null
}

branch() {  # name x_target
  local name=$1 xt=$2 m ck it
  m=$(monitor "$name" "$START_IT" 250)
  train "$name" "$TRAIN_DT" "$F2_HOURS" 100000 "$START_CKPT" "$OUT/train_$name.log" "${Q[@]}" \
    env.serve_reward.x_target="$xt"
  touch "$OUT/$name/DONE"
  waitpid "$m"
  ck=$(field "$name" select ckpt)
  [ -z "$ck" ] && ck=$(field "$name" best ckpt)
  if [ -z "$ck" ]; then
    say "$name: no valid play evaluation at $TRAIN_DT -> no 1 ms tail"
    return
  fi
  it=$(python3 -c "import re; print(re.search(r'model_(\d+)\.pt', '$ck').group(1))")
  say "$name: 1 ms tail from $ck (select $(field "$name" select it), best-in $(field "$name" best it))"
  m=$(monitor "${name}_1ms" "$it" 100)
  train "${name}_1ms" 0.001 "$TAIL_HOURS" "$TAIL_ITERS" "$ck" "$OUT/train_${name}_1ms.log" "${Q[@]}" \
    env.serve_reward.x_target="$xt" --save_interval_override 100
  touch "$OUT/${name}_1ms/DONE"
  waitpid "$m"
  say "$name: done; 1 ms select $(field "${name}_1ms" select it) ($(field "${name}_1ms" select in)), best-in $(field "${name}_1ms" best it) ($(field "${name}_1ms" best in))"
}

say "queue start (revised, no F1), HEAD $(git rev-parse --short HEAD), from $START_CKPT, train dt $TRAIN_DT x ${F2_HOURS} h, tail $TAIL_ITERS it at 1 ms, ref $REF_IN, floor $SELECT_FLOOR, net_margin $NET_MARGIN"
PIDS=()
for b in $BRANCHES; do
  branch "${b%%:*}" "${b##*:}" &
  PIDS+=($!)
  sleep 90   # stagger the env start-up
done
wait "${PIDS[@]}"
say "queue done"
