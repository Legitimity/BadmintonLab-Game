#!/usr/bin/env bash
# P3_FINETUNE_DT1MS F1 -> F2 queue (training at $TRAIN_DT, the 1 ms tail on the F2 branches).
#   F1  p3dt_f1: from p3_c7 model_3000, stage C fixed, v_close_max 12, entropy 0, std in [0.15, 0.35], $F1_HOURS.
#       Monitor (mode f1, base = E0 play in) evaluates every 250 iterations; p3_dt_best = best valid evaluation
#       (fault <= 1 %, fell_after_hit <= 2 %), else p3_final (iteration 3000).
#   F2  p3q_short / p3q_deep in parallel from p3_dt_best, $F2_HOURS at $TRAIN_DT, then $TAIL_ITERS at 1 ms from the
#       branch's best checkpoint (monitor mode f2, ref = p3_dt_best's play in).
# Status: $OUT/QUEUE_STATUS.txt; per run: $OUT/<run>/STATUS.txt and state.json.
set -u
REPO=${REPO:-/root/autodl-tmp/BadmintonLab-Game-V3L}
PY=${PY:-/root/autodl-tmp/env_isaaclab/bin/python}
SWPY=${SWPY:-/root/autodl-tmp/env_swanlab/bin/python}
OUT=${OUT:-/root/autodl-tmp/logs/v3l/dt1ms}
LOGROOT="$REPO/logs/rsl_rl/badmintonlab_serve"
CK0=${CK0:-$LOGROOT/2026-11-06_22-49-04_p3_c7/model_3000.pt}
TRAIN_DT=${TRAIN_DT:-0.002}
F1_HOURS=${F1_HOURS:-2.25}
F2_HOURS=${F2_HOURS:-1.75}
TAIL_ITERS=${TAIL_ITERS:-300}
BASE_IN=${BASE_IN:?E0 play in of p3_final}
NET_MARGIN=${NET_MARGIN:-0.20}     # 0.12 only if the E0 consistency check showed net < 1 %
SKIP_F1=${SKIP_F1:-}                # testing / restart: use DT_BEST_CKPT and DT_BEST_IN directly
EXTRA=${EXTRA:-}
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
STATUS="$OUT/QUEUE_STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
read -r -a EXTRA_A <<< "$EXTRA"
COMMON=(--headless --num_envs 2048 --mini_batches 16 --seed 1 --stage C --no_curriculum --entropy_coef 0
        --std_range 0.15,0.35 --swanlab_group v3l-p3dt "${EXTRA_A[@]}")

train() {  # run_name dt hours max_iters checkpoint log [hydra/extra args...]
  local name=$1 dt=$2 hours=$3 iters=$4 ck=$5 log=$6
  shift 6
  say "$name: launch dt=$dt hours=$hours iters=$iters from $ck"
  timeout $(python3 -c "print(int($hours*3600+1800))") "$PY" scripts/rsl_rl/train_serve.py "${COMMON[@]}" --physics_dt "$dt" \
    --resume --checkpoint "$ck" --max_iterations "$iters" --time_budget_hours "$hours" --run_name "$name" "$@" >> "$log" 2>&1
  say "$name: exit $? $(grep -o 'stopped: .*' "$log" | tail -1)"
}

monitor() {  # run_name mode start_iter base ref
  local name=$1
  mkdir -p "$OUT/$name"
  rm -f "$OUT/$name/DONE"
  "$SWPY" scripts/tools/dt1ms_monitor.py --run_name "$name" --out "$OUT/$name" --done_file "$OUT/$name/DONE" \
    --mode "$2" --start_iter "$3" ${4:+--base "$4"} ${5:+--ref "$5"} > "$OUT/$name/monitor.log" 2>&1 &
  echo $!
}

waitpid() {  # monitors are started in a subshell: poll instead of wait
  while kill -0 "$1" 2>/dev/null; do sleep 15; done
}

best_field() {  # run_name field [key: best | select]
  python3 -c "import json,sys; b=json.load(open('$OUT/$1/state.json')).get('${3:-best}'); print(b['$2'] if b else '')" 2>/dev/null
}

say "queue start, HEAD $(git rev-parse --short HEAD), train dt $TRAIN_DT, base $BASE_IN, net_margin $NET_MARGIN"
REW=(env.serve_reward.v_close_max=12.0)

# ---------------------------------------------------------------- F1
if [ -z "$SKIP_F1" ]; then
  M=$(monitor p3dt_f1 f1 3000 "$BASE_IN" "")
  train p3dt_f1 "$TRAIN_DT" "$F1_HOURS" 100000 "$CK0" "$OUT/train_p3dt_f1.log" "${REW[@]}"
  touch "$OUT/p3dt_f1/DONE"
  waitpid "$M"
  DT_BEST_CKPT=$(best_field p3dt_f1 ckpt)
  DT_BEST_IN=$(best_field p3dt_f1 in)
  if [ -z "$DT_BEST_CKPT" ] || python3 -c "import sys; sys.exit(0 if float('$DT_BEST_IN') < float('$BASE_IN') else 1)"; then
    say "F1: no checkpoint beats p3_final in play -> p3_dt_best = p3_final"
    DT_BEST_CKPT=$CK0
    DT_BEST_IN=$BASE_IN
  fi
fi
DT_BEST_CKPT=${DT_BEST_CKPT:?}
DT_BEST_IN=${DT_BEST_IN:?}
echo "$DT_BEST_CKPT $DT_BEST_IN" > "$OUT/P3_DT_BEST.txt"
say "p3_dt_best: $DT_BEST_CKPT (play in $DT_BEST_IN)"
START=$(python3 -c "import re; print(int(re.search(r'model_(\d+)\.pt', '$DT_BEST_CKPT').group(1)))")

# ---------------------------------------------------------------- F2
Q=(env.serve_reward.v_close_max=12.0 env.serve_reward.x_sigma=0.8 env.serve_reward.y_target=1.7
   env.serve_reward.y_sigma=0.5 env.serve_reward.y_safe=2.25 env.serve_reward.net_margin=$NET_MARGIN
   --stage_weights C.flight_time=1.0)
branch() {  # name x_target
  local name=$1 xt=$2
  local m
  m=$(monitor "$name" f2 "$START" "" "$DT_BEST_IN")
  train "$name" "$TRAIN_DT" "$F2_HOURS" 100000 "$DT_BEST_CKPT" "$OUT/train_$name.log" "${Q[@]}" env.serve_reward.x_target="$xt"
  touch "$OUT/$name/DONE"
  waitpid "$m"
  local ck
  ck=$(best_field "$name" ckpt select)
  [ -z "$ck" ] && ck=$(best_field "$name" ckpt)
  if [ -z "$ck" ]; then
    say "$name: no valid evaluation, 1 ms tail from its latest.pt"
    ck=$(ls -d "$LOGROOT"/*_"$name" | tail -1)/latest.pt
  fi
  local s2
  s2=$(python3 -c "import re; print(int(re.search(r'model_(\d+)\.pt', '$ck').group(1)) if 'model_' in '$ck' else 0)")
  # 1 ms tail: its own run name; the monitor evaluates every 100 iterations and the final checkpoint.
  mkdir -p "$OUT/${name}_1ms"
  rm -f "$OUT/${name}_1ms/DONE"
  "$SWPY" scripts/tools/dt1ms_monitor.py --run_name "${name}_1ms" --out "$OUT/${name}_1ms" --done_file "$OUT/${name}_1ms/DONE" \
    --mode none --start_iter "$s2" --eval_every 100 --ref "$DT_BEST_IN" > "$OUT/${name}_1ms/monitor.log" 2>&1 &
  m=$!
  train "${name}_1ms" 0.001 3.0 "$TAIL_ITERS" "$ck" "$OUT/train_${name}_1ms.log" "${Q[@]}" env.serve_reward.x_target="$xt" \
    --save_interval_override 100
  touch "$OUT/${name}_1ms/DONE"
  waitpid "$m"
  say "$name: done; 2 ms select $(best_field "$name" it select) best-in $(best_field "$name" it); 1 ms select $(best_field "${name}_1ms" it select) best-in $(best_field "${name}_1ms" it)"
}
branch p3q_short 1.0 &
B1=$!
sleep 120
branch p3q_deep 5.6 &
B2=$!
wait $B1 $B2
say "queue done"
