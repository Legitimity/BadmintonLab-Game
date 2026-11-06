#!/usr/bin/env bash
# P3 queue (P3_PLAN_V2 4.4-4.7, P3_SPEEDUP_PLAN 4.2).
#   S1  E1: groups courts:mini_batches ($E1_GROUPS), stage B fixed weights, 45 min each, sequential.
#       Re-entrant: a group already finished is skipped, one still running is waited for.
#   S2  pick the group with scripts/tools/p3_select_scale.py (wall-clock time to 5-min in >= 0.6)
#   S3  formal training, run $FORMAL_RUN: first launch starts from the winning E1 group's final checkpoint
#       (curriculum on, counted from stage B); crash -> resume its own latest.pt with the remaining budget;
#       STOP_REQUESTED from the monitor -> no restart. 8192 courts: curriculum iteration counts halved.
#   S4  monitor in parallel: play eval every 1000 iterations and at stage C, automatic stops / early stop
# Progress: $OUT/P3_STATUS.txt
set -u
REPO=/root/autodl-tmp/BadmintonLab-Game-V3L
PY=/root/autodl-tmp/env_isaaclab/bin/python
SWPY=/root/autodl-tmp/env_swanlab/bin/python
OUT=/root/autodl-tmp/logs/v3l/p3
LOGROOT="$REPO/logs/rsl_rl/badmintonlab_serve"
FORMAL_HOURS=${FORMAL_HOURS:-18}
FORMAL_RUN=${FORMAL_RUN:-p3_f1}
CMP_HOURS=${CMP_HOURS:-0.75}
STAGE_MAX=${STAGE_MAX:-A=4500,B=12000,C=12000}
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
STATUS="$OUT/P3_STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
say "P3 queue (v3) start, HEAD $(git rev-parse --short HEAD), formal run $FORMAL_RUN, budget ${FORMAL_HOURS} h, stage max $STAGE_MAX"

# ---------------------------------------------------------------- S1 (E1 of P3_SPEEDUP_PLAN 4.2)
# Groups "courts:mini_batches", fixed wall time each, stage B fixed weights, seed 1.
E1_GROUPS=${E1_GROUPS:-"2048:16 8192:64 8192:32 4096:32"}
mem_sampler() {  # file
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 20; done ) >> "$1" 2>/dev/null &
  MEMPID=$!
}
cmp_run() {  # courts mini_batches
  local n=$1 mb=$2 name="p3e1_${1}_mb${2}"
  if grep -q "stopped: " "$OUT/$name.log" 2>/dev/null; then
    say "S1 $name already done: $(grep -o 'stopped: .*' "$OUT/$name.log" | tail -1)"; return
  fi
  if pgrep -f "train_serve.py.*--run_name $name( |$)" > /dev/null; then
    say "S1 $name already running, waiting"
    mem_sampler "$OUT/$name.mem"
    while pgrep -f "train_serve.py.*--run_name $name( |$)" > /dev/null; do sleep 30; done
    kill "$MEMPID" 2>/dev/null
    say "S1 $name finished $(grep -o 'stopped: .*' "$OUT/$name.log" | tail -1)"
    return
  fi
  say "S1 $name start"
  mem_sampler "$OUT/$name.mem"
  timeout 4500 "$PY" scripts/rsl_rl/train_serve.py --headless --num_envs "$n" --mini_batches "$mb" --seed 1 --stage B \
    --no_curriculum --run_name "$name" --time_budget_hours "$CMP_HOURS" --swanlab_group v3l-p3e1 \
    > "$OUT/$name.log" 2>&1
  local code=$?
  kill "$MEMPID" 2>/dev/null
  say "S1 $name exit=$code $(grep -o 'stopped: .*' "$OUT/$name.log" | tail -1)"
}
if [ -z "${SKIP_CMP:-}" ]; then
  for g in $E1_GROUPS; do cmp_run "${g%%:*}" "${g##*:}"; done
fi

# ---------------------------------------------------------------- S2: selection (plan 4.2 rules)
SEL=$("$SWPY" scripts/tools/p3_select_scale.py --repo "$REPO" --groups "$E1_GROUPS" --out "$OUT/e1_summary.json" 2>>"$STATUS")
SCALE=${SEL%%:*}; MBS=${SEL##*:}
if [ -z "$SCALE" ] || [ -z "$MBS" ]; then SCALE=2048; MBS=16; say "S2 selection failed, default 2048:16"; fi
say "S2 selected: $SCALE courts, $MBS mini-batches"

# Start point: the winner's final checkpoint (its curriculum state is empty -> counting starts at stage B).
INIT=$(ls -d "$LOGROOT"/*_p3e1_${SCALE}_mb${MBS} 2>/dev/null | tail -1)/latest.pt
if [ -f "$INIT" ]; then RESUME=(--resume --checkpoint "$INIT"); say "S3 start point: $INIT"
else RESUME=(); say "S3 start point: none found ($INIT), training from the pose init"; fi
# Curriculum counts iterations; an 8192 iteration holds 4x the samples of 2048 -> halve (EMA / blend / stage minimum).
CUR=(); AFTER_C=1000
if [ "$SCALE" -ge 8192 ]; then CUR=(--curriculum_iters ema=100,blend=150,min=150); AFTER_C=500; fi
say "S3 curriculum overrides: ${CUR[*]:-none}; early stop $AFTER_C iterations after stage C or M1"

# ---------------------------------------------------------------- S3 + S4
DONE="$OUT/FORMAL_DONE"
rm -f "$DONE" "$OUT/STOP_REQUESTED" "$OUT/monitor_state.json"
"$SWPY" scripts/tools/p3_monitor.py --run_name "$FORMAL_RUN" --out "$OUT" --done_file "$DONE" --after_c_iters "$AFTER_C" \
  > "$OUT/monitor.log" 2>&1 &
MON=$!
DEADLINE=$(( $(date +%s) + $(python3 -c "print(int($FORMAL_HOURS*3600))") ))
say "S3 formal training: deadline $(date -d @$DEADLINE '+%F %T')"
tries=0
while :; do
  left=$(( DEADLINE - $(date +%s) ))
  if [ "$left" -lt 900 ]; then say "S3 budget exhausted"; break; fi
  hours=$(python3 -c "print(round($left/3600, 3))")
  start_line=$(( $(wc -l < "$OUT/formal.log" 2>/dev/null || echo 0) + 1 ))
  say "S3 launch (attempt $((tries+1)), budget ${hours} h)"
  # --auto_resume takes $FORMAL_RUN's own latest.pt when it exists (after a crash), else the E1 start point.
  timeout $(( left + 1800 )) "$PY" scripts/rsl_rl/train_serve.py --headless --num_envs "$SCALE" --mini_batches "$MBS" \
    --seed 1 --stage B --stage_max_iters "$STAGE_MAX" "${CUR[@]}" "${RESUME[@]}" --auto_resume \
    --time_budget_hours "$hours" --swanlab_group v3l-p3 --run_name "$FORMAL_RUN" >> "$OUT/formal.log" 2>&1
  code=$?
  reason=$(tail -n +"$start_line" "$OUT/formal.log" | grep -o 'stopped: .*' | tail -1)
  say "S3 exit=$code $reason"
  if [ -f "$OUT/STOP_REQUESTED" ]; then say "S3 stopped by monitor: $(cat "$OUT/STOP_REQUESTED")"; break; fi
  case "$reason" in
    *time_budget*|*max_iterations*|*nonfinite*|*stage_*_max_iters*) break ;;
  esac
  tries=$((tries+1))
  if [ "$tries" -ge 3 ]; then say "S3 gave up after 3 crashes"; break; fi
  sleep 30
done
touch "$DONE"
say "S3 done; waiting for the monitor to evaluate the final checkpoint"
wait "$MON"
git -C /root/autodl-tmp/BadmintonLab-Game-Play status --porcelain | head -3 | sed 's/^/  play dirty: /' >> "$STATUS"
say "P3 queue finished"
