#!/usr/bin/env bash
# P3 queue v4 (P3_REWARD_FIX_AND_E1B): reward V2.1 check (E1b), then the formal run p3_f2.
#   S1  E1b: groups label=courts:mini_batches:seed ($E1B_GROUPS), stage B fixed weights, 45 min each, sequential,
#       GPU memory sampled every 20 s. Re-entrant: a finished group is skipped, a running one is waited for.
#   S2  summary + selection + verdict (scripts/tools/p3_select_scale.py, E1 old-reward runs as controls):
#         effective    -> S3
#         insufficient -> rerun b2 once with launch_dir B weight 3 and reach_lambda 0.5, summarize, stop (report)
#         interferes / unclear -> stop (report)
#   S3  formal training $FORMAL_RUN from the selected run's final checkpoint, curriculum on from stage B,
#       crash -> resume its own latest.pt with the remaining budget; STOP_REQUESTED from the monitor -> no
#       restart. 8192 courts: curriculum iteration counts halved.
#   S4  monitor in parallel: play eval every 1000 iterations and at stage C, automatic stops / early stop.
# Progress: $OUT/P3_STATUS.txt; E1b summary: $OUT/e1b_summary.json
set -u
REPO=/root/autodl-tmp/BadmintonLab-Game-V3L
PY=/root/autodl-tmp/env_isaaclab/bin/python
SWPY=/root/autodl-tmp/env_swanlab/bin/python
OUT=/root/autodl-tmp/logs/v3l/p3
LOGROOT="$REPO/logs/rsl_rl/badmintonlab_serve"
FORMAL_HOURS=${FORMAL_HOURS:-18}
FORMAL_RUN=${FORMAL_RUN:-p3_f2}
CMP_HOURS=${CMP_HOURS:-0.75}
STAGE_MAX=${STAGE_MAX:-A=4500,B=12000,C=12000}
E1B_GROUPS=${E1B_GROUPS:-"b1=2048:16:1 b2=8192:64:1 b3=8192:32:1 b4=2048:16:2"}
CONTROLS=${CONTROLS:-"e1_2048_16=p3e1_2048_mb16 e1_8192_64=p3e1_8192_mb64 e1_8192_32=p3e1_8192_mb32"}
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
STATUS="$OUT/P3_STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
say "P3 queue (v4, reward V2.1 / E1b) start, HEAD $(git rev-parse --short HEAD), formal run $FORMAL_RUN, budget ${FORMAL_HOURS} h"

mem_sampler() {  # file
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 20; done ) >> "$1" 2>/dev/null &
  MEMPID=$!
}
cmp_run() {  # run_name courts mini_batches seed [extra train_serve args...]
  local name=$1 n=$2 mb=$3 seed=$4
  shift 4
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
  say "S1 $name start ($n courts, $mb mini-batches, seed $seed${*:+, $*})"
  mem_sampler "$OUT/$name.mem"
  timeout 4500 "$PY" scripts/rsl_rl/train_serve.py --headless --num_envs "$n" --mini_batches "$mb" --seed "$seed" --stage B \
    --no_curriculum --run_name "$name" --time_budget_hours "$CMP_HOURS" --swanlab_group v3l-p3e1b "$@" \
    > "$OUT/$name.log" 2>&1
  local code=$?
  kill "$MEMPID" 2>/dev/null
  say "S1 $name exit=$code $(grep -o 'stopped: .*' "$OUT/$name.log" | tail -1)"
}

# ---------------------------------------------------------------- S1: E1b
RUNS=""
for g in $E1B_GROUPS; do
  label=${g%%=*}; spec=${g#*=}
  IFS=: read -r n mb seed <<< "$spec"
  name="p3e1b_${n}_mb${mb}_s${seed}"
  RUNS="$RUNS $label=$name"
  [ -z "${SKIP_CMP:-}" ] && cmp_run "$name" "$n" "$mb" "$seed"
done

# ---------------------------------------------------------------- S2: summary, selection, verdict
select_runs() {  # runs verdict_labels out
  "$SWPY" scripts/tools/p3_select_scale.py --repo "$REPO" --runs "$1" --controls "$CONTROLS" --seed_pair b1,b4 \
    --verdict "$2" --out "$3" 2>>"$STATUS"
}
jget() { python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or '')" "$1" "$2"; }
SEL=$(select_runs "$RUNS" b1,b2,b3,e1_2048_16 "$OUT/e1b_summary.json")
VERDICT=$(jget "$OUT/e1b_summary.json" verdict)
say "S2 verdict: ${VERDICT:-none} ($(jget "$OUT/e1b_summary.json" verdict_reason)); selection: ${SEL:-none}"
if [ "$VERDICT" = "insufficient" ]; then
  W3="p3e1b_8192_mb64_s1_w3"
  cmp_run "$W3" 8192 64 1 --stage_weights B.launch_dir=3 env.serve_reward.reach_lambda=0.5
  select_runs "$RUNS b2w3=$W3" b1,b2w3,b3,e1_2048_16 "$OUT/e1b_retry_summary.json" > /dev/null
  say "S2 retry verdict: $(jget "$OUT/e1b_retry_summary.json" verdict) ($(jget "$OUT/e1b_retry_summary.json" verdict_reason))"
  say "S2 stop: reward fix needed a retry; report to local before the formal run (P3_REWARD_FIX_AND_E1B 4.4)"
  say "P3 queue finished"; exit 0
fi
if [ "$VERDICT" != "effective" ] || [ -z "$SEL" ]; then
  say "S2 stop: verdict '${VERDICT:-none}', no formal run; report to local (P3_REWARD_FIX_AND_E1B 4.4)"
  say "P3 queue finished"; exit 0
fi
IFS=: read -r SCALE MBS SELRUN <<< "$SEL"
say "S2 selected: $SCALE courts, $MBS mini-batches, start point run $SELRUN ($(jget "$OUT/e1b_summary.json" reason))"

INIT=$(ls -d "$LOGROOT"/*_"$SELRUN" 2>/dev/null | tail -1)/latest.pt
if [ -f "$INIT" ]; then RESUME=(--resume --checkpoint "$INIT"); say "S3 start point: $INIT"
else RESUME=(); say "S3 start point: none found ($INIT), training from the pose init"; fi
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
  # --auto_resume takes $FORMAL_RUN's own latest.pt when it exists (after a crash), else the E1b start point.
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
