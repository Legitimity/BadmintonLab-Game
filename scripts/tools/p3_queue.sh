#!/usr/bin/env bash
# P3 queue (P3_PLAN_V2 4.4-4.7, P3_SPEEDUP_PLAN 4.2).
#   S1  E1: groups courts:mini_batches ($E1_GROUPS), stage B fixed weights, 45 min each, sequential.
#       Re-entrant: a group already finished is skipped.
#   S2  pick the group with scripts/tools/p3_select_scale.py (wall-clock time to 5-min in >= 0.6)
#   S3  formal training (resumes run p3_s1), curriculum on, stage maxima relaxed, $FORMAL_HOURS h
#       (crash -> resume with the remaining budget; STOP_REQUESTED from the monitor -> no restart)
#   S4  monitor in parallel: play eval every 1000 iterations and at stage C, automatic stops / early stop
# Progress: $OUT/P3_STATUS.txt
set -u
REPO=/root/autodl-tmp/BadmintonLab-Game-V3L
PY=/root/autodl-tmp/env_isaaclab/bin/python
SWPY=/root/autodl-tmp/env_swanlab/bin/python
OUT=/root/autodl-tmp/logs/v3l/p3
FORMAL_HOURS=${FORMAL_HOURS:-18}
CMP_HOURS=${CMP_HOURS:-0.75}
STAGE_MAX=${STAGE_MAX:-A=4500,B=12000,C=12000}
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
STATUS="$OUT/P3_STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
say "P3 queue (v2) start, HEAD $(git rev-parse --short HEAD), formal budget ${FORMAL_HOURS} h, stage max $STAGE_MAX"

# ---------------------------------------------------------------- S1 (E1 of P3_SPEEDUP_PLAN 4.2)
# Groups "courts:mini_batches", fixed wall time each, stage B fixed weights, seed 1.
E1_GROUPS=${E1_GROUPS:-"2048:16 8192:64 8192:32 4096:32"}
cmp_run() {  # courts mini_batches
  local n=$1 mb=$2 name="p3e1_${1}_mb${2}"
  if grep -q "stopped: " "$OUT/$name.log" 2>/dev/null; then
    say "S1 $name already done: $(grep -o 'stopped: .*' "$OUT/$name.log" | tail -1)"; return
  fi
  say "S1 $name start"
  timeout 4500 "$PY" scripts/rsl_rl/train_serve.py --headless --num_envs "$n" --mini_batches "$mb" --seed 1 --stage B \
    --no_curriculum --run_name "$name" --time_budget_hours "$CMP_HOURS" --swanlab_group v3l-p3e1 \
    > "$OUT/$name.log" 2>&1
  say "S1 $name exit=$? $(grep -o 'stopped: .*' "$OUT/$name.log" | tail -1)"
}
if [ -z "${SKIP_CMP:-}" ]; then
  for g in $E1_GROUPS; do cmp_run "${g%%:*}" "${g##*:}"; done
fi

# ---------------------------------------------------------------- S2: selection (plan 4.2 rules)
SEL=$("$SWPY" scripts/tools/p3_select_scale.py --repo "$REPO" --groups "$E1_GROUPS" --out "$OUT/e1_summary.json" 2>>"$STATUS")
SCALE=${SEL%%:*}; MBS=${SEL##*:}
if [ -z "$SCALE" ] || [ -z "$MBS" ]; then SCALE=2048; MBS=16; say "S2 selection failed, default 2048:16"; fi
MB="--mini_batches $MBS"
say "S2 selected: $SCALE courts, $MBS mini-batches"

# ---------------------------------------------------------------- S3 + S4
DONE="$OUT/FORMAL_DONE"
rm -f "$DONE" "$OUT/STOP_REQUESTED" "$OUT/monitor_state.json"
"$SWPY" scripts/tools/p3_monitor.py --run_name p3_s1 --out "$OUT" --done_file "$DONE" > "$OUT/monitor.log" 2>&1 &
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
  timeout $(( left + 1800 )) "$PY" scripts/rsl_rl/train_serve.py --headless --num_envs "$SCALE" $MB --seed 1 --stage B \
    --stage_max_iters "$STAGE_MAX" --run_name p3_s1 --time_budget_hours "$hours" --auto_resume \
    --swanlab_group v3l-p3 >> "$OUT/formal.log" 2>&1
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
