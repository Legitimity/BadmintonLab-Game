#!/usr/bin/env bash
# P3 formal run F2 (P3_LAUNCH_F2), serial variant chosen by the user: each C branch runs 8 h alone
# (two 2048 processes in parallel do not share the GPU evenly).
#   S1  p3_ab: 2048:16 from the E1b b1 checkpoint, curriculum on from stage B, --stop_after_stage B.
#       Writes model_passed_B_<P>.pt (curriculum state already in C, blend_left=300) and stops.
#       B at its maximum (4000) without passing -> stop and report, no S2.
#   S2  C branches from model_passed_B_<P>.pt, one after the other, $C_HOURS each (crash restarts inside
#       the branch's own window, at most 3 per branch). Order: $BRANCHES (default p3_c7 first).
#         p3_c7: --stage_param C.v_close_max=7.0      p3_c8: v_close_max=8.0 (default)
#       Each branch has its own monitor (play eval every 1000 iterations, early stops held back until
#       P+1500 by --min_stop_iter so both cover the comparison window) and its own --out.
#   S3  p3_compare_c.py: same-iteration comparison (P+1300..P+1500) + zero-opponent play check -> c_compare.json.
# Progress: $OUT/P3_STATUS.txt (queue), $OUT/<run>/P3_STATUS.txt (monitors).
set -u
REPO=/root/autodl-tmp/BadmintonLab-Game-V3L
PY=/root/autodl-tmp/env_isaaclab/bin/python
SWPY=/root/autodl-tmp/env_swanlab/bin/python
OUT=${OUT:-/root/autodl-tmp/logs/v3l/p3}
LOGROOT="$REPO/logs/rsl_rl/badmintonlab_serve"
AB_HOURS=${AB_HOURS:-18}
C_HOURS=${C_HOURS:-8}
BRANCHES=${BRANCHES:-"p3_c7 p3_c8"}
NUM_ENVS=${NUM_ENVS:-2048}
MBS=${MBS:-16}
STAGE_MAX=${STAGE_MAX:-A=4500,B=4000,C=12000}
GATES=${GATES:-B.crossed=0.65,C.in=0.75}
GATE_IN=${GATE_IN:-B=0.60,C=0.75}
CUR_ITERS=${CUR_ITERS:-}            # testing only, e.g. ema=5,blend=10,min=5
AB_INIT=${AB_INIT:-}                # default: latest.pt of the newest *_p3e1b_2048_mb16_s1 run
EXTRA=${EXTRA:-}                    # testing only, extra train_serve args (e.g. --swanlab_mode disabled)
MON_EXTRA=${MON_EXTRA:-}            # testing only, extra monitor args
TAG=${TAG:-}                        # testing only, run name suffix (keeps test runs away from the real names)
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
STATUS="$OUT/P3_STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
say "P3 F2 queue start, HEAD $(git rev-parse --short HEAD), $NUM_ENVS:$MBS, branches '$BRANCHES' x ${C_HOURS} h (serial), gates $GATES"
CUR=(); [ -n "$CUR_ITERS" ] && CUR=(--curriculum_iters "$CUR_ITERS")
read -r -a EXTRA_A <<< "$EXTRA"
read -r -a MON_A <<< "$MON_EXTRA"

# run_segment run_name hours log [extra train_serve args...]; sets SEG_REASON. Crash -> resume own latest.pt.
run_segment() {
  local name=$1 hours=$2 log=$3
  shift 3
  local deadline tries=0 left h start_line code
  deadline=$(( $(date +%s) + $(python3 -c "print(int($hours*3600))") ))
  SEG_REASON=""
  while :; do
    left=$(( deadline - $(date +%s) ))
    if [ "$left" -lt 600 ]; then say "$name: budget exhausted"; SEG_REASON="budget"; break; fi
    h=$(python3 -c "print(round($left/3600, 3))")
    start_line=$(( $(wc -l < "$log" 2>/dev/null || echo 0) + 1 ))
    say "$name: launch (attempt $((tries+1)), budget ${h} h)"
    # --auto_resume takes $name's own latest.pt when it exists (after a crash), else --checkpoint.
    timeout $(( left + 1800 )) "$PY" scripts/rsl_rl/train_serve.py --headless --num_envs "$NUM_ENVS" --mini_batches "$MBS" \
      --seed 1 --stage_max_iters "$STAGE_MAX" --stage_gates "$GATES" "${CUR[@]}" "$@" --auto_resume \
      --time_budget_hours "$h" --swanlab_group v3l-p3 --run_name "$name" "${EXTRA_A[@]}" >> "$log" 2>&1
    code=$?
    SEG_REASON=$(tail -n +"$start_line" "$log" | grep -o 'stopped: .*' | tail -1)
    say "$name: exit=$code ${SEG_REASON:-no stop line}"
    if [ -f "$OUT/$name/STOP_REQUESTED" ]; then say "$name: stopped by monitor: $(cat "$OUT/$name/STOP_REQUESTED")"; break; fi
    case "$SEG_REASON" in
      *stop_after_B*|*time_budget*|*max_iterations*|*nonfinite*|*stage_*_max_iters*) break ;;
    esac
    tries=$((tries+1))
    if [ "$tries" -ge 3 ]; then say "$name: gave up after 3 crashes"; SEG_REASON="crashed"; break; fi
    sleep 30
  done
}

start_monitor() {  # run_name [monitor args...]
  local name=$1
  shift
  mkdir -p "$OUT/$name"
  rm -f "$OUT/$name/DONE" "$OUT/$name/STOP_REQUESTED"
  "$SWPY" scripts/tools/p3_monitor.py --run_name "$name" --out "$OUT/$name" --done_file "$OUT/$name/DONE" \
    --gate_in "$GATE_IN" "$@" "${MON_A[@]}" > "$OUT/$name/monitor.log" 2>&1 &
  MONS="${MONS:-} $!"
}

finish_monitor() {  # run_name; the monitor evaluates the final checkpoint while the next segment trains (play lock)
  touch "$OUT/$1/DONE"
  say "$1: monitor told to finish (final checkpoint eval runs in the background)"
}

wait_monitors() {
  say "waiting for the monitors:${MONS:-}"
  for m in ${MONS:-}; do wait "$m"; done
  git -C /root/autodl-tmp/BadmintonLab-Game-Play status --porcelain | head -3 | sed 's/^/  play dirty: /' >> "$STATUS"
}

passed_b() {  # newest model_passed_B_<P>.pt of p3_ab
  ls -t "$LOGROOT"/*_p3_ab$TAG/model_passed_B_*.pt 2>/dev/null | head -1
}

# ---------------------------------------------------------------- S1: p3_ab
PASSED=$(passed_b)
if [ -n "$PASSED" ]; then
  say "S1 p3_ab already passed B: $PASSED"
else
  if [ -z "$AB_INIT" ]; then
    AB_INIT=$(ls -d "$LOGROOT"/*_p3e1b_2048_mb16_s1 2>/dev/null | tail -1)/latest.pt
  fi
  if [ ! -f "$AB_INIT" ]; then
    d=$(dirname "$AB_INIT")
    AB_INIT=$(ls "$d"/model_*.pt 2>/dev/null | grep -E 'model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1)
    say "S1 no latest.pt, using the highest model_<k>.pt: ${AB_INIT:-none}"
  fi
  if [ ! -f "$AB_INIT" ]; then say "S1 stop: no b1 checkpoint"; say "P3 F2 queue finished"; exit 1; fi
  say "S1 p3_ab start point: $AB_INIT (b1, curriculum off there: stage B counted from 0; stage A skipped)"
  start_monitor p3_ab$TAG
  run_segment p3_ab$TAG "$AB_HOURS" "$OUT/p3_ab$TAG.log" --stage B --stop_after_stage B --resume --checkpoint "$AB_INIT"
  finish_monitor p3_ab$TAG
  PASSED=$(passed_b)
  case "$SEG_REASON" in
    *stop_after_B*) ;;
    *) say "S1 stop: p3_ab ended with '${SEG_REASON}' without passing B; no C branches (P3_LAUNCH_F2 3)"
       wait_monitors; say "P3 F2 queue finished"; exit 0 ;;
  esac
  if [ -z "$PASSED" ]; then say "S1 stop: stop_after_B but no model_passed_B_*.pt"; say "P3 F2 queue finished"; exit 1; fi
fi
P=$(basename "$PASSED" .pt); P=${P##*_}
say "S1 done: P=$P ($PASSED); comparison window $((P+1300))..$((P+1500)), early stops held until $((P+1500))"

# ---------------------------------------------------------------- S2: C branches, serial
for br in $BRANCHES; do
  name=$br$TAG
  case "$br" in
    p3_c7) BR=(--stage_param C.v_close_max=7.0) ;;
    p3_c8) BR=() ;;
    *) say "S2 unknown branch $name, skipped"; continue ;;
  esac
  if grep -q "stopped: " "$OUT/$name.log" 2>/dev/null && [ -f "$OUT/$name/DONE" ]; then
    say "S2 $name already done: $(grep -o 'stopped: .*' "$OUT/$name.log" | tail -1)"; continue
  fi
  say "S2 $name start (${BR[*]:-v_close_max default 8.0}), ${C_HOURS} h from $PASSED"
  start_monitor "$name" --min_stop_iter $((P+1500))
  run_segment "$name" "$C_HOURS" "$OUT/$name.log" --stage C "${BR[@]}" --resume --checkpoint "$PASSED"
  finish_monitor "$name"
done
wait_monitors

# ---------------------------------------------------------------- S3: compare
say "S3 compare"
"$SWPY" scripts/tools/p3_compare_c.py --P "$P" --out "$OUT" --runs "c7=p3_c7$TAG,c8=p3_c8$TAG" ${CMP_EXTRA:-} >> "$OUT/compare.log" 2>&1
say "S3 compare exit=$? ($(python3 -c "import json; d=json.load(open('$OUT/c_compare.json')); print(d.get('verdict'), '|', d.get('reason'))" 2>/dev/null))"
say "P3 F2 queue finished"
