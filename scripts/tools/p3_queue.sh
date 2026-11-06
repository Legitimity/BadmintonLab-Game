#!/usr/bin/env bash
# P3 queue (P3_PLAN_V2 4.4-4.7).
#   S1  scale comparison: 2048 vs 8192 (64 mini-batches), stage B fixed weights, 45 min each, sequential.
#       Re-entrant: a comparison already finished is skipped, one still running is waited for.
#   S2  pick the scale (8192 unless its in is > 3 points lower or its fall is higher, last 50 iterations)
#   S3  formal training from stage B, curriculum on, stage maxima relaxed, total wall time < 20 h
#       (crash -> resume with the remaining budget; STOP_REQUESTED from the monitor -> no restart)
#   S4  monitor in parallel: play eval every 1000 iterations and at stage C, automatic stops / early stop
# Progress: $OUT/P3_STATUS.txt
set -u
REPO=/root/autodl-tmp/BadmintonLab-Game-V3L
PY=/root/autodl-tmp/env_isaaclab/bin/python
SWPY=/root/autodl-tmp/env_swanlab/bin/python
OUT=/root/autodl-tmp/logs/v3l/p3
FORMAL_HOURS=${FORMAL_HOURS:-19.5}
CMP_HOURS=${CMP_HOURS:-0.75}
STAGE_MAX=${STAGE_MAX:-A=4500,B=12000,C=12000}
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
STATUS="$OUT/P3_STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
say "P3 queue (v2) start, HEAD $(git rev-parse --short HEAD), formal budget ${FORMAL_HOURS} h, stage max $STAGE_MAX"

# ---------------------------------------------------------------- S1
cmp_run() {  # num_envs extra...
  local n=$1; shift
  if grep -q "stopped: " "$OUT/cmp_$n.log" 2>/dev/null; then
    say "S1 compare $n already done: $(grep -o 'stopped: .*' "$OUT/cmp_$n.log" | tail -1)"; return
  fi
  if pgrep -f "train_serve.py.*--run_name p3cmp_$n( |$)" > /dev/null; then
    say "S1 compare $n running, waiting"
    while pgrep -f "train_serve.py.*--run_name p3cmp_$n( |$)" > /dev/null; do sleep 30; done
    say "S1 compare $n finished: $(grep -o 'stopped: .*' "$OUT/cmp_$n.log" | tail -1)"; return
  fi
  say "S1 compare $n start"
  timeout 4500 "$PY" scripts/rsl_rl/train_serve.py --headless --num_envs "$n" --seed 1 --stage B \
    --no_curriculum --run_name "p3cmp_$n" --time_budget_hours "$CMP_HOURS" --swanlab_group v3l-p3cmp "$@" \
    > "$OUT/cmp_$n.log" 2>&1
  say "S1 compare $n exit=$? $(grep -o 'stopped: .*' "$OUT/cmp_$n.log" | tail -1)"
}
cmp_run 2048
cmp_run 8192 --mini_batches 64

# ---------------------------------------------------------------- S2
SCALE=$("$SWPY" - "$REPO" 2>>"$STATUS" <<'EOF'
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
repo = sys.argv[1]
res = {}
for n in (2048, 8192):
    d = sorted(glob.glob(f"{repo}/logs/rsl_rl/badmintonlab_serve/*_p3cmp_{n}"))
    if not d:
        continue
    acc = E(glob.glob(d[-1] + "/events*")[0], size_guidance={"scalars": 0}); acc.Reload()
    tags = acc.Tags()["scalars"]
    def last(tag, k=50):
        if tag not in tags:
            return None
        v = [e.value for e in acc.Scalars(tag)][-k:]
        return sum(v) / len(v)
    it = acc.Scalars("Serve/in")[-1].step if "Serve/in" in tags else 0
    res[n] = dict(it=it, **{k: last(f"Serve/{k}") for k in ("legal", "crossed", "in", "fall", "fall_after_hit", "tunnel", "vc_mean", "el_mean")})
    print(f"  cmp {n}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res[n].items()), file=sys.stderr)
a, b = res.get(2048), res.get(8192)
if not b or b["in"] is None:
    print(2048)
elif not a or a["in"] is None:
    print(8192)
else:
    # Plan 4.4: 8192 unless its in is > 3 points lower or its fall is higher (0.01 noise margin).
    print(8192 if b["in"] >= a["in"] - 0.03 and b["fall"] <= a["fall"] + 0.01 else 2048)
EOF
)
SCALE=${SCALE:-8192}
MB=""; [ "$SCALE" = "8192" ] && MB="--mini_batches 64"
say "S2 scale: $SCALE $MB"

# ---------------------------------------------------------------- S3 + S4
DONE="$OUT/FORMAL_DONE"
rm -f "$DONE" "$OUT/STOP_REQUESTED"
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
