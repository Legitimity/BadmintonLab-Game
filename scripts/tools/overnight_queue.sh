#!/usr/bin/env bash
# Overnight P0-P2 test queue for the serve-v3-local branch (run from anywhere).
#   S1  P1 robot mode, candidates 1-7 (two at a time), pick the best candidate
#   S2  throughput benchmark (exclusive GPU)
#   S3  P2 short training (2048 courts)
#   S4  export + play evaluation (zero / model_test opponent, both sides; sequential: they edit play's cfg files)
#   S5  CPU tests again
# Progress: $OUT/QUEUE_STATUS.txt
set -u
REPO=/root/autodl-tmp/BadmintonLab-Game-V3L
PLAY=/root/autodl-tmp/BadmintonLab-Game-Play
PY=/root/autodl-tmp/env_isaaclab/bin/python
OUT=/root/autodl-tmp/logs/v3l
P2_ITERS=${P2_ITERS:-500}
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES
TRAIN_PP="$REPO/source/badmintonlab_game"
STATUS="$OUT/QUEUE_STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
git_commit() { git -c user.name=Legitimity -c user.email=legitimity233@gmail.com commit -qam "$1" && say "commit $(git rev-parse --short HEAD): $1"; }

say "queue start, HEAD $(git rev-parse --short HEAD)"

# ---------------------------------------------------------------- S1
say "S1 robot-mode candidates 1-7"
robot() {
  PYTHONPATH="$TRAIN_PP" timeout 1800 "$PY" scripts/tools/racket_impact_test.py --headless --mode robot \
    --num_envs 1024 --candidate "$1" --rallies 2 --out "$OUT/impact_robot_c$1" > "$OUT/impact_robot_c$1.log" 2>&1
  say "  candidate $1 exit=$?"
}
for pair in "1 2" "3 4" "5 6" "7"; do
  for c in $pair; do robot "$c" & done
  wait
done
BEST=$("$PY" - "$OUT" <<'EOF'
import json, sys, glob, os
out = sys.argv[1]
best, base = None, None
for c in range(8):
    f = os.path.join(out, f"impact_robot_c{c}.json") if c else os.path.join(out, "impact_robot_c0_diag.json")
    if c == 0 and not os.path.exists(f):
        f = os.path.join(out, "impact_robot_c0.json")
    if not os.path.exists(f):
        continue
    r = json.load(open(f))["rates"]["served_in_rate"]
    print(f"candidate {c}: served_in_rate {r:.3f}", file=sys.stderr)
    if c == 0:
        base = r
    if best is None or r > best[1]:
        best = (c, r)
# Switch only for a clear gain over the shipped candidate 0.
print(best[0] if best and base is not None and best[1] > base + 0.05 else 0)
EOF
)
"$PY" - "$OUT" >> "$STATUS" 2>&1 <<'EOF'
import json, sys, os
for c in range(8):
    for name in (f"impact_robot_c{c}_diag.json", f"impact_robot_c{c}.json"):
        f = os.path.join(sys.argv[1], name)
        if os.path.exists(f):
            d = json.load(open(f)); print(f"  c{c} ({name}): {json.dumps(d['rates'])} landing_x={d['landing_x']}"); break
EOF
say "S1 best candidate: $BEST"
if [ "$BEST" != "0" ]; then
  "$PY" scripts/tools/ready_pose.py --candidate "$BEST" --write >> "$OUT/ready_pose.log" 2>&1
  git_commit "P1: switch ready pose to swing candidate $BEST (robot-mode served_in best)"
fi

# ---------------------------------------------------------------- S2
say "S2 throughput benchmark (exclusive)"
LOG="$OUT/bench" PYTHONPATH="$TRAIN_PP" bash scripts/tools/bench.sh > "$OUT/bench_driver.log" 2>&1
say "S2 done exit=$?"

# ---------------------------------------------------------------- S3
say "S3 P2 short training: 2048 courts, $P2_ITERS iterations"
PYTHONPATH="$TRAIN_PP" timeout 10800 "$PY" scripts/rsl_rl/train_serve.py --headless --num_envs 2048 --seed 1 \
  --max_iterations "$P2_ITERS" --run_name p2_short --swanlab_mode cloud --swanlab_group v3l-p2 > "$OUT/p2_train.log" 2>&1
say "S3 done exit=$?"
RUN=$(ls -d "$REPO"/logs/rsl_rl/badmintonlab_serve/*_p2_short 2>/dev/null | sort | tail -1)
CKPT="$RUN/latest.pt"
[ -f "$CKPT" ] || CKPT=$(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1)
say "S3 checkpoint: ${CKPT:-none}"

# ---------------------------------------------------------------- S4
if [ -n "${CKPT:-}" ] && [ -f "$CKPT" ]; then
  EXP="$REPO/exports/p2_short"
  "$PY" scripts/export_torchscript.py --checkpoint "$CKPT" --out "$EXP" > "$OUT/p2_export.log" 2>&1
  say "S4 export exit=$?"
  # Safety net: eval_serve edits play's player cfgs; restore them from git whatever happens.
  for opp in zero model_test; do
    for side in 0 1; do
      timeout 3600 "$PY" scripts/eval_serve.py --play_root "$PLAY" --ours "$EXP" --opponent "$opp" --side "$side" \
        --num_envs 16 --rallies 50 --python "$PY" --out "$OUT/eval_${opp}_side${side}.json" > "$OUT/eval_${opp}_side${side}.log" 2>&1
      say "  eval opponent=$opp side=$side exit=$?"
      git -C "$PLAY" checkout -- source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player0_cfg.py \
        source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player1_cfg.py
    done
  done
else
  say "S4 skipped: no checkpoint"
fi
say "play tree status: $(git -C "$PLAY" status --porcelain | tr '\n' ' ')"

# ---------------------------------------------------------------- S5
CUDA_VISIBLE_DEVICES=-1 "$PY" -m pytest tests -q > "$OUT/pytest_final.log" 2>&1
say "S5 pytest: $(tail -1 "$OUT/pytest_final.log")"
say "queue finished"
