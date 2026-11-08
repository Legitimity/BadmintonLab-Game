#!/usr/bin/env bash
# P3_FINETUNE_DT1MS E0: p3_final in the training env (deterministic and sampled, physics 1/2/5 ms) and in play
# (a5e0780, 1 ms), zero opponent, both sides. Play evaluations hold <play>/.eval.lock (they rewrite play's cfgs).
#   OUT=/root/autodl-tmp/logs/v3l/dt1ms bash scripts/tools/dt1ms_e0.sh
set -u
REPO=${REPO:-/root/autodl-tmp/BadmintonLab-Game-V3L}
PLAY=${PLAY:-/root/autodl-tmp/BadmintonLab-Game-Play}
PY=${PY:-/root/autodl-tmp/env_isaaclab/bin/python}
OUT=${OUT:-/root/autodl-tmp/logs/v3l/dt1ms}
CK=${CK:-$REPO/logs/rsl_rl/badmintonlab_serve/2026-11-06_22-49-04_p3_c7/model_3000.pt}
EXP=${EXP:-$REPO/exports/p3_final}
PLAY_ENVS=${PLAY_ENVS:-128}
PLAY_RALLIES=${PLAY_RALLIES:-8}     # per court: 128 x 8 = 1024 rallies per side
DET_DTS=${DET_DTS:-"0.001 0.005 0.002"}
DET_ITERS=${DET_ITERS:-6}
mkdir -p "$OUT/e0"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/E0_STATUS.txt"; }
say "E0 start, HEAD $(git rev-parse --short HEAD), play $(git -C "$PLAY" rev-parse --short HEAD)"

play_eval() {  # side opponent rallies tag
  local t0=$(date +%s)
  (
    flock -x 9
    "$PY" scripts/eval_serve.py --play_root "$PLAY" --ours "$EXP" --opponent "$2" --side "$1" \
      --num_envs "$PLAY_ENVS" --rallies "$3" --out "$OUT/e0/play_$4_side$1.json" > "$OUT/e0/play_$4_side$1.log" 2>&1
  ) 9>>"$PLAY/.eval.lock"
  say "play $4 side $1: exit $? in $(( $(date +%s) - t0 )) s"
}

(
  for dt in $DET_DTS; do
    t0=$(date +%s)
    "$PY" scripts/tools/p3_det_eval.py --headless --num_envs 2048 --checkpoint "$CK" --iters "$DET_ITERS" \
      --physics_dt "$dt" --out "$OUT/e0/det_p3final_dt${dt}.json" > "$OUT/e0/det_p3final_dt${dt}.log" 2>&1
    say "det dt=$dt: exit $? in $(( $(date +%s) - t0 )) s"
  done
) &
DET=$!
play_eval 0 zero "$PLAY_RALLIES" zero
play_eval 1 zero "$PLAY_RALLIES" zero
wait $DET
say "E0 done"
