#!/usr/bin/env bash
cd /root/autodl-tmp/BadmintonLab-Game-V3L
export OMNI_KIT_ACCEPT_EULA=YES
PY=/root/autodl-tmp/env_isaaclab/bin/python
PLAY=/root/autodl-tmp/BadmintonLab-Game-Play
OUT=/root/autodl-tmp/logs/v3l/p5/v2_checks
mkdir -p $OUT
run() {  # tag ours opp side envs rallies seed [probe]
  local extra=""; [ -n "${8:-}" ] && extra="--probe_out $OUT/probe_$1.json"
  flock $PLAY/.eval.lock bash -c "$PY scripts/eval_serve.py --play_root $PLAY --ours $2 --opponent $3 --side $4 --num_envs $5 --rallies $6 --seed $7 --python $PY --out $OUT/$1.json $extra > $OUT/$1.log 2>&1; git -C $PLAY checkout -- source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player0_cfg.py source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player1_cfg.py"
  echo "[$(date '+%T')] $1 exit $?"
}
B=exports/baseline_14500/policy.pt
run probe_a4_1500_s0 exports/p5_pilot_a4_1500 exports/p3_final 0 64 8 42 probe
run probe_a4_1000_s0 exports/p5_pilot_a4_1000 exports/p3_final 0 64 8 42 probe
run probe_p3_s0 exports/p3_final exports/p3_final 0 64 8 42 probe
run base_p3_e32_s42 exports/p3_final $B 0 32 16 42
run base_p3_e64_s42 exports/p3_final $B 0 64 16 42
run base_p3_e64_s43 exports/p3_final $B 0 64 16 43
run base_p3_e64_s42_side1 exports/p3_final $B 1 64 16 42
run base_a4_e32_s42 exports/p5_pilot_a4_1500 $B 0 32 16 42
echo done
