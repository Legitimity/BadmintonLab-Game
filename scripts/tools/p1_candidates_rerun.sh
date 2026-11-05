#!/usr/bin/env bash
# P1 robot mode for swing candidates 0-7, each with its own trunk pose (two at a time).
# Switches the ready pose (separate commit) only if a candidate beats candidate 0 by > 5 points.
set -u
REPO=/root/autodl-tmp/BadmintonLab-Game-V3L
PY=/root/autodl-tmp/env_isaaclab/bin/python
O=/root/autodl-tmp/logs/v3l/p1_trunk
STATUS=/root/autodl-tmp/logs/v3l/QUEUE_STATUS.txt
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO/source/badmintonlab_game"
mkdir -p "$O"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
say "S6b rerun candidates 0-7 with own trunk pose (HEAD $(git rev-parse --short HEAD))"
for pair in "0 1" "2 3" "4 5" "6 7"; do
  for c in $pair; do
    (timeout 1800 "$PY" scripts/tools/racket_impact_test.py --headless --mode robot --num_envs 1024 --candidate "$c" \
      --rallies 2 --out "$O/robot_c$c" > "$O/robot_c$c.log" 2>&1; say "  candidate $c exit=$?") &
  done
  wait
done
BEST=$("$PY" - "$O" 2>>"$STATUS" <<'EOF'
import json, os, sys
d = sys.argv[1]
rates = {}
for c in range(8):
    f = os.path.join(d, f"robot_c{c}.json")
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    r = j["rates"]
    rates[c] = r["served_in_rate"]
    print(f"  c{c}: in={r['served_in_rate']:.3f} no_hit={r['no_hit_rate']:.3f} tunnel={r['no_hit_tunnel_rate']:.3f} "
          f"net={r['net_rate']:.3f} out={r['out_rate']:.3f} landing_x={j['landing_x']} v_post={j['post_hit_speed']} "
          f"miss_q={j['no_hit_without_tunnel_min_dist']['quantiles']}", file=sys.stderr)
best = max(rates, key=rates.get) if rates else 0
print(best if rates and rates[best] > rates.get(0, 0.0) + 0.05 else 0)
EOF
)
say "S6b best candidate: $BEST"
if [ "$BEST" != "0" ]; then
  "$PY" scripts/tools/ready_pose.py --candidate "$BEST" --write >> /root/autodl-tmp/logs/v3l/ready_pose.log 2>&1
  git -c user.name=Legitimity -c user.email=legitimity233@gmail.com commit -qam \
    "P1: switch ready pose to swing candidate $BEST (robot-mode served_in, own trunk pose)" \
    && say "commit $(git rev-parse --short HEAD): ready pose -> candidate $BEST"
fi
touch "$O/DONE"
say "S6b finished"
