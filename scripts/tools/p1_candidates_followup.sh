#!/usr/bin/env bash
# Follow-up to overnight_queue.sh: waits for it, then re-runs P1 robot mode for all swing
# candidates with each candidate's own trunk pose (the first S1 pass reused candidate 0's trunk).
set -u
REPO=/root/autodl-tmp/BadmintonLab-Game-V3L
PY=/root/autodl-tmp/env_isaaclab/bin/python
OUT=/root/autodl-tmp/logs/v3l
STATUS="$OUT/QUEUE_STATUS.txt"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
until grep -q "queue finished" "$STATUS" 2>/dev/null; do sleep 60; done
say "S6 robot-mode candidates 0-7 with per-candidate trunk pose"
mkdir -p "$OUT/p1_trunk"
robot() {
  PYTHONPATH="$REPO/source/badmintonlab_game" timeout 1800 "$PY" scripts/tools/racket_impact_test.py --headless --mode robot \
    --num_envs 1024 --candidate "$1" --rallies 2 --out "$OUT/p1_trunk/robot_c$1" > "$OUT/p1_trunk/robot_c$1.log" 2>&1
  say "  candidate $1 exit=$?"
}
for pair in "0 1" "2 3" "4 5" "6 7"; do
  for c in $pair; do robot "$c" & done
  wait
done
BEST=$("$PY" - "$OUT/p1_trunk" 2>>"$STATUS" <<'EOF'
import json, os, sys
d = sys.argv[1]
rates = {}
for c in range(8):
    f = os.path.join(d, f"robot_c{c}.json")
    if os.path.exists(f):
        j = json.load(open(f))
        rates[c] = j["rates"]["served_in_rate"]
        r = j["rates"]
        print(f"  c{c}: in={r['served_in_rate']:.3f} no_hit={r['no_hit_rate']:.3f} tunnel={r['no_hit_tunnel_rate']:.3f} "
              f"net={r['net_rate']:.3f} out={r['out_rate']:.3f} landing_x={j['landing_x']} v_post={j['post_hit_speed']} "
              f"miss_q={j['no_hit_without_tunnel_min_dist']['quantiles']}", file=sys.stderr)
best = max(rates, key=rates.get) if rates else 0
print(best if rates and rates[best] > rates.get(0, 0.0) + 0.05 else 0)
EOF
)
say "S6 best candidate: $BEST"
if [ "$BEST" != "0" ]; then
  "$PY" scripts/tools/ready_pose.py --candidate "$BEST" --write >> "$OUT/ready_pose.log" 2>&1
  git -c user.name=Legitimity -c user.email=legitimity233@gmail.com commit -qam \
    "P1: switch ready pose to swing candidate $BEST (best robot-mode served_in with own trunk pose)" \
    && say "commit $(git rev-parse --short HEAD): ready pose -> candidate $BEST"
fi
say "followup finished"
