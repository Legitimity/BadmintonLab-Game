"""Pick the S2 starting checkpoint from eval_receive --real_serves results (first ball per source)."""
import json
import sys
from pathlib import Path

best = None
for p in sys.argv[1:]:
    try:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"skip {p}: {e}", file=sys.stderr)
        continue
    st = {k: r.get(f"first_{k}_station_ok") or 0.0 for k in ("s0", "s2", "s3", "s4")}
    m = (st["s0"] + st["s3"] + st["s4"]) / 3
    print(f"{r.get('checkpoint')} m034={m:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in st.items()) + f" fall_ball={r.get('fall_ball', float('nan')):.4f}", file=sys.stderr)
    if best is None or m > best[0]:
        best = (m, r.get("checkpoint"))
print(best[1] if best else "")
