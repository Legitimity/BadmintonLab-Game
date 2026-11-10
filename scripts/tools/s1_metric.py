"""Real-serve footwork metric from the S1 log (first ball of an episode = a play rally start).

Usage: python s1_metric.py <log> <history.json> [--window 100]
Appends {"iter", "s0", "s2", "s3", "s4", "syn", "m034"} to the history and prints the decision:
  CONTINUE / RESET / STOP (plateau rule: gain of m034 = mean(s0, s3, s4) < 0.03 for two consecutive windows;
  the first plateau triggers one leg-std reset, the next plateau stops).
"""
import json
import re
import sys
from pathlib import Path

log, hist_path = Path(sys.argv[1]), Path(sys.argv[2])
window = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 100
GAIN = 0.03

lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
first = [ln for ln in lines if ln.startswith("[FIRST]")][-window:]
it = None
for ln in reversed(lines):
    m = re.search(r"Learning iteration (\d+)/", ln)
    if m:
        it = int(m.group(1))
        break
acc = {}
for ln in first:
    for tok in ln.split()[1:]:
        m = re.match(r"(\w+):hit=([\d.na]+)/st=([\d.na]+)/n=(\d+)", tok)
        if not m or m.group(3) == "nan":
            continue
        name, st, n = m.group(1), float(m.group(3)), int(m.group(4))
        s, c = acc.get(name, (0.0, 0))
        acc[name] = (s + st * n, c + n)
rate = {k: (s / c if c else 0.0) for k, (s, c) in acc.items()}
rec = dict(iter=it, **{k: round(rate.get(k, 0.0), 4) for k in ("syn", "s0", "s2", "s3", "s4")})
rec["m034"] = round((rec["s0"] + rec["s3"] + rec["s4"]) / 3, 4)

hist = json.loads(hist_path.read_text()) if hist_path.exists() else {"windows": [], "reset_done": False}
hist["windows"].append(rec)
w = hist["windows"]
decision = "CONTINUE"
if len(w) >= 3:
    g1 = w[-1]["m034"] - w[-2]["m034"]
    g2 = w[-2]["m034"] - w[-3]["m034"]
    rec["gain"] = round(g1, 4)
    if g1 < GAIN and g2 < GAIN:
        decision = "STOP" if hist["reset_done"] else "RESET"
elif len(w) == 2:
    rec["gain"] = round(w[-1]["m034"] - w[-2]["m034"], 4)
if hist["reset_done"] and len(w) >= 2 and w[-2].get("after_reset") and rec.get("gain", 1.0) < GAIN:
    decision = "STOP"  # the chunk right after the reset did not help either
if decision == "RESET":
    hist["reset_done"] = True
    rec["reset_next"] = True
if len(w) >= 2 and w[-2].get("reset_next"):
    rec["after_reset"] = True
rec["decision"] = decision
hist_path.write_text(json.dumps(hist, indent=1))
print(decision, json.dumps(rec))
