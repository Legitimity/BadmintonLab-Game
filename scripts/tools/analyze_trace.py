"""11-13: what happens in long play rallies (trace from eval_serve.py --trace_out).

For every rally of the traced courts: length, hits, and per hit the hitter, the shuttle speed just after
the hit, the apex of that flight, where it would come down (next opponent-side x), and both robots'
positions. Prints a summary of long (>= --long_s or unresolved) vs short rallies and plots a few long
rallies top-down and side-on.

    python scripts/tools/analyze_trace.py recv_logs/trace_s4_2000_self.pt
"""

import argparse
import json
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--long_s", type=float, default=30.0)
ap.add_argument("--plots", type=int, default=4)
a = ap.parse_args()
d = torch.load(a.trace)
cols = d["cols"]
X = d["data"]  # [envs, steps, cols]
C = {k: i for i, k in enumerate(cols)}
dt = d["step_dt"]

rallies = []
for e in range(X.shape[0]):
    x = X[e]
    for r in x[:, C["rally"]].unique().tolist():
        m = x[:, C["rally"]] == r
        y = x[m]
        if y.shape[0] < 5:
            continue
        t = y[:, C["t"]] - y[0, C["t"]]
        s = y[:, C["sx"]:C["sz"] + 1]
        v = torch.zeros_like(s)
        v[1:] = (s[1:] - s[:-1]) / dt
        hits = []
        for side in (0, 1):
            for k in (y[:, C[f"hit{side}"]] > 0.5).nonzero().flatten().tolist():
                k2 = min(k + 3, y.shape[0] - 1)
                seg = s[k:]
                hits.append(dict(
                    t=float(t[k]), side=side, speed=float(v[k2].norm()), vz=float(v[k2, 2]),
                    apex=float(seg[: int(3.0 / dt), 2].max()),
                    sx=float(s[k, 0]), sy=float(s[k, 1]), sz=float(s[k, 2]),
                    r0=(float(y[k, C["r0x"]]), float(y[k, C["r0y"]])), r1=(float(y[k, C["r1x"]]), float(y[k, C["r1y"]])),
                ))
        hits.sort(key=lambda h: h["t"])
        stuck = bool(y[:, C["stuck"]].max() > 0.5)
        dist0 = float((y[1:, C["r0x"]:C["r0y"] + 1] - y[:-1, C["r0x"]:C["r0y"] + 1]).norm(dim=-1).sum())
        dist1 = float((y[1:, C["r1x"]:C["r1y"] + 1] - y[:-1, C["r1x"]:C["r1y"] + 1]).norm(dim=-1).sum())
        rallies.append(dict(env=e, rally=int(r), len=float(t[-1]), stuck=stuck, n_hits=len(hits), hits=hits,
                            walk0=dist0, walk1=dist1, fell0=float(y[:, C["r0z"]].min()) < 0.45, fell1=float(y[:, C["r1z"]].min()) < 0.45))


def summ(rs, name):
    if not rs:
        print(name, "none")
        return
    hs = [h for r in rs for h in r["hits"][1:]]  # skip the serve
    q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")  # noqa: E731
    print(f"== {name}: {len(rs)} rallies, len p50 {q([r['len'] for r in rs], .5):.1f} s, hits/rally {sum(r['n_hits'] for r in rs) / len(rs):.1f}, "
          f"unresolved {sum(r['stuck'] for r in rs)}, fell {sum(r['fell0'] or r['fell1'] for r in rs)}")
    if not hs:
        return
    print(f"   returns: speed p10/50/90 {q([h['speed'] for h in hs], .1):.1f}/{q([h['speed'] for h in hs], .5):.1f}/{q([h['speed'] for h in hs], .9):.1f} m/s, "
          f"apex p10/50/90 {q([h['apex'] for h in hs], .1):.2f}/{q([h['apex'] for h in hs], .5):.2f}/{q([h['apex'] for h in hs], .9):.2f} m, "
          f"contact height p50 {q([h['sz'] for h in hs], .5):.2f} m, contact |x| p50 {q([abs(h['sx']) for h in hs], .5):.2f} m")
    walk = [(r["walk0"] + r["walk1"]) / 2 / max(r["len"], 1e-3) for r in rs]
    print(f"   footwork: mean walking speed p50 {q(walk, .5):.2f} m/s; hitter contact x spread (std) {torch.tensor([h['sx'] for h in hs]).std():.2f} m, y spread {torch.tensor([h['sy'] for h in hs]).std():.2f} m")
    # interval between consecutive hits
    gaps = [b["t"] - a_["t"] for r in rs for a_, b in zip(r["hits"], r["hits"][1:])]
    print(f"   time between hits p10/50/90 {q(gaps, .1):.2f}/{q(gaps, .5):.2f}/{q(gaps, .9):.2f} s")


long_r = [r for r in rallies if r["stuck"] or r["len"] >= a.long_s]
short_r = [r for r in rallies if r not in long_r]
summ(long_r, "long")
summ(short_r, "short")
out = Path(a.trace).with_suffix(".summary.json")
out.write_text(json.dumps(rallies, indent=0), encoding="utf-8")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(0)
sel = sorted(long_r, key=lambda r: -r["len"])[: a.plots]
if sel:
    fig, axes = plt.subplots(len(sel), 2, figsize=(14, 3.2 * len(sel)), squeeze=False)
    for ax, r in zip(axes, sel):
        x = X[r["env"]]
        y = x[x[:, C["rally"]] == r["rally"]]
        ax[0].plot(y[:, C["sx"]], y[:, C["sy"]], lw=0.6, color="tab:green", label="shuttle")
        ax[0].plot(y[:, C["r0x"]], y[:, C["r0y"]], lw=1.2, color="tab:blue", label="robot0")
        ax[0].plot(y[:, C["r1x"]], y[:, C["r1y"]], lw=1.2, color="tab:red", label="robot1")
        ax[0].plot([-6.7, 6.7, 6.7, -6.7, -6.7], [-2.59, -2.59, 2.59, 2.59, -2.59], color="k", lw=0.5)
        ax[0].axvline(0, color="k", lw=1)
        ax[0].set_aspect("equal")
        ax[0].set_title(f"env {r['env']} rally {r['rally']}: {r['len']:.0f} s, {r['n_hits']} hits, top view")
        ax[0].legend(fontsize=7, loc="upper right")
        ax[1].plot(y[:, C["t"]] - y[0, C["t"]], y[:, C["sz"]], lw=0.6, color="tab:green", label="shuttle z")
        ax[1].plot(y[:, C["t"]] - y[0, C["t"]], y[:, C["sx"]] / 3, lw=0.6, color="tab:gray", label="shuttle x / 3")
        for h in r["hits"]:
            ax[1].axvline(h["t"], color="tab:blue" if h["side"] == 0 else "tab:red", lw=0.4)
        ax[1].set_xlabel("s")
        ax[1].legend(fontsize=7)
    fig.tight_layout()
    png = Path(a.trace).with_suffix(".png")
    fig.savefig(png, dpi=90)
    print("plot", png)
