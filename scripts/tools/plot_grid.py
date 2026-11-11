"""Heatmaps of an eval_grid.py result (11-11): per landing cell (x band net -> back line, y band), one panel
per apex tier, for touch rate, return won (in) rate and fall rate. Also prints a text table.

  python scripts/tools/plot_grid.py grid.json [--out grid.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

APEX = ("flat (<2.5 m)", "mid (2.5-4 m)", "high (>4 m)")
METRICS = (("touched", "touch", "Blues"), ("won", "return in", "Greens"), ("fell", "fall", "Reds"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    d = json.loads(a.path.read_text(encoding="utf-8"))
    g = d["grid"]
    nx, ny, na = len(g["x"]) - 1, len(g["y"]) - 1, len(g["apex"]) - 1
    cell = np.array(d["cell"])
    pr = {k: np.array(v) for k, v in d["per_row"].items()}
    done = pr["done"] > 0
    valid = np.array(d["valid"], dtype=bool)
    tabs = {}
    for key, _, _ in METRICS:
        m = np.full((na, nx, ny), np.nan)
        for c in range(nx * ny * na):
            sel = (cell == c) & done
            if sel.sum() >= 5:
                ix, rest = divmod(c, ny * na)
                iy, ia = divmod(rest, na)
                m[ia, ix, iy] = (pr[key][sel] / pr["done"][sel]).mean()
        tabs[key] = m
    cov = np.full((na, nx, ny), np.nan)
    for c in range(nx * ny * na):
        sel = cell == c
        if sel.sum():
            ix, rest = divmod(c, ny * na)
            iy, ia = divmod(rest, na)
            cov[ia, ix, iy] = valid[sel].mean()

    xl = [f"{-g['x'][i]:.1f}-{-g['x'][i + 1]:.1f}" for i in range(nx)]
    yl = [f"{g['y'][i]:+.1f}..{g['y'][i + 1]:+.1f}" for i in range(ny)]
    fig, axes = plt.subplots(len(METRICS) + 1, na, figsize=(4.2 * na, 3.6 * (len(METRICS) + 1)), constrained_layout=True)
    rows = list(METRICS) + [("cov", "target visible", "Purples")]
    for r, (key, title, cmap) in enumerate(rows):
        src = cov if key == "cov" else tabs[key]
        for ia in range(na):
            ax = axes[r, ia]
            im = ax.imshow(src[ia], vmin=0, vmax=1 if key != "fell" else 0.3, cmap=cmap, origin="upper")
            for ix in range(nx):
                for iy in range(ny):
                    v = src[ia, ix, iy]
                    if not np.isnan(v):
                        ax.text(iy, ix, f"{v:.2f}", ha="center", va="center", fontsize=8,
                                color="white" if v > (0.6 if key != "fell" else 0.18) else "black")  # fmt: skip
            ax.set_xticks(range(ny), yl, fontsize=7, rotation=30)
            ax.set_yticks(range(nx), xl, fontsize=7)
            ax.set_title(f"{title} | {APEX[ia]}", fontsize=10)
            if ia == 0:
                ax.set_ylabel("landing: distance from net (m)")
            if r == len(rows) - 1:
                ax.set_xlabel("landing y (m, + = our backhand side)")
        fig.colorbar(im, ax=axes[r, :].tolist(), shrink=0.8)
    name = Path(d["checkpoint"]).stem
    fig.suptitle(f"grid test {name}  ({d['finished']}/{d['rows']} balls, dt {d['dt_ms']} ms, 1 ball from the ready pose)")
    out = a.out or a.path.with_suffix(".png")
    fig.savefig(out, dpi=110)
    print("saved", out)
    for key, title, _ in METRICS:
        print(f"== {title} (rows: distance from net; cols: y bands), per apex tier")
        for ia in range(na):
            print(f"  {APEX[ia]}")
            for ix in range(nx):
                print("   ", xl[ix].rjust(9), " ".join("  -  " if np.isnan(v) else f"{v:.2f}" for v in tabs[key][ia, ix]))
    tot = {k: float((pr[k][done] / pr["done"][done]).mean()) for k, _, _ in METRICS}
    print("overall", {k: round(v, 3) for k, v in tot.items()}, "target visible", round(float(valid.mean()), 3))


if __name__ == "__main__":
    main()
