"""Test-only combined player (11-11): serve expert for our own opening drop, receive expert otherwise.

The switch is a fixed rule on the observation (not learned): serve mode while the shuttle is within
``r_serve`` m of the root horizontally and its horizontal speed (relative to the root) is below
``v_serve`` m/s and no incoming-ball target is set (intercept valid flag 0), i.e. the static opening drop
on our side. Used to measure multi-shot rallies in play
before the distillation; the final submission stays a single distilled MLP.

Usage (CPU):
  python scripts/tools/make_combo.py --serve D:/.../p3_final --recv D:/.../exports/s3_s3_1500_10748 --out D:/.../exports/combo_x
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

# 100-dim receive obs = 95-dim serve obs + 5 intercept terms. Serve obs layout:
# heading 2, robot pos 3, shuttle_pos_b 3, shuttle_vel_b 3, ...
SH_POS = slice(5, 8)
SH_VEL = slice(8, 11)


class Combo(torch.nn.Module):
    def __init__(self, serve: torch.nn.Module, recv: torch.nn.Module, r_serve: float, v_serve: float):
        super().__init__()
        self.serve = serve
        self.recv = recv
        self.r_serve = r_serve
        self.v_serve = v_serve

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        p = obs[:, 5:7]
        v = obs[:, 8:10]
        no_target = obs[:, 99] < 0.5  # intercept valid flag: an incoming ball always sets it
        serving = no_target & (torch.linalg.norm(p, dim=-1) < self.r_serve) & (torch.linalg.norm(v, dim=-1) < self.v_serve)
        a_s = self.serve(obs[:, :95])
        a_r = self.recv(obs)
        return torch.where(serving.unsqueeze(-1), a_s, a_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", type=Path, required=True)
    ap.add_argument("--recv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--r_serve", type=float, default=2.0)
    ap.add_argument("--v_serve", type=float, default=1.5)
    a = ap.parse_args()
    serve = torch.jit.load(str(a.serve / "policy.pt"), map_location="cpu").eval()
    recv = torch.jit.load(str(a.recv / "policy.pt"), map_location="cpu").eval()
    m = torch.jit.script(Combo(serve, recv, a.r_serve, a.v_serve).eval())
    a.out.mkdir(parents=True, exist_ok=True)
    m.save(str(a.out / "policy.pt"))
    shutil.copy2(a.recv / "player_cfg.py", a.out / "player_cfg.py")
    # Self-check on random observations: both branches reproduce the experts.
    x = torch.randn(64, 100)
    x[:32, 5:7] = 0.5
    x[:32, 8:10] = 0.1
    x[:32, 99] = 0.0
    x[32:, 5:7] = 4.0
    y = torch.jit.load(str(a.out / "policy.pt")).eval()(x)
    e_s = (y[:32] - serve(x[:32, :95])).abs().max().item()
    e_r = (y[32:] - recv(x[32:])).abs().max().item()
    meta = dict(serve=str(a.serve), recv=str(a.recv), r_serve=a.r_serve, v_serve=a.v_serve, err_serve=e_s, err_recv=e_r)
    (a.out / "combo_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))
    assert e_s < 1e-5 and e_r < 1e-5


if __name__ == "__main__":
    main()
