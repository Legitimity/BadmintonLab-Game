"""CPU checks of the serve task wiring with Isaac stubbed out.

Isaac Lab's ``configclass`` is pure Python, so the real decorator is used; only
modules that need Kit (``pxr``, ``omni``, physics) are stubbed. This proves the
config classes assemble, event names do not collide when play merges both sides,
the observation layout is 95-D, and the reset pose writes what it should.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/serve"
ISAACLAB = Path("D:/rl-badminton-train/IsaacLab/source/isaaclab")


def _real_configclass():
    """Load Isaac Lab's configclass module by file: the package __init__ pulls in pxr."""
    util = ISAACLAB / "isaaclab/utils"
    pkg = types.ModuleType("_il_utils")
    pkg.__path__ = [str(util)]
    sys.modules["_il_utils"] = pkg
    for name in ("string", "dict", "configclass"):
        spec = importlib.util.spec_from_file_location(f"_il_utils.{name}", util / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_il_utils.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["_il_utils.configclass"].configclass


@pytest.fixture(scope="module")
def player_cfg():
    configclass = _real_configclass()

    class Term:
        def __init__(self, func=None, params=None, mode=None, **kw):
            self.func, self.params, self.mode = func, params or {}, mode

    class SceneEntityCfg:
        def __init__(self, name, joint_names=None, body_names=None):
            self.name, self.joint_names, self.body_names = name, joint_names, body_names

    stubs = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.assets": types.SimpleNamespace(Articulation=object, RigidObject=object),
        "isaaclab.envs": types.SimpleNamespace(),
        "isaaclab.envs.mdp": types.SimpleNamespace(
            **{n: (lambda *a, **k: None) for n in ("base_ang_vel", "base_lin_vel", "joint_pos_rel", "joint_vel_rel", "last_action", "projected_gravity", "reset_root_state_uniform")}
        ),
        "isaaclab.managers": types.SimpleNamespace(EventTermCfg=Term, ObservationGroupCfg=object, ObservationTermCfg=Term, SceneEntityCfg=SceneEntityCfg),
        "isaaclab.utils": types.SimpleNamespace(configclass=configclass),
        "isaaclab.utils.math": types.SimpleNamespace(quat_apply=None, quat_apply_inverse=None),
    }
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("player_cfg_under_test", SERVE / "player_cfg.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


DIMS = dict(
    robot_heading_court=2, robot_pos_court=3, shuttle_pos_b=3, shuttle_vel_b=3, racket_pos_b=3,
    shuttle_from_racket_b=3, base_lin_vel=3, base_ang_vel=3, projected_gravity=3, joint_pos=23, joint_vel=23, last_action=23,
)


def test_actor_obs_is_95_and_has_no_opponent(player_cfg):
    for side in (0, 1):
        cfg = player_cfg.make_obs_cfg(side)()
        names = [k for k, v in vars(cfg).items() if hasattr(v, "func")]
        assert set(names) == set(DIMS), names
        assert sum(DIMS[n] for n in names) == player_cfg.OBS_DIM == 95
        me = f"robot_{side}"
        for n in names:
            p = getattr(cfg, n).params
            asset = p.get("asset_cfg")
            assert asset is None or asset.name == me, (side, n, asset.name)
        assert getattr(cfg, "last_action").params["action_name"] == f"player_{side}"


def test_event_names_do_not_collide_when_play_merges_sides(player_cfg):
    e0, e1 = player_cfg.make_events_cfg(0), player_cfg.make_events_cfg(1)
    configclass = sys.modules["_il_utils.configclass"].configclass

    @configclass
    class Merged(e1, e0):  # play: class DualEventsCfg(Player0EventsCfg, Player1EventsCfg) under @configclass
        pass

    m = Merged()
    names = sorted(f.name for f in __import__("dataclasses").fields(m) if f.name.startswith("reset_player_"))
    assert sorted(names) == ["reset_player_0_joints", "reset_player_0_root", "reset_player_1_joints", "reset_player_1_root"]
    assert m.reset_player_0_root.params["asset_cfg"].name == "robot_0"
    assert m.reset_player_1_root.params["asset_cfg"].name == "robot_1"


def test_side1_is_a_rotation_not_a_mirror(player_cfg):
    """A mirror would move the racket to the left hand side of the court geometry."""
    r0 = player_cfg.make_events_cfg(0)().reset_player_0_root.params["pose_range"]
    r1 = player_cfg.make_events_cfg(1)().reset_player_1_root.params["pose_range"]
    assert r1["x"][0] == -r0["x"][0] and r1["y"][0] == -r0["y"][0]
    assert math.isclose(r1["yaw"][0], r0["yaw"][0] + math.pi)
    for r in (r0, r1):
        assert all(lo == hi for lo, hi in r.values()), "fixed pose: no randomisation"


def test_reset_pose_writes_listed_joints_and_clamps(player_cfg):
    names = ["Waist", "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw", "Left_Knee_Pitch"]

    class Robot:
        def __init__(self):
            self.data = types.SimpleNamespace(
                default_joint_pos=torch.tensor([[0.0, 0.2, 1.3, 0.0, 0.5, 0.4]]).repeat(3, 1),
                soft_joint_pos_limits=torch.tensor([[-1.0, 1.0]]).repeat(3, 6, 1),
            )
            self.written = None

        def find_joints(self, keys, preserve_order=False):
            return [names.index(k) for k in keys], keys

        def write_joint_state_to_sim(self, pos, vel, env_ids=None):
            self.written = (pos, vel, env_ids)

    robot = Robot()
    env = types.SimpleNamespace(scene={"robot_0": robot})
    ids = torch.tensor([0, 2])
    pose = {"Right_Shoulder_Pitch": -0.5, "Right_Shoulder_Roll": 2.0, "Waist": 0.3}
    player_cfg.reset_joints_to_pose(env, ids, types.SimpleNamespace(name="robot_0"), pose)
    pos, vel, env_ids = robot.written
    assert torch.equal(env_ids, ids) and torch.all(vel == 0)
    assert torch.allclose(pos[0], torch.tensor([0.3, -0.5, 1.0, 0.0, 0.5, 0.4]))  # Roll clamped to the soft limit
