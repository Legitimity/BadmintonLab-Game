"""Configuration of the ``play``-aligned serve task (``BadmintonLab-Game-Serve``).

P0 only aligns the rules with the evaluator; observations, rewards and
initial poses are still the default task's.  Later phases change them here,
never in ``dual_agent_env_cfg.py``.
"""

from isaaclab.utils import configclass

from badmintonlab_game.assets.court import create_badminton_court

from .badminton_scene_cfg import _COURT_CFG
from .dual_agent_env_cfg import DualAgentBadmintonEnvCfg

_SERVE_COURT_CFG = _COURT_CFG.replace(net_filter_on_collision_geometry=True, inclusive_net_crossing=True)


@configclass
class ServeBadmintonEnvCfg(DualAgentBadmintonEnvCfg):
    max_serve_height: float = 1.15
    """Serve fault: the server's first racket contact with the shuttle centre above this height (m)."""

    def __post_init__(self):
        super().__post_init__()
        self.court = _SERVE_COURT_CFG
        for asset_name, asset_cfg in create_badminton_court(self.court).items():
            setattr(self.scene, asset_name, asset_cfg)
