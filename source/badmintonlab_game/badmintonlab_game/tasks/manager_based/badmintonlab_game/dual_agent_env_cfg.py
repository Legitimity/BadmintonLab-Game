import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from badmintonlab_game.assets.robots.booster import T1_ACTION_SCALE

from . import mdp
from .badminton_scene_cfg import (
    _COURT_CFG,
    _ROBOT_RESET_Y_RANGE,
    DualAgentSceneCfg,
)


@configclass
class DualActionsCfg:
    player_0 = mdp.JointPositionActionCfg(
        asset_name="robot_0", joint_names=[".*"], use_default_offset=True, scale=T1_ACTION_SCALE
    )
    player_1 = mdp.JointPositionActionCfg(
        asset_name="robot_1", joint_names=[".*"], use_default_offset=True, scale=T1_ACTION_SCALE
    )


@configclass
class DualObservationsCfg:
    @configclass
    class Player0Cfg(ObsGroup):
        robot_heading_court = ObsTerm(
            func=mdp.canonical_robot_heading_court, params={"asset_cfg": SceneEntityCfg("robot_0"), "side": 0}
        )
        robot_pos_court = ObsTerm(
            func=mdp.canonical_robot_pos_court, params={"asset_cfg": SceneEntityCfg("robot_0"), "side": 0}
        )
        shuttle_pos_b = ObsTerm(
            func=mdp.shuttle_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot_0"), "shuttle_cfg": SceneEntityCfg("shuttle")},
        )
        shuttle_vel_b = ObsTerm(
            func=mdp.shuttle_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot_0"), "shuttle_cfg": SceneEntityCfg("shuttle")},
        )
        opponent_heading_court = ObsTerm(
            func=mdp.canonical_robot_heading_court, params={"asset_cfg": SceneEntityCfg("robot_1"), "side": 0}
        )
        opponent_pos_court = ObsTerm(
            func=mdp.canonical_robot_pos_court, params={"asset_cfg": SceneEntityCfg("robot_1"), "side": 0}
        )
        opponent_projected_gravity = ObsTerm(
            func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}
        )
        opponent_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "player_0"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class Player1Cfg(ObsGroup):
        robot_heading_court = ObsTerm(
            func=mdp.canonical_robot_heading_court, params={"asset_cfg": SceneEntityCfg("robot_1"), "side": 1}
        )
        robot_pos_court = ObsTerm(
            func=mdp.canonical_robot_pos_court, params={"asset_cfg": SceneEntityCfg("robot_1"), "side": 1}
        )
        shuttle_pos_b = ObsTerm(
            func=mdp.shuttle_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot_1"), "shuttle_cfg": SceneEntityCfg("shuttle")},
        )
        shuttle_vel_b = ObsTerm(
            func=mdp.shuttle_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot_1"), "shuttle_cfg": SceneEntityCfg("shuttle")},
        )
        opponent_heading_court = ObsTerm(
            func=mdp.canonical_robot_heading_court, params={"asset_cfg": SceneEntityCfg("robot_0"), "side": 1}
        )
        opponent_pos_court = ObsTerm(
            func=mdp.canonical_robot_pos_court, params={"asset_cfg": SceneEntityCfg("robot_0"), "side": 1}
        )
        opponent_projected_gravity = ObsTerm(
            func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}
        )
        opponent_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "player_1"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    player_0: Player0Cfg = Player0Cfg()
    player_1: Player1Cfg = Player1Cfg()


@configclass
class DualEventsCfg:
    reset_robot_0_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_0"),
            "pose_range": {
                "x": (-0.5 * _COURT_CFG.half_length - 0.35, -0.5 * _COURT_CFG.half_length + 0.35),
                "y": _ROBOT_RESET_Y_RANGE,
                "z": (-0.01, 0.01),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.12, 0.12),
            },
            "velocity_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_robot_1_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1"),
            "pose_range": {
                "x": (0.5 * _COURT_CFG.half_length - 0.35, 0.5 * _COURT_CFG.half_length + 0.35),
                "y": _ROBOT_RESET_Y_RANGE,
                "z": (-0.01, 0.01),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (math.pi - 0.12, math.pi + 0.12),
            },
            "velocity_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_robot_0_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", joint_names=[".*"]),
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_robot_1_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", joint_names=[".*"]),
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_shuttle_pose = EventTerm(
        func=mdp.reset_shuttle_drop_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("shuttle"),
            "canonical_position": (-0.5 * _COURT_CFG.half_length + 0.8, 0.0, 2.2),
        },
    )


@configclass
class Player0RewardsCfg:
    """Default duel objective: win/loss plus zero-at-ideal task and control costs."""

    anti_fall = RewTerm(
        func=mdp.agent_root_height_violation_cost,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot_0"), "minimum_height": 0.55, "height_sigma": 10.0},
    )
    root_orientation = RewTerm(
        func=mdp.agent_root_orientation_violation_cost,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot_0"),
            "side": 0,
            "max_rp_angle_deg": 45.0,
            "max_yaw_angle_deg": 90.0,
            "rad_sigma": 10.0,
        },
    )
    racket_ground = RewTerm(
        func=mdp.racket_ground_support_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("racket_ground_contact_0", body_names=["racket_link"]),
            "force_threshold": 3.0,
            "force_scale": 10.0,
            "max_penalty": 1.0,
        },
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=[".*"])}
    )
    action_rate = RewTerm(func=mdp.agent_action_rate_l2, weight=-0.1, params={"action_name": "player_0"})
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot_0", body_names=["left_foot_link", "right_foot_link"])},
    )
    interval_approach = RewTerm(
        func=mdp.agent_interval_approach_cost,
        weight=-0.25,
        params={
            "side": 0,
            "min_height": 0.5,
            "max_height": 2.0,
            "full_reward_radius": 0.1,
            "distance_sigma": 0.45,
        },
    )
    outgoing_quality = RewTerm(
        func=mdp.agent_outgoing_shot_quality_cost,
        weight=-0.25,
        params={
            "shuttle_cfg": SceneEntityCfg("shuttle"),
            "side": 0,
            "target_speed": 10.0,
            "minimum_net_clearance_height": 1.55,
            "direction_sigma": 0.15,
        },
    )
    rally_result = RewTerm(func=mdp.agent_rally_result_reward, weight=10.0, params={"side": 0})


@configclass
class Player1RewardsCfg:
    """Player-1 mirror of the default duel objective."""

    anti_fall = RewTerm(
        func=mdp.agent_root_height_violation_cost,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot_1"), "minimum_height": 0.55, "height_sigma": 10.0},
    )
    root_orientation = RewTerm(
        func=mdp.agent_root_orientation_violation_cost,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot_1"),
            "side": 1,
            "max_rp_angle_deg": 45.0,
            "max_yaw_angle_deg": 90.0,
            "rad_sigma": 10.0,
        },
    )
    racket_ground = RewTerm(
        func=mdp.racket_ground_support_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("racket_ground_contact_1", body_names=["racket_link"]),
            "force_threshold": 3.0,
            "force_scale": 10.0,
            "max_penalty": 1.0,
        },
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=[".*"])}
    )
    action_rate = RewTerm(func=mdp.agent_action_rate_l2, weight=-0.1, params={"action_name": "player_1"})
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot_1", body_names=["left_foot_link", "right_foot_link"])},
    )
    interval_approach = RewTerm(
        func=mdp.agent_interval_approach_cost,
        weight=-0.25,
        params={
            "side": 1,
            "min_height": 0.5,
            "max_height": 2.0,
            "full_reward_radius": 0.1,
            "distance_sigma": 0.45,
        },
    )
    outgoing_quality = RewTerm(
        func=mdp.agent_outgoing_shot_quality_cost,
        weight=-0.25,
        params={
            "shuttle_cfg": SceneEntityCfg("shuttle"),
            "side": 1,
            "target_speed": 10.0,
            "minimum_net_clearance_height": 1.55,
            "direction_sigma": 0.15,
        },
    )
    rally_result = RewTerm(func=mdp.agent_rally_result_reward, weight=10.0, params={"side": 1})


@configclass
class DualTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    rally_resolved = DoneTerm(func=mdp.rally_resolved)


@configclass
class DualAgentBadmintonEnvCfg(ManagerBasedRLEnvCfg):
    scene: DualAgentSceneCfg = DualAgentSceneCfg(num_envs=2048, env_spacing=15.0)
    observations: DualObservationsCfg = DualObservationsCfg()
    actions: DualActionsCfg = DualActionsCfg()
    events: DualEventsCfg = DualEventsCfg()
    rewards: Player0RewardsCfg = Player0RewardsCfg()
    player_1_rewards: Player1RewardsCfg = Player1RewardsCfg()
    terminations: DualTerminationsCfg = DualTerminationsCfg()
    shuttle_mass = 0.00519
    shuttle_drag_k = 0.00110
    gravity_z = -9.81
    shuttle_landing_detection_height = 0.03
    hit_contact_threshold = 1.0
    return_history_window = 8

    process_reward_cap: int = 4

    def __post_init__(self):
        self.decimation = 20  # play a5e0780: 1 ms physics, 50 Hz control
        self.episode_length_s = 10.0
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        self.viewer.origin_type = "world"
        self.viewer.eye = (7.0, -9.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)
        self.court = _COURT_CFG
        from badmintonlab_game.assets.court import create_badminton_court

        for asset_name, asset_cfg in create_badminton_court(self.court).items():
            setattr(self.scene, asset_name, asset_cfg)
