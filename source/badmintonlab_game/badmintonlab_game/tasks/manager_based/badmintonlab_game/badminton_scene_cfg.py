# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two robots, shuttle, court sensors, ground, and lighting for self-play."""

import dataclasses

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from badmintonlab_game.assets.court import BadmintonCourtCfg
from badmintonlab_game.assets.robots.booster import BOOSTER_T1_CFG

from .mdp.events import spawn_urdf_with_racket_material

_COURT_CFG = BadmintonCourtCfg()
_ROBOT_RESET_Y_RANGE = (-0.15 * _COURT_CFG.singles_half_width, 0.15 * _COURT_CFG.singles_half_width)


def _racket_sensor(side: int, *, ground: bool = False) -> ContactSensorCfg:
    """Create a racket sensor filtered to either the shuttle or the ground."""
    target = "/World/ground/terrain/GroundPlane/CollisionPlane" if ground else "{ENV_REGEX_NS}/Shuttle"
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot{side}/racket_link",
        history_length=20,  # = decimation (1 ms physics)
        track_air_time=False,
        force_threshold=1.0,
        filter_prim_paths_expr=[target],
    )


@configclass
class DualAgentSceneCfg(InteractiveSceneCfg):
    """Robot, shuttle, contact sensors, ground, and lighting shared by dual play."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot_0: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot0")
    robot_1: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    shuttle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shuttle",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), roughness=0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.05,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00519),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.4,
                restitution=0.05,
                friction_combine_mode="average",
                restitution_combine_mode="min",
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 0.0, 1.55)),
    )
    racket_contact_0 = _racket_sensor(0)
    racket_contact_1 = _racket_sensor(1)
    racket_ground_contact_0 = _racket_sensor(0, ground=True)
    racket_ground_contact_1 = _racket_sensor(1, ground=True)
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    def __post_init__(self):
        super().__post_init__()
        for robot in (self.robot_0, self.robot_1):
            robot.spawn = dataclasses.replace(
                robot.spawn, merge_fixed_joints=False, func=spawn_urdf_with_racket_material
            )
