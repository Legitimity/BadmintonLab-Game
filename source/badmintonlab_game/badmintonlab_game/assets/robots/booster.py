from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from badmintonlab_game.assets.robots import actuator
from badmintonlab_game.assets.robots.actuator import BoosterDelayedPDActuatorCfg

PROJECT_ROOT = Path(__file__).resolve().parents[5]


BOOSTER_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        usd_dir=str(PROJECT_ROOT / "tmp" / "usd"),
        fix_base=False,
        asset_path=str(PROJECT_ROOT / "robots" / "T1" / "T1_23dof.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Hip_Pitch": -0.2,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            booster_joint_cfgs=actuator.BoosterJointE4310(natural_freq=4.0, damping_ratio=1.5),
        ),
        "waist": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=["Waist"],
            booster_joint_cfgs=actuator.BoosterJointE6408(natural_freq=4.0, damping_ratio=1.5),
        ),
        "legs": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            booster_joint_cfgs={
                ".*_Hip_Pitch": actuator.BoosterJointE8112(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Roll": actuator.BoosterJointE6408(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Yaw": actuator.BoosterJointE6408(natural_freq=4.0, damping_ratio=1.5),
                ".*_Knee_Pitch": actuator.BoosterJointE8116(natural_freq=4.0, damping_ratio=1.5),
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Ankle_Pitch",
                ".*_Ankle_Roll",
            ],
            booster_joint_cfgs={
                ".*_Ankle_Pitch": actuator.BoosterT1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4315(),
                    serial_index=0,
                    natural_freq=4.0,
                    damping_ratio=1.5,
                    effort_ratio=(1.7, 0.4),
                    velocity_ratio=(0.6, 1.0),
                    armature_ratio=(2, 0.6),
                    knee_point_velocity_ratio=(1.0, 1.0),
                ),
                ".*_Ankle_Roll": actuator.BoosterT1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4315(),
                    serial_index=1,
                    natural_freq=4.0,
                    damping_ratio=1.5,
                    effort_ratio=(1.7, 0.4),
                    velocity_ratio=(0.6, 1.0),
                    armature_ratio=(2, 0.6),
                    knee_point_velocity_ratio=(1.0, 1.0),
                ),
            },
        ),
        "head": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*Head.*"],
            booster_joint_cfgs=actuator.BoosterJointDM4310(natural_freq=4.0, damping_ratio=1.5),
        ),
    },
)

ACTUATOR_DELAY_S = (0.010, 0.040)  # min / max actuator delay in seconds (play: 2/8 steps at 5 ms, 10/40 at 1 ms)


def actuator_delay_steps(physics_dt: float) -> tuple[int, int]:
    """Delay bounds in physics steps for a physics step of ``physics_dt`` seconds (always 10-40 ms)."""
    return round(ACTUATOR_DELAY_S[0] / physics_dt), round(ACTUATOR_DELAY_S[1] / physics_dt)


def set_actuator_delays(robot_cfg: ArticulationCfg, physics_dt: float) -> None:
    """Rewrites every actuator's ``min_delay`` / ``max_delay`` for ``physics_dt``."""
    lo, hi = actuator_delay_steps(physics_dt)
    for name, act in list(robot_cfg.actuators.items()):
        robot_cfg.actuators[name] = act.replace(min_delay=lo, max_delay=hi)


T1_ACTION_SCALE = {}
for a in BOOSTER_T1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    d = a.damping
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    if not isinstance(d, dict):
        d = {n: d for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            T1_ACTION_SCALE[n] = 0.5 * e[n] / s[n]
