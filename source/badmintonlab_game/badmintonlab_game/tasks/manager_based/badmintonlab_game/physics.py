"""Physics step presets. ``play`` (a5e0780, 11-07) runs 1 ms physics, decimation 20 and actuator delays of
10-40 physics steps, i.e. the same 20 ms control step and 10-40 ms delay as the old 5 ms / 4 / 2-8 setup.

``apply_physics(env_cfg, dt_ms)`` keeps the control step at 20 ms and the delay at 10-40 ms for
``dt_ms`` in {1, 2, 4, 5}. The defaults in the config files are the play values (1 ms).
"""

from __future__ import annotations

CONTROL_MS = 20
DELAY_MS = (10, 40)


def apply_physics(env_cfg, dt_ms: int) -> None:
    dt_ms = int(dt_ms)
    if CONTROL_MS % dt_ms or DELAY_MS[0] % dt_ms:
        raise ValueError(f"dt_ms={dt_ms} must divide {CONTROL_MS} and {DELAY_MS[0]}")
    dec = CONTROL_MS // dt_ms
    env_cfg.sim.dt = dt_ms / 1000.0
    env_cfg.decimation = dec
    env_cfg.sim.render_interval = dec
    for name in ("racket_contact_0", "racket_contact_1", "racket_ground_contact_0", "racket_ground_contact_1"):
        sensor = getattr(env_cfg.scene, name, None)
        if sensor is not None:
            sensor.history_length = dec
    for robot_name in ("robot_0", "robot_1"):
        robot = getattr(env_cfg.scene, robot_name, None)
        if robot is None:
            continue
        for act in robot.actuators.values():
            if hasattr(act, "min_delay"):
                act.min_delay = DELAY_MS[0] // dt_ms
                act.max_delay = DELAY_MS[1] // dt_ms
