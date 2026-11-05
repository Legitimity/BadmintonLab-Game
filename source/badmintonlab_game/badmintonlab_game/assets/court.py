# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard-dimension badminton court & net assets and rule-judgement helpers.

The court geometry is generated directly from Python `Cuboid` spawn configs —
no URDF / USD / MJCF files are required. All shapes live under the per-env
namespace so each parallel environment owns its own court at its env origin.

Coordinate conventions
----------------------
* ``x`` runs along the length of the court (baseline-to-baseline).
* ``y`` runs along the width (sideline-to-sideline).
* ``z`` is up.
* The court center is at the env origin and the net plane is at ``x = 0``.
"""

from __future__ import annotations

import dataclasses

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class BadmintonCourtCfg:
    """Static dimensions & rule parameters of a standard badminton court."""

    length: float = 13.40
    doubles_width: float = 6.10
    singles_width: float = 5.18
    line_width: float = 0.04
    line_thickness: float = 0.005
    """Vertical thickness of court lines / floor (m). Kept small but >0 to avoid
    z-fighting with the ground terrain."""
    line_z_offset: float = 0
    """Vertical offset to lift line geometry above the ground plane."""

    net_x: float = 0.0
    net_center_height: float = 1.524
    net_edge_height: float = 1.55
    net_visual_height: float = 0.76
    net_thickness: float = 0.03
    net_collision_thickness: float = 0.08
    """Thickness of the invisible physics collider behind the net. Thicker than
    the visual band so fast players/rackets cannot tunnel through it."""

    post_size: float = 0.05
    post_height: float = 1.55

    show_floor: bool = True
    show_service_lines: bool = True
    show_center_line: bool = True

    use_singles_bounds: bool = True
    """If ``True``, ``is_out_of_bounds`` uses the singles sideline by default."""

    net_filter_on_collision_geometry: bool = False
    """If ``True``, author the (net, shuttle) filtered pair on the collider geometry
    before cloning, as the ``play`` evaluation package does. ``False`` keeps the
    original behaviour of the default training task (filter on the Xform)."""

    inclusive_net_crossing: bool = False
    """If ``True``, ``crossed_net`` also counts a step that ends exactly on the net
    plane, matching ``play``. ``False`` keeps the strict sign-change test."""

    prim_sub_path: str = "Court"

    @property
    def half_length(self) -> float:
        return 0.5 * self.length

    @property
    def doubles_half_width(self) -> float:
        return 0.5 * self.doubles_width

    @property
    def singles_half_width(self) -> float:
        return 0.5 * self.singles_width


def _line_cfg(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float] = (0.95, 0.95, 0.95),
) -> AssetBaseCfg:
    """Build a thin static cuboid used as a court line."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _solid_cfg(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    with_collision: bool,
    opacity: float = 1.0,
) -> AssetBaseCfg:
    """Build a static cuboid (optionally with collision) for floor / net / posts."""
    spawn = sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.9, opacity=opacity),
        collision_props=sim_utils.CollisionPropertiesCfg() if with_collision else None,
    )
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def spawn_net_collider(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the invisible net physics collider, filtered out of the shuttle's contacts.

    The net must physically block players and rackets, but the shuttle has to stay
    collision-free with it: an "into the net" fault is judged analytically from the
    shuttle's trajectory ("is_net_fault"), and a physical net would bounce the
    shuttle and invalidate that judgement. USD ``FilteredPairsAPI`` disables the
    ``(net, shuttle)`` collision pair. The relationship is authored inside the
    source environment before replication, so every cloned environment inherits it.
    """
    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid
    from pxr import UsdPhysics

    prim = spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    shuttle_path = prim.GetPath().GetParentPath().AppendChild("Shuttle")
    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(prim)
    filtered_pairs.CreateFilteredPairsRel().AddTarget(shuttle_path)
    return prim


@sim_utils.clone
def spawn_net_collider_on_geometry(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """``play``-aligned net collider: filter the shuttle on the actual collision prim.

    The cuboid's Xform is not a collider: ``CollisionAPI`` lives on its geometry
    descendant. Filtering is authored there, before the outer clone decorator
    replicates the source environment.
    """
    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid
    from pxr import UsdPhysics

    prim = spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    shuttle_path = prim.GetPath().GetParentPath().AppendChild("Shuttle")
    pending = [prim]
    while pending:
        child = pending.pop()
        if child.HasAPI(UsdPhysics.CollisionAPI):
            filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(child)
            filtered_pairs.CreateFilteredPairsRel().AddTarget(shuttle_path)
        pending.extend(child.GetChildren())
    return prim


def create_badminton_court(cfg: BadmintonCourtCfg, env_ns: str = "{ENV_REGEX_NS}") -> dict[str, AssetBaseCfg]:
    """Build the court geometry as a dict of ``AssetBaseCfg`` entries.

    The returned mapping can be attached to an ``InteractiveSceneCfg`` via
    ``setattr`` inside ``__post_init__`` — each value carries its own
    ``prim_path`` so the regex namespace is resolved at scene-construction time.
    """
    base = f"{env_ns}"
    half_len = cfg.half_length
    half_w = cfg.doubles_half_width
    half_sw = cfg.singles_half_width

    line_t = cfg.line_thickness
    line_w = cfg.line_width
    z_line = cfg.line_z_offset + 0.5 * line_t

    assets: dict[str, AssetBaseCfg] = {}

    if cfg.show_floor:
        assets["court_floor"] = _solid_cfg(
            prim_path=f"{base}/Floor",
            size=(cfg.length, cfg.doubles_width, line_t),
            pos=(0.0, 0.0, 0.5 * cfg.line_z_offset),
            color=(0.05, 0.30, 0.10),
            with_collision=False,
        )

    assets["court_baseline_pos"] = _line_cfg(
        prim_path=f"{base}/BaselinePos",
        size=(line_w, cfg.doubles_width, line_t),
        pos=(half_len, 0.0, z_line),
    )
    assets["court_baseline_neg"] = _line_cfg(
        prim_path=f"{base}/BaselineNeg",
        size=(line_w, cfg.doubles_width, line_t),
        pos=(-half_len, 0.0, z_line),
    )

    assets["court_doubles_sideline_pos"] = _line_cfg(
        prim_path=f"{base}/DoublesSidelinePos",
        size=(cfg.length, line_w, line_t),
        pos=(0.0, half_w, z_line),
    )
    assets["court_doubles_sideline_neg"] = _line_cfg(
        prim_path=f"{base}/DoublesSidelineNeg",
        size=(cfg.length, line_w, line_t),
        pos=(0.0, -half_w, z_line),
    )

    assets["court_singles_sideline_pos"] = _line_cfg(
        prim_path=f"{base}/SinglesSidelinePos",
        size=(cfg.length, line_w, line_t),
        pos=(0.0, half_sw, z_line),
    )
    assets["court_singles_sideline_neg"] = _line_cfg(
        prim_path=f"{base}/SinglesSidelineNeg",
        size=(cfg.length, line_w, line_t),
        pos=(0.0, -half_sw, z_line),
    )

    assets["court_net_line"] = _line_cfg(
        prim_path=f"{base}/NetLine",
        size=(line_w, cfg.doubles_width, line_t),
        pos=(cfg.net_x, 0.0, z_line),
    )

    if cfg.show_service_lines:
        assets["court_short_service_pos"] = _line_cfg(
            prim_path=f"{base}/ShortServicePos",
            size=(line_w, cfg.doubles_width, line_t),
            pos=(1.98, 0.0, z_line),
        )
        assets["court_short_service_neg"] = _line_cfg(
            prim_path=f"{base}/ShortServiceNeg",
            size=(line_w, cfg.doubles_width, line_t),
            pos=(-1.98, 0.0, z_line),
        )
        long_service_x = half_len - 0.76
        assets["court_long_service_pos"] = _line_cfg(
            prim_path=f"{base}/LongServicePos",
            size=(line_w, cfg.doubles_width, line_t),
            pos=(long_service_x, 0.0, z_line),
        )
        assets["court_long_service_neg"] = _line_cfg(
            prim_path=f"{base}/LongServiceNeg",
            size=(line_w, cfg.doubles_width, line_t),
            pos=(-long_service_x, 0.0, z_line),
        )

    if cfg.show_center_line:
        cl_pos = 0.5 * (6.7 + 1.98)
        cl_len = 6.7 - 1.98
        assets["court_center_line_pos"] = _line_cfg(
            prim_path=f"{base}/CenterLinePos",
            size=(cl_len, line_w, line_t),
            pos=(cl_pos, 0.0, z_line),
        )
        assets["court_center_line_neg"] = _line_cfg(
            prim_path=f"{base}/CenterLineNeg",
            size=(cl_len, line_w, line_t),
            pos=(-cl_pos, 0.0, z_line),
        )

    net_center_z = cfg.net_edge_height - 0.5 * cfg.net_visual_height
    assets["net"] = _solid_cfg(
        prim_path=f"{base}/Net",
        size=(cfg.net_thickness * 0.5, cfg.doubles_width, cfg.net_visual_height),
        pos=(cfg.net_x, 0.0, net_center_z),
        color=(0.08, 0.08, 0.08),
        with_collision=False,
    )

    net_collider = _solid_cfg(
        prim_path=f"{base}/NetCollision",
        size=(cfg.net_collision_thickness, cfg.doubles_width, cfg.net_visual_height),
        pos=(cfg.net_x, 0.0, net_center_z),
        color=(0.08, 0.08, 0.08),
        with_collision=True,
    )
    net_collider.spawn = dataclasses.replace(
        net_collider.spawn,
        visible=False,
        func=spawn_net_collider_on_geometry if cfg.net_filter_on_collision_geometry else spawn_net_collider,
    )
    assets["net_collider"] = net_collider

    post_z = 0.5 * cfg.post_height
    assets["net_post_pos"] = _solid_cfg(
        prim_path=f"{base}/NetPostPos",
        size=(cfg.post_size, cfg.post_size, cfg.post_height),
        pos=(cfg.net_x, half_w, post_z),
        color=(0.10, 0.10, 0.10),
        with_collision=True,
    )
    assets["net_post_neg"] = _solid_cfg(
        prim_path=f"{base}/NetPostNeg",
        size=(cfg.post_size, cfg.post_size, cfg.post_height),
        pos=(cfg.net_x, -half_w, post_z),
        color=(0.10, 0.10, 0.10),
        with_collision=True,
    )

    return assets


def net_height_at_y(y: torch.Tensor, cfg: BadmintonCourtCfg) -> torch.Tensor:
    """Net top-edge height as a function of lateral position ``y``.

    Linearly interpolates between ``net_center_height`` (at ``y=0``) and
    ``net_edge_height`` (at the doubles sideline). Beyond the sideline the
    height is clamped to the edge value.
    """
    half_w = max(cfg.doubles_half_width, 1e-6)
    ratio = torch.clamp(y.abs() / half_w, max=1.0)
    return cfg.net_center_height + (cfg.net_edge_height - cfg.net_center_height) * ratio


def is_out_of_bounds(
    shuttle_pos: torch.Tensor, cfg: BadmintonCourtCfg, use_singles: bool | None = None
) -> torch.Tensor:
    """Batched out-of-bounds check.

    Args:
        shuttle_pos: ``(N, 3)`` tensor of shuttle positions in the court-local
            frame (i.e. world position minus env origin).
        cfg: court configuration.
        use_singles: override the default in ``cfg.use_singles_bounds``.

    Returns:
        Boolean tensor of shape ``(N,)``.
    """
    if use_singles is None:
        use_singles = cfg.use_singles_bounds
    half_len = cfg.half_length
    half_w = cfg.singles_half_width if use_singles else cfg.doubles_half_width
    return (shuttle_pos[:, 0].abs() > half_len) | (shuttle_pos[:, 1].abs() > half_w)


def crossed_net(prev_shuttle_pos: torch.Tensor, curr_shuttle_pos: torch.Tensor, cfg: BadmintonCourtCfg) -> torch.Tensor:
    """True for envs whose shuttle crossed the net plane within the net width."""
    dx_prev = prev_shuttle_pos[:, 0] - cfg.net_x
    dx_curr = curr_shuttle_pos[:, 0] - cfg.net_x
    if cfg.inclusive_net_crossing:
        crossed_x = ((dx_prev < 0.0) & (dx_curr >= 0.0)) | ((dx_prev > 0.0) & (dx_curr <= 0.0))
    else:
        crossed_x = (dx_prev * dx_curr) < 0.0

    dx = curr_shuttle_pos[:, 0] - prev_shuttle_pos[:, 0]
    safe_dx = torch.where(dx.abs() > 1e-8, dx, torch.ones_like(dx))
    t = (cfg.net_x - prev_shuttle_pos[:, 0]) / safe_dx
    t = torch.clamp(t, 0.0, 1.0)
    y_cross = prev_shuttle_pos[:, 1] + t * (curr_shuttle_pos[:, 1] - prev_shuttle_pos[:, 1])
    within_net_width = y_cross.abs() <= cfg.doubles_half_width

    return crossed_x & within_net_width


def is_net_fault(
    prev_shuttle_pos: torch.Tensor, curr_shuttle_pos: torch.Tensor, cfg: BadmintonCourtCfg
) -> torch.Tensor:
    """Detect a 'into the net' fault via linear interpolation of the crossing.

    If the shuttle crosses ``x = net_x`` within a step we compute the
    interpolated ``(y_cross, z_cross)`` and compare ``z_cross`` against the
    local net top-edge height. This is independent of physics contact, so
    sub-step penetration / tunneling cannot hide a net fault.
    """
    crossed = crossed_net(prev_shuttle_pos, curr_shuttle_pos, cfg)
    dx = curr_shuttle_pos[:, 0] - prev_shuttle_pos[:, 0]
    safe_dx = torch.where(dx.abs() > 1e-8, dx, torch.ones_like(dx))
    t = (cfg.net_x - prev_shuttle_pos[:, 0]) / safe_dx
    t = torch.clamp(t, 0.0, 1.0)
    y_cross = prev_shuttle_pos[:, 1] + t * (curr_shuttle_pos[:, 1] - prev_shuttle_pos[:, 1])
    z_cross = prev_shuttle_pos[:, 2] + t * (curr_shuttle_pos[:, 2] - prev_shuttle_pos[:, 2])
    return crossed & (z_cross < net_height_at_y(y_cross, cfg))
