"""Receive-expert training environment, stage S1 footwork (P5 V3.1, arXiv 2511.11218 v4 stage 1).

Robot 0 is the only learner (the receiver). Robot 1 is parked off-court and holds the ready pose.
Each episode feeds up to ``balls_per_episode`` shuttles from the feeder table; the robot is not reset
between balls. Rules are the ``play`` port in ``serve/rules.py`` (inherited from the serve env).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_apply_inverse

from ..dual_agent_env import DualAgentBadmintonEnv
from ..serve.serve_env import ServeBadmintonEnv
from . import core
from . import player_cfg as recv_player_cfg

UPPER_JOINTS = (
    "Waist",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
)
DENSE = ("track", "face_front", "base_height", "base_orientation", "feet_air_time", "step_height", "foot_posture",
         "no_double_flight", "feet_slide", "acc_reg", "hold_ready", "approach",
         "racket_style", "torque", "energy")  # fmt: skip
EVENTS = ("touchdown_speed", "gait_symmetry", "fall", "swing", "contact")
TERMS = DENSE + EVENTS
S1_WEIGHTS = dict(
    track=4.0, approach=2.0, face_front=0.5, base_height=-10.0, base_orientation=-2.0, feet_air_time=1.0, step_height=0.5,
    foot_posture=-0.5, no_double_flight=-1.0, feet_slide=-0.1, acc_reg=-2.5e-7, hold_ready=0.3,
    touchdown_speed=-0.5, gait_symmetry=-0.2, fall=-10.0,
)  # fmt: skip
# S2 (P5 V3.1 5.2, paper IV-A stage 2): lower the regional tracking, add the hit-instant reward (20, six per
# episode), the racket style term (0.2), a contact bonus, torque / energy costs. 11-08 choices: approach 2 -> 1;
# track 4 -> 2 (V3.3: with 1.0 the probe's station rate fell 0.73 -> 0.46 and far-ball hits 0.16 -> 0.05).
S2_WEIGHTS = dict(
    S1_WEIGHTS, track=2.0, approach=1.0, swing=20.0, contact=2.0, racket_style=0.2, torque=-2.0e-5, energy=-2.0e-4,
)  # fmt: skip
SWING_UPPER = ("Waist", "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw")
STAT_KEYS = ("balls", "station_ok", "face_ok", "d_sum", "head_sum", "touch", "episodes", "falls", "vel_sum", "vel_n", "fall_low", "fall_tilt", "fall_step_sum", "timeouts",
             "ended", "hit", "ret_in", "ret_net", "ret_out", "epos_sum", "eori_sum", "vn_sum", "swing_sum")  # fmt: skip
N_LVL = 6  # V4 speed levels (legacy distance levels use 0-3)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


class ReceiveBadmintonEnv(ServeBadmintonEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        # 11-09 S1 V4: wide table with a free hitting height, levels by required speed, return-home gaps.
        self.v4 = bool(getattr(cfg, "feeder_v4", False))
        if self.v4:
            self.table = core.build_feeder_table_v4(cfg.feeder_rows, seed=cfg.feeder_seed, device=dev)
            self._v_sorted = core.required_speed(self.table.d_need, self.table.t_arr)
        else:
            self.table = core.build_feeder_table(cfg.feeder_rows, seed=cfg.feeder_seed, device=dev)
        self.level = 0
        self._bounds = self._level_bounds(self.level)
        self.ready_xy = torch.tensor(core.READY_ROOT_XY, device=dev)
        self.weights = {k: float(S1_WEIGHTS.get(k, 0.0)) for k in TERMS}
        self.balls_per_episode = int(cfg.balls_per_episode)
        self.gap_steps = int(round(cfg.ball_gap_s / self.step_dt))
        self.ball_timeout_steps = int(round(cfg.ball_timeout_s / self.step_dt))
        z = lambda *s, **k: torch.zeros(*s, device=dev, **k)  # noqa: E731
        self.ball_active = z(n, dtype=torch.bool)
        self.ball_steps = z(n, dtype=torch.long)
        self.gap_left = z(n, dtype=torch.long)
        self.balls_done = z(n, dtype=torch.long)
        self.arr_step = z(n, dtype=torch.long)
        self.station = z(n, 2)
        self.int_pos = z(n, 3)
        self.t_arr = z(n)
        self.ball_touched = z(n, dtype=torch.bool)
        self.fell = z(n, dtype=torch.bool)
        self.values = {k: z(n) for k in TERMS}
        self.stats = z(len(STAT_KEYS))
        self.arrival_d = []  # per-iteration list of tensors, popped by the runner
        self.arrival_epos = []
        self.ball_lvl = z(n, dtype=torch.long)
        self.lvl_stats = z(N_LVL, 5)  # arrivals, station_ok, d_sum, ended, hit
        # 11-08 V3.4: real serves recorded in play, mixed with the synthetic feeder.
        self.real = None
        self.real_frac = 0.0
        self.real_src_id = None  # (rows,) style index of each real row
        self.real_names = []
        self.ball_src = z(n, dtype=torch.long)  # 0 synthetic, 1 + style index for real serves
        self.src_stats = z(8, 4)  # per source: arrivals, station_ok, ended, hit
        # 11-09: same per source, first ball of an episode only (starts at the ready pose, like a play rally).
        self.src_stats_first = z(8, 4)
        self.approach_cap = 0.8
        self.sigma_p, self.sigma_r, self.sigma_v = 2.0, 8.0, 3.0
        self.n_star = core.target_normal(50.0, device=dev)
        self.v_cap = float(cfg.swing_v_cap)
        self._hit_now = z(n, dtype=torch.bool)
        self._e_pos = z(n)
        self._e_ori = z(n)
        self._v_n = z(n)
        self._axis_y = z(n, 3)
        self._arrive = z(n, dtype=torch.bool)
        robot = self._robots[0]
        self._foot_ids = robot.find_bodies(["left_foot_link", "right_foot_link"], preserve_order=True)[0]
        self._feet = self.scene.sensors["feet_contact_0"]
        self._feet_sensor_ids = self._feet.find_bodies(["left_foot_link", "right_foot_link"], preserve_order=True)[0]
        q_ready = robot.data.default_joint_pos[0].clone()
        ids, _ = robot.find_joints(list(recv_player_cfg.READY_JOINTS), preserve_order=True)
        q_ready[ids] = torch.tensor(list(recv_player_cfg.READY_JOINTS.values()), device=dev)
        self.q_ready = q_ready
        self._upper_ids = robot.find_joints(list(UPPER_JOINTS), preserve_order=True)[0]
        self._upper_s1 = self._upper_ids
        self._upper_s2 = robot.find_joints([j for j in UPPER_JOINTS if j not in SWING_UPPER], preserve_order=True)[0]
        self.stage = "S1"
        self.ready_yaw = float(recv_player_cfg.READY_ROOT["yaw"])
        self.z_ready = float(cfg.z_ready)
        self._hold_action_1 = None  # set by the wrapper

    # The serve reward computer is not used here.
    def _make_serve_reward(self):
        return None

    def _level_bounds(self, level: int) -> tuple[int, int]:
        if self.v4:
            return core.speed_level_bounds(self._v_sorted, level)
        return core.level_bounds(self.table, level)

    def set_level(self, level: int) -> None:
        self.level = int(level)
        self._bounds = self._level_bounds(self.level)

    def set_real_serves(self, paths: list, frac: float) -> list:
        """Load recorded serve files (receiver frame) and feed ``frac`` of the balls from them, styles equally."""
        tabs, infos, src = [], [], []
        for i, p in enumerate(paths):
            d = torch.load(p, map_location="cpu")
            maker = core.table_from_serves_v4 if self.v4 else core.table_from_serves
            t, info = maker(d["pos"], d["vel"], device=self.device)
            info["file"] = p
            tabs.append(t)
            infos.append(info)
            src.append(torch.full((len(t),), i, device=self.device, dtype=torch.long))
            self.real_names.append(p.replace("\\", "/").split("/")[-1].replace("serves_", "").replace(".pt", ""))
        self.real = core.concat_tables(tabs)
        self.real_src_id = torch.cat(src)
        # Equal share per style regardless of how many rows each file kept.
        counts = torch.bincount(self.real_src_id).float()
        self._real_w = (1.0 / counts)[self.real_src_id]
        self.real_frac = float(frac)
        return infos

    def set_stage(self, stage: str) -> None:
        """S2: hold_ready only on the non-swing upper joints (left arm, head), so the right arm and waist can swing."""
        self.stage = stage
        self._upper_ids = self._upper_s2 if stage == "S2" else self._upper_s1

    def set_stage_params(self, weights: dict, sigma_p: float, sigma_r: float) -> None:
        unknown = set(weights) - set(TERMS)
        if unknown:
            raise KeyError(sorted(unknown))
        self.weights = {k: float(weights.get(k, 0.0)) for k in TERMS}
        self.sigma_p, self.sigma_r = float(sigma_p), float(sigma_r)

    # ------------------------------------------------------------------ feeding
    def _inject(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return
        lo, hi = self._bounds
        rows = core.sample_rows(ids.numel(), lo, hi, 0.2, self.device)
        t = self.table
        f = {k: getattr(t, k)[rows] for k in ("launch_pos", "launch_vel", "int_pos", "t_arr", "d_need", "station")}
        src = torch.zeros(ids.numel(), dtype=torch.long, device=self.device)
        if self.real is not None and self.real_frac > 0.0:
            use = torch.rand(ids.numel(), device=self.device) < self.real_frac
            rr = torch.multinomial(self._real_w, ids.numel(), replacement=True)
            for k in f:
                f[k] = torch.where(use.view(-1, *([1] * (f[k].dim() - 1))), getattr(self.real, k)[rr], f[k])
            src = torch.where(use, 1 + self.real_src_id[rr], src)
        self.ball_src[ids] = src
        origins = self.scene.env_origins[ids]
        pos_w = f["launch_pos"] + origins
        state = torch.zeros(ids.numel(), 13, device=self.device)
        state[:, :3] = pos_w
        state[:, 3] = 1.0
        state[:, 7:10] = f["launch_vel"]
        self._shuttle.write_root_state_to_sim(state, env_ids=ids)
        opening = torch.ones_like(ids)
        self.rules.reset(ids, opening, pos_w)
        self.rules.next_hitter[ids] = 0
        self.rules.last_hitter[ids] = 1  # a feed: no serve-height rule, receiver's turn, out faults count
        self.initial_hitter[ids] = 1
        self.station[ids] = f["station"]
        self.int_pos[ids] = f["int_pos"]
        self.t_arr[ids] = f["t_arr"]
        if self.v4:
            # Level of the ball as faced: required speed from where the robot is now (it is not reset between balls).
            root_xy = self._robots[0].data.root_pos_w[ids, :2] - origins[:, :2]
            # Right after a reset the articulation buffers may still hold the old pose: use the ready root.
            fresh = (self.episode_length_buf[ids] == 0)[:, None]
            root_xy = torch.where(fresh, self.ready_xy.expand_as(root_xy), root_xy)
            d_now = torch.linalg.norm(f["station"] - root_xy, dim=-1)
            self.ball_lvl[ids] = core.speed_level(core.required_speed(d_now, f["t_arr"]))
        else:
            self.ball_lvl[ids] = core.ball_level(f["d_need"])
        self.arr_step[ids] = torch.round(f["t_arr"] / self.step_dt).long().clamp(min=1)
        self.ball_active[ids] = True
        self.ball_steps[ids] = 0
        self.ball_touched[ids] = False
        self._prev_phys_shuttle[ids] = pos_w
        self.prev_shuttle_pos_w[ids] = pos_w
        self.curr_shuttle_pos_w[ids] = pos_w

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor):
        self.action_manager.process_action(action.to(self.device))
        self.rules.begin_control_step()
        self._hit_now.zero_()
        self.recorder_manager.record_pre_step()
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self._apply_shuttle_forces()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._physics_substep()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._finish_control_step()
        self._ball_bookkeeping()
        self._compute_values()
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        reward_0 = self.reward_manager.compute(dt=self.step_dt)
        reward_1 = torch.zeros_like(reward_0)
        self.agent_reward_buf = torch.stack((reward_0, reward_1), dim=1)
        self.reward_buf = reward_0
        ended = self.reset_buf
        self.stats[STAT_KEYS.index("episodes")] += ended.float().sum()
        self.stats[STAT_KEYS.index("falls")] += (ended & self.fell).float().sum()
        self.stats[STAT_KEYS.index("fall_low")] += (ended & self._fell_low).float().sum()
        self.stats[STAT_KEYS.index("fall_tilt")] += (ended & self._fell_tilt).float().sum()
        self.stats[STAT_KEYS.index("fall_step_sum")] += ((ended & self.fell).float() * self.episode_length_buf.float()).sum()
        self.stats[STAT_KEYS.index("timeouts")] += self.reset_time_outs.float().sum()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_ids)
            self._reset_idx(reset_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.recorder_manager.record_post_reset(reset_ids)
        # Next ball for the courts whose gap has elapsed.
        nxt = (~self.ball_active) & (self.gap_left <= 0) & (self.balls_done < self.balls_per_episode) & ~self.reset_buf
        self._inject(nxt.nonzero(as_tuple=False).squeeze(-1))
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _physics_substep(self) -> None:
        pos_w = self._shuttle.data.root_pos_w
        local_z = pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.rules.detect_landing(local_z, self.cfg.shuttle_landing_detection_height)
        contacts = torch.stack(
            [
                torch.linalg.norm(contact.data.force_matrix_w, dim=-1).flatten(1).amax(dim=1) > self.cfg.hit_contact_threshold
                for contact in self._contacts
            ],
            dim=1,
        )
        self.rules.update_hits(contacts, local_z)
        self.ball_touched |= self.rules.just_hit[:, 0]
        self._hit_now |= self.rules.just_hit[:, 0]
        self._prev_phys_shuttle.copy_(pos_w)
        self.rules.update_rally_result(pos_w, self.scene.env_origins)

    def _ball_bookkeeping(self) -> None:
        robot = self._robots[0]
        root_xy = robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        act = self.ball_active
        self.ball_steps += act.long()
        self.gap_left = torch.where(act, self.gap_left, (self.gap_left - 1).clamp(min=0))
        arrive = act & (self.ball_steps == self.arr_step)
        d = torch.linalg.norm(self.station - root_xy, dim=-1)
        head = _wrap(self._yaw() - self.ready_yaw).abs()
        a = arrive.float()
        s = self.stats
        s[STAT_KEYS.index("balls")] += a.sum()
        s[STAT_KEYS.index("station_ok")] += (a * (d < 0.35).float()).sum()
        s[STAT_KEYS.index("face_ok")] += (a * (head < math.radians(30.0)).float()).sum()
        s[STAT_KEYS.index("d_sum")] += (a * d).sum()
        s[STAT_KEYS.index("head_sum")] += (a * head).sum()
        s[STAT_KEYS.index("touch")] += (a * self.ball_touched.float()).sum()
        # Speed towards the station while far from it.
        to = self.station - root_xy
        u = to / d.clamp(min=1e-6).unsqueeze(-1)
        v = (robot.data.root_lin_vel_w[:, :2] * u).sum(-1)
        far = (act & (d > 0.3)).float()
        s[STAT_KEYS.index("vel_sum")] += (far * v).sum()
        s[STAT_KEYS.index("vel_n")] += far.sum()
        self.arrival_d.append(torch.where(arrive, d, torch.full_like(d, -1.0)))
        if len(self.arrival_d) > 64:
            self.arrival_d = self.arrival_d[-64:]
        # Hit-instant quantities at t* (the control step the shuttle reaches the generated target).
        sids = self._sweet_spot_ids
        quat = torch.stack([r.data.body_link_quat_w[:, i] for r, i in zip(self._robots, sids)], dim=1)
        q = quat[:, 0]
        ss = robot.data.body_link_pos_w[:, sids[0]] - self.scene.env_origins
        vss = self._racket_link_lin_vel_w(quat)[:, 0]
        ax, ay = core.quat_axes_xy(q)
        self._e_pos = torch.linalg.norm(ss - self.int_pos, dim=-1)
        self._e_ori = core.face_error(ax, self.n_star)
        self._v_n = (vss * self.n_star).sum(-1)
        self._axis_y = ay
        self._arrive = arrive
        s[STAT_KEYS.index("epos_sum")] += (a * self._e_pos).sum()
        s[STAT_KEYS.index("eori_sum")] += (a * self._e_ori).sum()
        s[STAT_KEYS.index("vn_sum")] += (a * self._v_n).sum()
        self.arrival_epos.append(torch.where(arrive, self._e_pos, torch.full_like(d, -1.0)))
        if len(self.arrival_epos) > 64:
            self.arrival_epos = self.arrival_epos[-64:]
        lv = self.ball_lvl
        ls = self.lvl_stats
        ls[:, 0].index_add_(0, lv, a)
        ls[:, 1].index_add_(0, lv, a * (d < 0.35).float())
        ls[:, 2].index_add_(0, lv, a * d)
        ended = act & (self.rally_resolved | (self.ball_steps >= self.ball_timeout_steps))
        hit = self.ball_touched & ended
        s[STAT_KEYS.index("ended")] += ended.float().sum()
        s[STAT_KEYS.index("hit")] += hit.float().sum()
        s[STAT_KEYS.index("ret_in")] += (hit & self.rally_resolved & (self.rally_winner == 0)).float().sum()
        s[STAT_KEYS.index("ret_net")] += (hit & self.rules.step_net_fault).float().sum()
        s[STAT_KEYS.index("ret_out")] += (hit & self.rules.step_out_fault).float().sum()
        ls[:, 3].index_add_(0, lv, ended.float())
        ls[:, 4].index_add_(0, lv, hit.float())
        ss = self.src_stats
        sv = self.ball_src.clamp(max=ss.shape[0] - 1)
        ss[:, 0].index_add_(0, sv, a)
        ss[:, 1].index_add_(0, sv, a * (d < 0.35).float())
        ss[:, 2].index_add_(0, sv, ended.float())
        ss[:, 3].index_add_(0, sv, hit.float())
        first = (self.balls_done == 0).float()
        sf = self.src_stats_first
        sf[:, 0].index_add_(0, sv, a * first)
        sf[:, 1].index_add_(0, sv, a * (d < 0.35).float() * first)
        sf[:, 2].index_add_(0, sv, ended.float() * first)
        sf[:, 3].index_add_(0, sv, hit.float() * first)
        self.ball_active &= ~ended
        self.balls_done += ended.long()
        self.gap_left = torch.where(ended, torch.full_like(self.gap_left, self.gap_steps), self.gap_left)
        self._d = d
        self._v_to = v
        if self.v4:
            to_h = self.ready_xy - root_xy
            dh = torch.linalg.norm(to_h, dim=-1)
            vh = (robot.data.root_lin_vel_w[:, :2] * to_h / dh.clamp(min=1e-6).unsqueeze(-1)).sum(-1)
            self._d_home = dh
            self._d = torch.where(act, d, dh)
            self._v_to = torch.where(act, v, vh)

    def _yaw(self) -> torch.Tensor:
        q = self._robots[0].data.root_quat_w
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _compute_values(self) -> None:
        robot = self._robots[0]
        w = self.weights
        dt = self.step_dt
        v = self.values
        d = self._d
        active = self.ball_active.float()
        far = (self.ball_active & (d > 0.3)).float()
        if self.v4:
            # Between balls: go back towards the ready root (every play rally starts there), half weight.
            gap = (~self.ball_active).float() * 0.5
            active = active + gap
            far = far + gap * (self._d_home > 0.3).float()
        v["track"] = w["track"] * core.r_track(d) * active
        # 11-08 local change (S1 300-iter check failed: v_to_station ~0): dense speed towards the station.
        lo = -self.approach_cap if self.v4 else -0.5  # V4: symmetric clamp (no away-fast / back-fast farming)
        v["approach"] = w["approach"] * self._v_to.clamp(lo, self.approach_cap) * far
        dpsi = _wrap(self._yaw() - self.ready_yaw)
        v["face_front"] = w["face_front"] * torch.exp(-((dpsi / 0.4) ** 2))
        z = robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        v["base_height"] = w["base_height"] * (z - self.z_ready) ** 2
        g = robot.data.projected_gravity_b
        v["base_orientation"] = w["base_orientation"] * (g[:, :2] ** 2).sum(-1)
        feet = self._feet.data
        air = feet.current_air_time[:, self._feet_sensor_ids]
        ctime = feet.current_contact_time[:, self._feet_sensor_ids]
        contact = ctime > 0.0
        single = contact.sum(-1) == 1
        mode_t = torch.where(contact, ctime, air)
        fat = torch.where(single, mode_t.min(dim=-1).values, torch.zeros_like(d)).clamp(max=0.4)
        v["feet_air_time"] = w["feet_air_time"] * fat * far
        foot_pos = robot.data.body_pos_w[:, self._foot_ids]
        foot_vel = robot.data.body_lin_vel_w[:, self._foot_ids]
        h = (foot_pos[..., 2] - foot_pos[..., 2].min(dim=-1, keepdim=True).values).clamp(min=0.0)
        sh = (torch.exp(-(((h - 0.08) / 0.03) ** 2)) * (~contact).float()).sum(-1) * single.float()
        v["step_height"] = w["step_height"] * sh * far
        fq = robot.data.body_quat_w[:, self._foot_ids]
        down = torch.zeros_like(foot_pos)
        down[..., 2] = -1.0
        gf = quat_apply_inverse(fq.reshape(-1, 4), down.reshape(-1, 3)).reshape(down.shape)
        v["foot_posture"] = w["foot_posture"] * (gf[..., :2] ** 2).sum(-1).sum(-1)
        v["no_double_flight"] = w["no_double_flight"] * (~contact).all(dim=-1).float()
        v["feet_slide"] = w["feet_slide"] * (torch.linalg.norm(foot_vel[..., :2], dim=-1) * contact.float()).sum(-1)
        v["acc_reg"] = w["acc_reg"] * (robot.data.joint_acc**2).sum(-1)
        err = robot.data.joint_pos[:, self._upper_ids] - self.q_ready[self._upper_ids]
        v["hold_ready"] = w["hold_ready"] * torch.exp(-(err**2).mean(-1) / 0.1)
        # S2 terms (weights 0 in S1).
        arr = self._arrive.float()
        sw = core.r_swing(self._e_pos, self._e_ori, self._v_n, self.sigma_p, self.sigma_r, self.sigma_v, self.v_cap)
        v["swing"] = w["swing"] * sw * arr / dt
        self.stats[STAT_KEYS.index("swing_sum")] += (sw * arr).sum()
        v["racket_style"] = w["racket_style"] * core.r_racket_style(self._axis_y) * active
        v["contact"] = w["contact"] * self._hit_now.float() / dt
        tau = robot.data.applied_torque
        v["torque"] = w["torque"] * (tau**2).sum(-1)
        v["energy"] = w["energy"] * (tau * robot.data.joint_vel).abs().sum(-1)
        first = self._feet.compute_first_contact(dt)[:, self._feet_sensor_ids]
        v["touchdown_speed"] = w["touchdown_speed"] * ((foot_vel[..., 2] ** 2) * first.float()).sum(-1) / dt
        last_air = feet.last_air_time[:, self._feet_sensor_ids]
        sym = (last_air[:, 0] - last_air[:, 1]).abs() * first.any(dim=-1).float()
        v["gait_symmetry"] = w["gait_symmetry"] * sym / dt
        self._fell_low = z < self.cfg.fall_height
        self._fell_tilt = g[:, 2] > -math.cos(math.radians(45.0))
        fell = self._fell_low | self._fell_tilt
        self.fell = fell
        v["fall"] = w["fall"] * fell.float() / dt

    def _reset_idx(self, env_ids: Sequence[int]):
        DualAgentBadmintonEnv._reset_idx(self, env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.balls_done[ids] = 0
        self.gap_left[ids] = 0
        self.ball_active[ids] = False
        self.fell[ids] = False
        self._inject(ids)

    def pop_stats(self) -> dict:
        c = dict(zip(STAT_KEYS, self.stats.tolist()))
        self.stats.zero_()
        ds = torch.cat(self.arrival_d) if self.arrival_d else torch.zeros(0, device=self.device)
        self.arrival_d = []
        ds = ds[ds >= 0]
        ep = torch.cat(self.arrival_epos) if self.arrival_epos else torch.zeros(0, device=self.device)
        self.arrival_epos = []
        ep = ep[ep >= 0]
        lvl = self.lvl_stats.tolist()
        self.lvl_stats.zero_()
        src = self.src_stats.tolist()
        self.src_stats.zero_()
        src1 = self.src_stats_first.tolist()
        self.src_stats_first.zero_()
        b = max(c["balls"], 1.0)
        en = max(c["ended"], 1.0)
        out = dict(
            balls=c["balls"],
            station_ok=c["station_ok"] / b,
            face_ok=c["face_ok"] / b,
            d_mean=c["d_sum"] / b,
            d_p50=float(ds.median()) if ds.numel() else float("nan"),
            head_deg=math.degrees(c["head_sum"] / b),
            touch=c["touch"] / b,
            fall=c["falls"] / max(c["episodes"], 1.0),
            fall_ball=c["falls"] / b,  # 11-08: per fed ball (play has one ball per rally)
            episodes=c["episodes"],
            v_to_station=c["vel_sum"] / max(c["vel_n"], 1.0),
            fall_low=c["fall_low"],
            fall_tilt=c["fall_tilt"],
            fall_t=c["fall_step_sum"] * self.step_dt / max(c["falls"], 1.0),
            timeouts=c["timeouts"],
            hit=c["hit"] / en,
            ret_in=c["ret_in"] / en,
            ret_net=c["ret_net"] / en,
            ret_out=c["ret_out"] / en,
            epos_mean=c["epos_sum"] / b,
            epos_p50=float(ep.median()) if ep.numel() else float("nan"),
            eori_deg=math.degrees(c["eori_sum"] / b),
            vn_mean=c["vn_sum"] / b,
            swing_mean=c["swing_sum"] / b,
        )
        nan = float("nan")
        for i, (na, ok, dsum, ne, h) in enumerate(lvl):
            out[f"L{i}_n"] = na
            out[f"L{i}_station_ok"] = ok / na if na else nan
            out[f"L{i}_d_mean"] = dsum / na if na else nan
            out[f"L{i}_hit"] = h / ne if ne else nan
        names = ["syn"] + list(self.real_names)
        for i, name in enumerate(names):
            na, ok, ne, h = src[i]
            out[f"src_{name}_station_ok"] = ok / na if na else nan
            out[f"src_{name}_hit"] = h / ne if ne else nan
            out[f"src_{name}_n"] = ne
            na1, ok1, ne1, h1 = src1[i]
            out[f"first_{name}_station_ok"] = ok1 / na1 if na1 else nan
            out[f"first_{name}_hit"] = h1 / ne1 if ne1 else nan
            out[f"first_{name}_n"] = ne1
        return out


# ---------------------------------------------------------------------- manager terms
def recv_term(env, term: str) -> torch.Tensor:
    """Reward-manager getter; use weight 1.0 (the stage weight is already applied)."""
    if not hasattr(env, "values"):
        return torch.zeros(env.num_envs, device=env.device)
    return env.values[term]


def recv_fall_done(env) -> torch.Tensor:
    if not hasattr(env, "fell"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.fell


def recv_balls_done(env) -> torch.Tensor:
    if not hasattr(env, "balls_done"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (env.balls_done >= env.balls_per_episode) & ~env.ball_active


# ---------------------------------------------------------------------- critic terms
def critic_targets(env) -> torch.Tensor:
    """[station rel root (2, trunk frame xy), true intercept rel root (3), time to arrival, active,
    balls left / 6, level / 3, time since injection] = 10."""
    n = env.num_envs
    if not hasattr(env, "station"):
        return torch.zeros(n, 10, device=env.device)
    robot = env._robots[0]
    origin = env.scene.env_origins
    st = torch.zeros(n, 3, device=env.device)
    st[:, :2] = env.station
    st_b = quat_apply_inverse(robot.data.root_quat_w, st + origin - robot.data.root_pos_w)[:, :2]
    ip_b = quat_apply_inverse(robot.data.root_quat_w, env.int_pos + origin - robot.data.root_pos_w)
    t_left = (env.arr_step - env.ball_steps).float() * env.step_dt
    act = env.ball_active.float()
    left = (env.balls_per_episode - env.balls_done).float() / max(env.balls_per_episode, 1)
    return torch.cat(
        (
            st_b * act[:, None],
            ip_b * act[:, None],
            (t_left * act).unsqueeze(-1),
            act.unsqueeze(-1),
            left.unsqueeze(-1),
            torch.full((n, 1), env.level / 3.0, device=env.device),
            (env.ball_steps.float() * env.step_dt * act).unsqueeze(-1),
        ),
        dim=-1,
    )


def critic_feet(env) -> torch.Tensor:
    """[contact L, contact R, air time L, air time R] = 4."""
    if not hasattr(env, "_feet"):
        return torch.zeros(env.num_envs, 4, device=env.device)
    data = env._feet.data
    ids = env._feet_sensor_ids
    return torch.cat(((data.current_contact_time[:, ids] > 0).float(), data.current_air_time[:, ids].clamp(max=1.0)), dim=-1)
