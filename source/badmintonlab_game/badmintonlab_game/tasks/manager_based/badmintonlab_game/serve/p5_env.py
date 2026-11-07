"""P5 receive environment (P5_RECEIVE_DESIGN 2): natural and feeder rallies, receive rewards, sample weights.

* Every court draws its rally type at reset: ``natural`` with probability ``natural_share`` (the P3 serve, the
  server's rewards unchanged), otherwise ``feeder``: the opening hitter is moved off court
  (``feeder_x``, ``+-feeder_y``, facing the net), the shuttle drops from the usual point and, at the physics step
  it first reaches the drawn launch height, gets the table's launch velocity and lateral offset
  (``FeederTable``). The launch step is known at reset (deterministic free fall, ``feeder.drop_time``), so the
  per-physics-step check is a host-side comparison and costs no device sync.
* The rule state of a launch is ``MatchRules.inject_serve`` plus ``hit_count[feeder] = 1``.
* Natural serves announce their incoming ball one control step after the server's first hit (flight scan of
  the post-contact state, same drag law), feeder rallies at the launch.
* ``p5_aux()``: (N, 2, 2) per side ``[learn_w, teach_w]``. ``learn_w`` = 0 for the feeder; ``teach_w`` = 1 for
  the server of a natural rally from reset until ``teach_after_hit`` s after its first hit.
* Feeder landing check (2.2): the landing point of feeder rallies the receiver did not touch, against the
  table; ``pop_verify()`` returns per-level counts. ``verify_mode`` also moves the receiver off court.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from isaaclab.utils.math import quat_from_euler_xyz

from .feeder import LAUNCH_X, FeederParams, FeederTable, drop_time, legal_mask, scan_trajectories
from .receive import RECV_STAT_KEYS, ReceiveParams, ReceiveRewardComputer, ReceiveStepInputs
from .serve_env import ServeBadmintonEnv
from .serve_reward import to_canonical

VERIFY_KEYS = ("n", "ok", "err_sum")


class ReceiveBadmintonEnv(ServeBadmintonEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        p5 = cfg.p5
        self.p5 = p5
        self.recv_reward = ReceiveRewardComputer(n, dev, cfg.recv_reward, self.serve_reward.p, self.step_dt)
        self.feeder_params = FeederParams(floor=float(cfg.shuttle_landing_detection_height))
        self.table = self._load_table()
        self._k_drop = self._drop_steps(self.table)
        self.natural_share = float(p5.natural_share)
        self.feeder_level = int(p5.level)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        self.natural = torch.ones(n, dtype=torch.bool, device=dev)
        self.feeder_row = torch.zeros(n, dtype=torch.long, device=dev)
        self.feeder_row_level = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.launched = z(n, dt=torch.bool)
        self.server_hit_t = torch.full((n,), -1.0, device=dev)
        self._nat_pending = z(n, dt=torch.bool)
        self._launch_np = np.full(n, -1, dtype=np.int64)
        self._next_launch = math.inf
        self._prev_phys_vel = z(n, 3)
        self._landed_prev = z(n, dt=torch.bool)
        self._land_pos = z(n, 3)
        self._verify = torch.zeros(len(self.feeder_params.levels), len(VERIFY_KEYS), device=dev)
        # Natural serves: legal incoming = clears the net edge, lands in the receiver's singles court.
        self._nat_params = FeederParams(net_margin=0.0, floor=self.feeder_params.floor)
        # The P5 state of a court (rally type, feeder pose, launch) is drawn by _reset_idx, first at env.reset().

    # ------------------------------------------------------------------ feeder table
    def _load_table(self) -> FeederTable:
        drag = float(self.cfg.serve_reward.drag_over_mass)
        g = float(self.cfg.serve_reward.gravity)
        table = FeederTable.build(self.feeder_params, drag, g, device="cpu")
        log_dir = getattr(self.cfg, "log_dir", None)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"data": table.data, "cuts": table.cuts, "total_rows": table.total_rows,
                        "params": vars(self.feeder_params)}, Path(log_dir) / "feeder_table.pt")
        print(f"[P5] feeder table: {len(table)} legal of {table.total_rows} rows, levels {table.level_counts()}", flush=True)
        table.data = table.data.to(self.device)
        return table

    def _drop_steps(self, table: FeederTable) -> torch.Tensor:
        z = table.col("z")
        uniq, inv = torch.unique(z.cpu().double(), return_inverse=True)
        k, _ = drop_time(uniq, float(self.cfg.serve_reward.drag_over_mass), float(self.cfg.serve_reward.gravity),
                         float(self.physics_dt))
        return k.long()[inv].to(self.device)

    def row_level(self, rows: torch.Tensor) -> torch.Tensor:
        cuts = torch.tensor(self.table.cuts[1:], device=rows.device)
        return torch.searchsorted(cuts, rows, right=True).clamp(max=len(self.table.cuts) - 2)

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor):
        self.action_manager.process_action(action.to(self.device))
        self.rules.begin_control_step()
        self.recorder_manager.record_pre_step()
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for sub in range(self.cfg.decimation):
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
            if self._sim_step_counter >= self._next_launch:
                self._launch_due(sub)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._finish_control_step()
        if self.cfg.rewards.interval_approach is not None and self.cfg.player_1_rewards.interval_approach is not None:
            self._update_approach_state()
        self._update_return_history()
        self.serve_reward.end_step(self._step_inputs())
        self._receive_control_step()
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.serve_reward.count_timeouts(self.reset_time_outs)
        self.recv_reward.count_rallies(self.reset_buf, self.reset_time_outs)
        reward_0 = self.reward_manager.compute(dt=self.step_dt)
        reward_1 = self.player_1_reward_manager.compute(dt=self.step_dt)
        self.approach_reward_event.zero_()
        self.agent_reward_buf = torch.stack((reward_0, reward_1), dim=1)
        self.reward_buf = self.agent_reward_buf.mean(dim=1)
        self.extras["rally_resolved"] = self.rally_resolved.clone()
        self.extras["rally_winner"] = self.rally_winner.clone()
        self.extras["serve_height_fault"] = self.serve_height_fault.clone()
        self.extras["double_hit_fault"] = self.double_hit_fault.clone()
        self.extras["hit_count"] = self.hit_count.clone()
        self.extras["rally_net_fault"] = self.net_fault.clone()
        self.extras["rally_out_fault"] = self.out_fault.clone()
        self.extras["shuttle_local_pos"] = self._shuttle.data.root_pos_w - self.scene.env_origins
        self.extras["natural"] = self.natural.clone()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_ids)
            self._reset_idx(reset_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.recorder_manager.record_post_reset(reset_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _physics_substep(self) -> None:
        """``ServeBadmintonEnv._physics_substep`` plus the receiver's contact and pass-through capture."""
        pos_w = self._shuttle.data.root_pos_w
        local_z = pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.rules.detect_landing(local_z, self.cfg.shuttle_landing_detection_height)
        newly = self.rules.shuttle_landed & ~self._landed_prev
        self._land_pos = torch.where(newly[:, None], pos_w - self.scene.env_origins, self._land_pos)
        self._landed_prev.copy_(self.rules.shuttle_landed)
        contacts = torch.stack(
            [
                torch.linalg.norm(contact.data.force_matrix_w, dim=-1).flatten(1).amax(dim=1)
                > self.cfg.hit_contact_threshold
                for contact in self._contacts
            ],
            dim=1,
        )
        before = self.rules.just_hit.clone()
        self.rules.update_hits(contacts, local_z)
        new_hits = self.rules.just_hit & ~before
        sid = self._sweet_spot_ids
        quat = torch.stack([r.data.body_link_quat_w[:, i] for r, i in zip(self._robots, sid)], dim=1)
        racket_vel = self._racket_link_lin_vel_w(quat)
        self.serve_reward.capture_contact(new_hits, self.initial_hitter, quat, racket_vel)
        self.recv_reward.capture_contact(new_hits, self.initial_hitter, quat, racket_vel, self._prev_phys_vel)
        racket_pos = torch.stack([r.data.body_link_pos_w[:, i] for r, i in zip(self._robots, sid)], dim=1)
        self.serve_reward.capture_tunnel(self.initial_hitter, racket_pos, quat, pos_w, self._prev_phys_shuttle, contacts)
        self.recv_reward.capture_tunnel(self.initial_hitter, racket_pos, quat, pos_w, self._prev_phys_shuttle, contacts)
        self._prev_phys_shuttle.copy_(pos_w)
        self._prev_phys_vel.copy_(self._shuttle.data.root_lin_vel_w)
        self.rules.update_rally_result(pos_w, self.scene.env_origins)

    # ------------------------------------------------------------------ feeder launch
    def _launch_due(self, sub: int) -> None:
        due = np.nonzero(self._launch_np == self._sim_step_counter)[0]
        self._launch_np[due] = -1
        pending = self._launch_np[self._launch_np >= 0]
        self._next_launch = int(pending.min()) if pending.size else math.inf
        if due.size == 0:
            return
        ids = torch.as_tensor(due, device=self.device, dtype=torch.long)
        feeder = self.initial_hitter[ids]
        row = self.table.data[self.feeder_row[ids]]
        col = lambda name: row[:, self._col(name)]  # noqa: E731
        pos_c = torch.stack((torch.full_like(col("z"), LAUNCH_X), col("y"), col("z")), -1)
        vel_c = torch.stack((col("vx"), col("vy"), col("vz")), -1)
        pos_w = to_canonical(pos_c, feeder) + self.scene.env_origins[ids]
        vel_w = to_canonical(vel_c, feeder)
        quat = self._shuttle.data.root_quat_w[ids]
        self._shuttle.write_root_pose_to_sim(torch.cat((pos_w, quat), -1), env_ids=ids)
        self._shuttle.write_root_velocity_to_sim(torch.cat((vel_w, torch.zeros_like(vel_w)), -1), env_ids=ids)
        live = ~self.rules.rally_resolved[ids]
        self.rules.inject_serve(ids, feeder)
        self.rules.curr_pos[ids] = pos_w
        self._prev_phys_shuttle[ids] = pos_w
        self._prev_phys_vel[ids] = vel_w
        self.hit_count[ids, feeder] = torch.where(live, torch.ones_like(feeder), self.hit_count[ids, feeder])
        self.launched[ids] = live
        n = self.num_envs
        mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask[ids] = live
        tgt = torch.zeros(n, 3, device=self.device)
        tgt[ids] = torch.stack((col("tgt_x"), col("tgt_y"), col("tgt_z")), -1)
        t_arr = torch.zeros(n, device=self.device)
        t_now = (self.episode_length_buf[ids].float() + (sub + 1) / self.cfg.decimation) * self.step_dt
        t_arr[ids] = t_now + col("t_tgt")
        need = torch.zeros(n, device=self.device)
        need[ids] = col("d_need")
        level = torch.full((n,), -1.0, device=self.device)
        level[ids] = self.feeder_row_level[ids].float()
        self.recv_reward.set_incoming(mask, tgt, t_arr, need, level)

    @staticmethod
    def _col(name: str) -> int:
        from .feeder import COLUMNS

        return COLUMNS.index(name)

    # ------------------------------------------------------------------ control rate
    def _receive_control_step(self) -> None:
        n = self.num_envs
        ar = torch.arange(n, device=self.device)
        server = self.initial_hitter
        t = self.episode_length_buf.float() * self.step_dt
        # Natural serves: the incoming ball is announced one step after the server's first hit.
        if bool(self._nat_pending.any()):
            ids = (self._nat_pending & ~self.rally_resolved).nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() > 0:
                s = server[ids]
                pos_c = to_canonical(self._shuttle.data.root_pos_w[ids] - self.scene.env_origins[ids], s)
                vel_c = to_canonical(self._shuttle.data.root_lin_vel_w[ids], s)
                scan = scan_trajectories(pos_c, vel_c, float(self.cfg.serve_reward.drag_over_mass),
                                         float(self.cfg.serve_reward.gravity), self._nat_params,
                                         dt=self.p5.natural_scan_dt, scheme="midpoint", max_time=self.p5.natural_scan_s)
                ok = legal_mask(scan, self._nat_params)
                mask = torch.zeros(n, dtype=torch.bool, device=self.device)
                mask[ids] = ok
                tgt = torch.zeros(n, 3, device=self.device)
                tgt[ids] = scan["tgt"]
                t_arr = torch.zeros(n, device=self.device)
                t_arr[ids] = t[ids] + scan["t_tgt"]
                need = torch.zeros(n, device=self.device)
                need[ids] = torch.where(ok, scan["d_need"], torch.zeros_like(scan["d_need"]))
                self.recv_reward.set_incoming(mask, tgt, t_arr, need)
            self._nat_pending.zero_()
        first = self.natural & self.just_hit[ar, server] & (self.hit_count[ar, server] == 1) & (self.server_hit_t < 0)
        self.server_hit_t = torch.where(first, t, self.server_hit_t)
        self._nat_pending |= first

        origins = self.scene.env_origins
        sid = self._sweet_spot_ids
        self.recv_reward.end_step(ReceiveStepInputs(
            server=server,
            just_hit=self.just_hit,
            just_resolved=self.rally_just_resolved,
            winner=self.rally_winner,
            net_fault=self.net_fault,
            shuttle_pos=self._shuttle.data.root_pos_w - origins,
            shuttle_vel=self._shuttle.data.root_lin_vel_w,
            root_pos=torch.stack([r.data.root_pos_w - origins for r in self._robots], dim=1),
            root_vel=torch.stack([r.data.root_lin_vel_w for r in self._robots], dim=1),
            racket_pos=torch.stack([r.data.body_link_pos_w[:, i] - origins for r, i in zip(self._robots, sid)], dim=1),
            root_z=torch.stack([r.data.root_pos_w[:, 2] - origins[:, 2] for r in self._robots], dim=1),
            gravity_z=torch.stack([r.data.projected_gravity_b[:, 2] for r in self._robots], dim=1),
            fall_given=self.serve_reward.fall_given,
            time_s=t,
            natural=self.natural,
        ))

        # Feeder landing check: untouched feeder balls that landed, against the table.
        check = self.rally_just_resolved & ~self.natural & self.launched & self.rules.shuttle_landed & ~self.recv_reward.hit
        pred = torch.stack((self.table.data[self.feeder_row, self._col("land_x")],
                            self.table.data[self.feeder_row, self._col("land_y")]), -1)
        got = to_canonical(self._land_pos, server)[:, :2]
        err = torch.linalg.norm(got - pred, dim=-1)
        lv = self.feeder_row_level.clamp(min=0)
        w = check.float()
        self._verify.index_add_(0, lv, torch.stack((w, w * (err < self.p5.verify_tol).float(), w * err), -1))

    def pop_verify(self) -> dict:
        v = self._verify.tolist()
        self._verify.zero_()
        return {f"L{i}_{k}": row[j] for i, row in enumerate(v) for j, k in enumerate(VERIFY_KEYS)}

    def p5_aux(self) -> torch.Tensor:
        """(N, 2, 2): per side ``[learn_w, teach_w]`` of the current state (after any reset)."""
        n = self.num_envs
        side = torch.arange(2, device=self.device).expand(n, 2)
        server = self.initial_hitter.unsqueeze(1)
        is_server = side == server
        learn = ~(is_server & ~self.natural.unsqueeze(1))
        t = self.episode_length_buf.float() * self.step_dt
        window = (self.server_hit_t < 0) | (t - self.server_hit_t < self.p5.teach_after_hit)
        teach = is_server & (self.natural & window).unsqueeze(1)
        return torch.stack((learn.float(), teach.float()), dim=-1)

    def p5_critic_obs(self, side: int) -> torch.Tensor:
        """(N, 11): rally type (2), role (2), target - root in own frame (3), time to target, d_need, valid, level."""
        n = self.num_envs
        server = self.initial_hitter
        nat = self.natural.float()
        is_srv = (server == side).float()
        rr = self.recv_reward
        valid = rr.active & ~rr.hit & (server != side)
        root_l = self._robots[side].data.root_pos_w - self.scene.env_origins
        rel_srv = rr.tgt - to_canonical(root_l, server)
        rel_side = to_canonical(to_canonical(rel_srv, server), torch.full_like(server, side))
        t = self.episode_length_buf.float() * self.step_dt
        vf = valid.float()
        lvl = torch.where(self.natural, torch.zeros_like(nat), (self.feeder_row_level.float() + 1.0) / 4.0)
        return torch.cat((
            torch.stack((nat, 1.0 - nat, is_srv, 1.0 - is_srv), -1),
            rel_side * vf[:, None],
            ((rr.t_arrive - t).clamp(min=0.0, max=3.0) * vf)[:, None],
            (rr.d_need.clamp(max=3.0) * vf)[:, None],
            vf[:, None],
            lvl[:, None],
        ), dim=-1)

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if not hasattr(self, "recv_reward"):
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = ids.numel()
        nat = torch.rand(k, device=self.device) < self.natural_share
        self.natural[ids] = nat
        self.serve_reward.natural[ids] = nat
        self.recv_reward.reset(ids)
        self.recv_reward.natural[ids] = nat
        self.launched[ids] = False
        self.server_hit_t[ids] = -1.0
        self._nat_pending[ids] = False
        self._landed_prev[ids] = False
        self._land_pos[ids] = 0.0
        self._prev_phys_vel[ids] = 0.0
        if self.p5.verify_mode:
            levels = torch.randint(0, len(self.feeder_params.levels), (k,), device=self.device)
            rows = self.table.sample(levels, k)
        else:
            rows = self.table.sample(self.feeder_level, k)
        self.feeder_row[ids] = rows
        self.feeder_row_level[ids] = torch.where(nat, torch.full_like(rows, -1), self.row_level(rows))
        server = self.initial_hitter[ids]
        feeder_ids = ids[~nat]
        launch_at = torch.full((k,), -1, dtype=torch.long, device=self.device)
        launch_at[~nat] = self._sim_step_counter + self._k_drop[rows[~nat]]
        self._launch_np[ids.cpu().numpy()] = launch_at.cpu().numpy()
        pending = self._launch_np[self._launch_np >= 0]
        self._next_launch = int(pending.min()) if pending.size else math.inf
        if feeder_ids.numel() > 0:
            self._place_off_court(feeder_ids, server[~nat], self.p5.feeder_x, self.p5.feeder_y, random_side=True)
        if self.p5.verify_mode and feeder_ids.numel() > 0:
            self._place_off_court(feeder_ids, 1 - server[~nat], self.p5.verify_recv_x, self.p5.verify_recv_y,
                                  random_side=False, facing_net=True)

    def _place_off_court(self, ids, side, x_c: float, y_c: float, random_side: bool, facing_net: bool = True) -> None:
        """Root of robot ``side`` at canonical (x_c, +-y_c) of its own half, facing the net."""
        for s in (0, 1):
            sel = ids[side == s]
            if sel.numel() == 0:
                continue
            robot = self._robots[s]
            m = sel.numel()
            y = torch.full((m,), float(y_c), device=self.device)
            if random_side:
                y = y * torch.where(torch.rand(m, device=self.device) < 0.5, 1.0, -1.0)
            sign = 1.0 if s == 0 else -1.0
            pos = robot.data.root_pos_w[sel].clone()
            pos[:, 0] = self.scene.env_origins[sel, 0] + sign * float(x_c)
            pos[:, 1] = self.scene.env_origins[sel, 1] + sign * y
            yaw = torch.full((m,), 0.0 if s == 0 else math.pi, device=self.device)
            quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
            robot.write_root_pose_to_sim(torch.cat((pos, quat), -1), env_ids=sel)
            robot.write_root_velocity_to_sim(torch.zeros(m, 6, device=self.device), env_ids=sel)

    # ------------------------------------------------------------------ runner interface
    def set_level(self, level: int) -> None:
        self.feeder_level = int(min(max(level, 0), len(self.feeder_params.levels) - 1))

    def set_natural_share(self, share: float) -> None:
        self.natural_share = float(share)


__all__ = ["ReceiveBadmintonEnv", "RECV_STAT_KEYS", "ReceiveParams"]
