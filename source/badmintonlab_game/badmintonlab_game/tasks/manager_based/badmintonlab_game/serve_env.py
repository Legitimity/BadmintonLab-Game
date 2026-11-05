"""Training environment whose rule judgement is aligned with the ``play`` evaluator.

``DualAgentBadmintonEnv`` (the default task) judges hits once per control step
from the contact history and ignores the 1.15 m serve rule and double hits.
The evaluator judges every 200 Hz physics substep.  This subclass ports the
evaluator's substep judgement while keeping the default task's control-step
reward/metric bookkeeping, so the same reward terms remain usable:

* hits: rising edge of the racket/shuttle contact, read every substep;
* serve-height fault (first server contact above ``max_serve_height``) and
  double-hit fault (contact by the player who is not ``next_hitter``);
* rally result: first result inside a control step is latched;
* landing flag: latched;
* after resets: ``scene.write_data_to_sim()`` + ``sim.forward()``;
* shuttle aerodynamics through ``permanent_wrench_composer``.

Event flags raised in any substep are kept until the end of the control step
(``just_hit``, ``rally_just_resolved``, ``net_fault``, ``out_fault``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_apply_inverse

from badmintonlab_game.assets.court import is_net_fault, is_out_of_bounds

from .dual_agent_env import DualAgentBadmintonEnv


class ServeBadmintonEnv(DualAgentBadmintonEnv):
    """Dual-agent badminton with ``play``-aligned substep rule judgement."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        n = self.num_envs
        self.max_serve_height = float(cfg.max_serve_height)
        self._in_contact = torch.zeros((n, 2), dtype=torch.bool, device=self.device)
        self.rally_winner = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.serve_height_fault = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.double_hit_fault = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Latched rally end reasons (first result only).
        self.result_net = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.result_out = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.result_land = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._resolved_at_step_start = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Substep shuttle positions for rule judgement; the inherited
        # prev/curr_shuttle_pos_w stay control-step positions for the
        # approach and return bookkeeping.
        self._sub_prev_pos_w = self._shuttle.data.root_pos_w.clone()
        self._sub_curr_pos_w = self._shuttle.data.root_pos_w.clone()

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor):
        """Advance physics with substep rule judgement, then settle rewards per control step."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        self.just_hit.zero_()
        self._resolved_at_step_start.copy_(self.rally_resolved)
        self.prev_shuttle_pos_w.copy_(self.curr_shuttle_pos_w)
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
            self._detect_landing()
            self._substep_hits()
            self._substep_rally_result()
        self.curr_shuttle_pos_w.copy_(self._shuttle.data.root_pos_w)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._finalize_hits()
        self._finalize_rally_result()
        self._update_approach_state()
        self._update_return_history()
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        reward_0 = self.reward_manager.compute(dt=self.step_dt)
        reward_1 = self.player_1_reward_manager.compute(dt=self.step_dt)
        self.approach_reward_event.zero_()
        self.agent_reward_buf = torch.stack((reward_0, reward_1), dim=1)
        self.reward_buf = self.agent_reward_buf.mean(dim=1)

        # Same keys as the play evaluator, captured before reset clears them.
        self.extras["rally_resolved"] = self.rally_resolved.clone()
        self.extras["rally_winner"] = self.rally_winner.clone()
        self.extras["serve_height_fault"] = self.serve_height_fault.clone()
        self.extras["double_hit_fault"] = self.double_hit_fault.clone()

        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_ids)
            self._reset_idx(reset_ids)
            # Match play: push reset state and refresh link kinematics so the
            # first observation after a reset is not stale.
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.recorder_manager.record_post_reset(reset_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # -------------------------------------------------------------- physics
    def _apply_shuttle_forces(self):
        """Aerodynamic drag and gravity through the permanent wrench composer (as in play)."""
        velocity = self._shuttle.data.root_lin_vel_w
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        force = -self._drag_k * speed * velocity + self._mass * self._g_vec
        force_b = quat_apply_inverse(self._shuttle.data.root_quat_w, force)
        self._shuttle.permanent_wrench_composer.set_forces_and_torques(
            forces=force_b.unsqueeze(1), torques=torch.zeros_like(force_b).unsqueeze(1), is_global=False
        )

    def _detect_landing(self):
        """Latched landing flag (cleared only on reset)."""
        local_z = self._shuttle.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.shuttle_landed |= local_z <= self.cfg.shuttle_landing_detection_height

    # ------------------------------------------------------- substep rules
    def _substep_hits(self):
        """Rising-edge racket contacts with serve-height and double-hit faults (play semantics)."""
        contacts = torch.stack(
            [
                torch.linalg.norm(contact.data.force_matrix_w, dim=-1).flatten(1).amax(dim=1)
                > self.cfg.hit_contact_threshold
                for contact in self._contacts
            ],
            dim=1,
        )
        new_contacts = contacts & ~self._in_contact
        self._in_contact.copy_(contacts)
        height = self._shuttle.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        for side in range(2):
            hit = new_contacts[:, side] & ~self.rally_resolved
            wrong_turn = hit & (self.next_hitter != side)
            high_serve = hit & (self.last_hitter < 0) & (height > self.max_serve_height)
            fault = wrong_turn | high_serve
            self.double_hit_fault |= wrong_turn
            self.serve_height_fault |= high_serve
            self.rally_winner[fault] = 1 - side
            self.rally_resolved |= fault
            valid = hit & ~self.rally_resolved
            self.just_hit[:, side] |= valid
            self.hit_count[:, side] += valid.long()
            self.next_hitter[valid] = 1 - side
            self.last_hitter[valid] = side

    def _substep_rally_result(self):
        """Latch the first net / out / landing result of the rally (play semantics)."""
        self._sub_prev_pos_w.copy_(self._sub_curr_pos_w)
        self._sub_curr_pos_w.copy_(self._shuttle.data.root_pos_w)
        local = self._sub_curr_pos_w - self.scene.env_origins
        prev_local = self._sub_prev_pos_w - self.scene.env_origins
        net_fault = is_net_fault(prev_local, local, self.cfg.court)
        out_fault = is_out_of_bounds(local, self.cfg.court) & (self.last_hitter >= 0)
        active = ~self.rally_resolved
        shot_fault = active & (net_fault | out_fault)
        hitter = torch.where(self.last_hitter >= 0, self.last_hitter, self.next_hitter)
        self.rally_winner[shot_fault] = 1 - hitter[shot_fault]
        landed = active & self.shuttle_landed & ~shot_fault
        # Player 0 owns negative x; an unhit opening drop loses on its landing half.
        landing_side = (local[:, 0] >= self.cfg.court.net_x).long()
        self.rally_winner[landed] = 1 - landing_side[landed]
        self.result_net |= active & net_fault
        self.result_out |= active & out_fault & ~net_fault
        self.result_land |= landed
        self.rally_resolved |= net_fault | out_fault | self.shuttle_landed

    # ------------------------------------------------- control-step settle
    def _finalize_hits(self):
        """Control-step bookkeeping for hits accumulated over the substeps."""
        for side in range(2):
            hit = self.just_hit[:, side]
            self.outgoing_eval_step[hit, side] = self.episode_length_buf[hit] + 1

        self.agent_state[self.just_hit[:, 1], 0] = self.READY
        self.agent_state[self.just_hit[:, 0], 1] = self.READY
        self.agent_state[self.just_hit] = self.RECOVER

        for side in range(2):
            ready_ids = self.just_hit[:, 1 - side].nonzero(as_tuple=False).flatten()
            if ready_ids.numel() > 0:
                self.approach_reward_pending[ready_ids, side] = True
                self.approach_min_distance[ready_ids, side] = torch.inf
                slots = self.return_history_next[ready_ids, side]
                self.return_history[ready_ids, side, slots] = False
                self.return_history_next[ready_ids, side] = (slots + 1) % self.return_history_window
                self.return_opportunity_slot[ready_ids, side] = slots
                self.return_opportunity_pending[ready_ids, side] = True
                self.return_hit_pending[ready_ids, side] = False

    def _finalize_rally_result(self):
        """Expose the latched result as control-step event flags for rewards."""
        self.rally_just_resolved = self.rally_resolved & ~self._resolved_at_step_start
        self.net_fault = self.result_net & self.rally_just_resolved
        self.out_fault = self.result_out & self.rally_just_resolved
        just = self.rally_just_resolved
        self.rally_loser[just] = 1 - self.rally_winner[just]

    # ----------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: Sequence[int]):
        """Log rule metrics for finished rallies, then clear substep state."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        metrics = {}
        if env_ids.numel() > 0:
            resolved = self.rally_resolved[env_ids].float()
            server = self.initial_hitter[env_ids]
            server_hits = self.hit_count[env_ids].gather(1, server.unsqueeze(1)).squeeze(1)
            any_hit = self.hit_count[env_ids].sum(dim=1) > 0
            winner = self.rally_winner[env_ids]
            metrics = {
                "Metrics/Rule/resolved": resolved.mean(),
                "Metrics/Rule/timeout": 1.0 - resolved.mean(),
                "Metrics/Rule/serve_height_fault": self.serve_height_fault[env_ids].float().mean(),
                "Metrics/Rule/double_hit_fault": self.double_hit_fault[env_ids].float().mean(),
                "Metrics/Rule/net_fault": self.result_net[env_ids].float().mean(),
                "Metrics/Rule/out_fault": self.result_out[env_ids].float().mean(),
                "Metrics/Rule/landed": self.result_land[env_ids].float().mean(),
                "Metrics/Serve/legal_hit_rate": (server_hits > 0).float().mean(),
                "Metrics/Serve/no_hit_rate": (~any_hit).float().mean(),
                "Metrics/Serve/server_win_rate": (winner == server).float().mean(),
            }
        super()._reset_idx(env_ids)
        self.extras["log"].update(metrics)
        self._in_contact[env_ids] = False
        self.rally_winner[env_ids] = -1
        self.serve_height_fault[env_ids] = False
        self.double_hit_fault[env_ids] = False
        self.result_net[env_ids] = False
        self.result_out[env_ids] = False
        self.result_land[env_ids] = False
        self._resolved_at_step_start[env_ids] = False
        new_pos = self._shuttle.data.root_pos_w[env_ids]
        self._sub_prev_pos_w[env_ids] = new_pos
        self._sub_curr_pos_w[env_ids] = new_pos
