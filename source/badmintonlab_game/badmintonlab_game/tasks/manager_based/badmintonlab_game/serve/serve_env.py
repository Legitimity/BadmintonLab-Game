"""Training environment with the ``play`` match rules (design doc P0).

The rules live in :mod:`rules` and are differential-tested against the real ``play``
source. The parent class supplies the reward bookkeeping; its control-rate
``_update_hits`` and ``_update_rally_result`` are not called.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_apply_inverse

from ..dual_agent_env import DualAgentBadmintonEnv
from .rules import CourtGeometry, MatchRules


class ServeBadmintonEnv(DualAgentBadmintonEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.rules = MatchRules(self.num_envs, self.device, CourtGeometry.from_cfg(cfg.court), cfg.max_serve_height)
        # Alias the rule state so the parent's reward terms and termination read the same tensors.
        self.just_hit = self.rules.just_hit
        self.next_hitter = self.rules.next_hitter
        self.last_hitter = self.rules.last_hitter
        self.rally_resolved = self.rules.rally_resolved
        self.rally_just_resolved = self.rules.step_just_resolved
        self.shuttle_landed = self.rules.shuttle_landed
        self.net_fault = self.rules.step_net_fault
        self.out_fault = self.rules.step_out_fault
        self.rally_winner = self.rules.rally_winner
        self.serve_height_fault = self.rules.serve_height_fault
        self.double_hit_fault = self.rules.double_hit_fault
        self.extras.setdefault("log", {})
        self._stats = {k: torch.zeros(1, device=self.device) for k in _STAT_KEYS}

    def step(self, action: torch.Tensor):
        self.action_manager.process_action(action.to(self.device))
        self.rules.begin_control_step()
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
        # Same keys as play, preserved before reset clears the state.
        self.extras["rally_resolved"] = self.rally_resolved.clone()
        self.extras["rally_winner"] = self.rally_winner.clone()
        self.extras["serve_height_fault"] = self.serve_height_fault.clone()
        self.extras["double_hit_fault"] = self.double_hit_fault.clone()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_ids)
            self._reset_idx(reset_ids)
            # play refreshes kinematics here; observations read racket body poses right after.
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.recorder_manager.record_post_reset(reset_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _physics_substep(self) -> None:
        """Same order as play: landing, hits, result, once per physics step."""
        pos_w = self._shuttle.data.root_pos_w
        local_z = pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.rules.detect_landing(local_z, self.cfg.shuttle_landing_detection_height)
        contacts = torch.stack(
            [
                torch.linalg.norm(contact.data.force_matrix_w, dim=-1).flatten(1).amax(dim=1)
                > self.cfg.hit_contact_threshold
                for contact in self._contacts
            ],
            dim=1,
        )
        self.rules.update_hits(contacts, local_z)
        self.rules.update_rally_result(pos_w, self.scene.env_origins)

    def _finish_control_step(self) -> None:
        """Translate the rule state into the bookkeeping the parent's rewards use."""
        self.prev_shuttle_pos_w.copy_(self.curr_shuttle_pos_w)
        self.curr_shuttle_pos_w.copy_(self._shuttle.data.root_pos_w)
        self.hit_count += self.just_hit.long()
        self.rally_loser[self.rally_just_resolved] = 1 - self.rally_winner[self.rally_just_resolved]
        self.agent_state[self.just_hit[:, 1], 0] = self.READY
        self.agent_state[self.just_hit[:, 0], 1] = self.READY
        self.agent_state[self.just_hit] = self.RECOVER
        for side in range(2):
            hit = self.just_hit[:, side]
            self.outgoing_eval_step[hit, side] = self.episode_length_buf[hit] + 1
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
        self._accumulate_stats()

    def _accumulate_stats(self) -> None:
        done = self.rally_just_resolved
        if not done.any():
            return
        server = self.initial_hitter[done]
        winner = self.rally_winner[done]
        hit = self.hit_count[done].gather(1, server.unsqueeze(1)).squeeze(1) > 0
        height = self.serve_height_fault[done]
        double = self.double_hit_fault[done]
        s = self._stats
        s["rallies"] += done.sum()
        s["legal"] += hit.sum()
        s["height"] += height.sum()
        s["double"] += double.sum()
        s["no_hit"] += (~hit & ~height).sum()
        s["in"] += (hit & (winner == server) & ~double & ~height).sum()
        s["net"] += self.net_fault[done].sum()
        s["out"] += self.out_fault[done].sum()

    def _apply_shuttle_forces(self) -> None:
        velocity = self._shuttle.data.root_lin_vel_w
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        force = -self._drag_k * speed * velocity + self._mass * self._g_vec
        force_b = quat_apply_inverse(self._shuttle.data.root_quat_w, force)
        self._shuttle.permanent_wrench_composer.set_forces_and_torques(
            forces=force_b.unsqueeze(1), torques=torch.zeros_like(force_b).unsqueeze(1), is_global=False
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.rules.reset(ids, self.initial_hitter[ids], self._shuttle.data.root_pos_w[ids])
        self._log_stats()

    def _log_stats(self) -> None:
        s = self._stats
        n = max(float(s["rallies"].item()), 1.0)
        for key in _STAT_KEYS:
            if key != "rallies":
                self.extras["log"][f"Metrics/Serve/{key}_rate"] = float(s[key].item()) / n
        if s["rallies"].item() >= 256:
            for key in _STAT_KEYS:
                s[key].zero_()


_STAT_KEYS = ("rallies", "legal", "height", "double", "no_hit", "in", "net", "out")
