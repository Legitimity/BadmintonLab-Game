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
from . import player_cfg
from .rules import CourtGeometry, MatchRules
from .serve_reward import ServeRewardComputer, StepInputs

SWING_JOINTS = ("Waist", "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw")


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
        self.serve_reward = self._make_serve_reward()

    def _make_serve_reward(self) -> ServeRewardComputer:
        robot = self._robots[0]
        q_ready = robot.data.default_joint_pos[0].clone()
        ids, _ = robot.find_joints(list(player_cfg.READY_JOINTS), preserve_order=True)
        q_ready[ids] = torch.tensor(list(player_cfg.READY_JOINTS.values()), device=q_ready.device)
        hold = torch.ones_like(q_ready)
        swing, _ = robot.find_joints(list(SWING_JOINTS), preserve_order=True)
        hold[swing] = 0.0
        return ServeRewardComputer(self.num_envs, self.device, self.cfg.serve_reward, q_ready, hold, self.step_dt)

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
        self.serve_reward.end_step(self._step_inputs())
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.serve_reward.count_timeouts(self.reset_time_outs)
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
        # Extra diagnostics (not in play) for P1/P2 tools; read them before the reset clears the state.
        self.extras["hit_count"] = self.hit_count.clone()
        self.extras["rally_net_fault"] = self.net_fault.clone()
        self.extras["rally_out_fault"] = self.out_fault.clone()
        self.extras["shuttle_local_pos"] = self._shuttle.data.root_pos_w - self.scene.env_origins
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
        before = self.rules.just_hit.clone()
        self.rules.update_hits(contacts, local_z)
        new_hits = self.rules.just_hit & ~before
        if bool(new_hits.any()):
            sid = self._sweet_spot_ids
            quat = torch.stack([r.data.body_link_quat_w[:, i] for r, i in zip(self._robots, sid)], dim=1)
            vel = torch.stack([r.data.body_link_lin_vel_w[:, i] for r, i in zip(self._robots, sid)], dim=1)
            self.serve_reward.capture_contact(new_hits, self.initial_hitter, quat, vel)
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

    def _step_inputs(self) -> StepInputs:
        origins = self.scene.env_origins
        return StepInputs(
            server=self.initial_hitter,
            just_hit=self.just_hit,
            hit_count=self.hit_count,
            just_resolved=self.rally_just_resolved,
            winner=self.rally_winner,
            height_fault=self.serve_height_fault,
            double_fault=self.double_hit_fault,
            net_fault=self.net_fault,
            out_fault=self.out_fault,
            shuttle_pos=self._shuttle.data.root_pos_w - origins,
            shuttle_vel=self._shuttle.data.root_lin_vel_w,
            root_z=torch.stack([r.data.root_pos_w[:, 2] - origins[:, 2] for r in self._robots], dim=1),
            joint_pos=torch.stack([r.data.joint_pos for r in self._robots], dim=1),
        )

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
        # The runner logs the serve metrics once per iteration (see runner.ServeRunner.log).
        self.serve_reward.reset(ids)
