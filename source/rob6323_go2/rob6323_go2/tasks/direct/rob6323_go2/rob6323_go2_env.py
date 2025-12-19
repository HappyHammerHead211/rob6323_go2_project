# rob6323_go2/envs/rob6323_go2_env.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    """Direct RL environment for Unitree Go2."""

    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        # this builds self.robot, self.scene, self._contact_sensor, etc.
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)

        # joint position commands (scaled actions)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, action_dim, device=self.device)

        # desired base command (vx, vy, yaw_rate)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # history of actions for smoothness penalty
        self.last_actions = torch.zeros(
            self.num_envs,
            action_dim,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # low-level PD gains (per env, per joint)
        self.Kp = torch.full(
            (self.num_envs, action_dim),
            cfg.Kp,
            device=self.device,
            dtype=torch.float,
        )
        self.Kd = torch.full(
            (self.num_envs, action_dim),
            cfg.Kd,
            device=self.device,
            dtype=torch.float,
        )
        self.torque_limits = cfg.torque_limits
        self.desired_joint_pos = torch.zeros(self.num_envs, action_dim, device=self.device)

        # last torques for L2 regularization
        self._last_torques = torch.zeros(
            self.num_envs, action_dim, device=self.device, dtype=torch.float
        )

        # ------------ Bonus 1: actuator friction parameters ------------
        # stiction coeff F_s and viscous coeff mu_v for each env/joint
        self.f_s = torch.zeros(self.num_envs, action_dim, device=self.device)
        self.mu_v = torch.zeros(self.num_envs, action_dim, device=self.device)
        # ----------------------------------------------------------------

        # feet names
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        # body indices for feet (kinematics)
        feet_ids = []
        for name in foot_names:
            ids, _ = self.robot.find_bodies(name)
            feet_ids.append(ids[0])
        self._feet_ids = torch.tensor(feet_ids, device=self.device, dtype=torch.long)

        # contact sensor body indices
        base_ids, _ = self._contact_sensor.find_bodies("base")
        self._base_id = int(base_ids[0])

        feet_ids_sensor = []
        for name in foot_names:
            ids, _ = self._contact_sensor.find_bodies(name)
            feet_ids_sensor.append(ids[0])
        self._feet_ids_sensor = torch.tensor(
            feet_ids_sensor, device=self.device, dtype=torch.long
        )

        # undesired leg links (hips / thighs / calves)
        leg_link_patterns = [".*_hip.*", ".*_thigh.*", ".*_calf.*"]
        leg_ids = []
        leg_names = []
        for pat in leg_link_patterns:
            ids, names = self._contact_sensor.find_bodies(pat)
            leg_ids += [int(i) for i in ids]
            leg_names += list(names)

        feet_set = set(int(i) for i in self._feet_ids_sensor.tolist())
        leg_ids = sorted(set(leg_ids) - feet_set - {int(self._base_id)})
        self._leg_ids_sensor = torch.tensor(
            leg_ids, device=self.device, dtype=torch.long
        )

        # persistence counter for leg collisions
        self._leg_contact_counter = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )

        # gait-related state (for Raibert and feet clearance)
        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.clock_inputs = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_indices = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        # episode statistics
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",
                "raibert_heuristic",
                "orient",
                "lin_vel_z",
                "dof_vel",
                "ang_vel_xy",
                "feet_clearance",
                "contact_forces",
                "torque_l2",
                "leg_collision",
            ]
        }

        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------ #
    #  Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Feet positions in world frame: (num_envs, 4, 3)."""
        return self.robot.data.body_pos_w[:, self._feet_ids]

    # ------------------------------------------------------------------ #
    #  Scene setup
    # ------------------------------------------------------------------ #

    def _setup_scene(self):
        # robot and contact sensor
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        # register sensor in scene
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add robot
        self.scene.articulations["robot"] = self.robot

        # simple dome light
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    #  RL API
    # ------------------------------------------------------------------ #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # store actions in [-1, 1]
        self._actions = torch.clamp(actions, -1.0, 1.0)

        # convert actions to desired joint positions
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions
            + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        # PD torques: tau_PD = Kp (q_des - q) - Kd * q_dot
        pos_err = self.desired_joint_pos - self.robot.data.joint_pos
        vel = self.robot.data.joint_vel

        torques = self.Kp * pos_err - self.Kd * vel

        # ------------ Bonus 1: subtract actuator friction ------------
        if self.cfg.enable_actuator_friction:
            # static friction term: tau_stiction = F_s * tanh(qdot / 0.1)
            tau_stiction = self.f_s * torch.tanh(vel / 0.1)
            # viscous term: tau_viscous = mu_v * qdot
            tau_viscous = self.mu_v * vel
            tau_friction = tau_stiction + tau_viscous
            torques = torques - tau_friction
        # --------------------------------------------------------------

        # clamp by torque limits
        torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)

        self._last_torques = torques
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,                         # 3
                self.robot.data.root_ang_vel_b,                         # 3
                self.robot.data.projected_gravity_b,                    # 3
                self._commands,                                         # 3
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # 12
                self.robot.data.joint_vel,                              # 12
                self._actions,                                          # 12
                self.clock_inputs,                                      # 4
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # update gait phases and desired contacts
        self._step_contact_targets()

        # --- command tracking ---
        lin_vel_error = torch.sum(
            torch.square(
                self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]
            ),
            dim=1,
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(
            self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # --- action smoothness (first + second finite difference) ---
        rew_action_rate = torch.sum(
            torch.square(self._actions - self.last_actions[:, :, 0]),
            dim=1,
        )
        rew_action_rate += torch.sum(
            torch.square(
                self._actions
                - 2 * self.last_actions[:, :, 0]
                + self.last_actions[:, :, 1]
            ),
            dim=1,
        )
        rew_action_rate *= self.cfg.action_scale ** 2

        # roll action history
        self.last_actions = torch.roll(self.last_actions, shifts=1, dims=2)
        self.last_actions[:, :, 0] = self._actions

        # --- Raibert heuristic ---
        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # --- shaping: base orientation, vertical velocity, joint velocity, angular velocity ---
        rew_orient = torch.sum(
            torch.square(self.robot.data.projected_gravity_b[:, :2]),
            dim=1,
        )
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        rew_ang_vel_xy = torch.sum(
            torch.square(self.robot.data.root_ang_vel_b[:, :2]),
            dim=1,
        )

        # --- feet clearance during swing ---
        feet_height = self.foot_positions_w[:, :, 2]
        target_height = 0.10
        swing_mask = (self.desired_contact_states < 0.5).float()
        below_target = torch.clamp(target_height - feet_height, min=0.0)
        rew_feet_clearance = torch.sum(
            below_target * below_target * swing_mask, dim=1
        )

        # --- leg collision penalty (hips / thighs / calves) ---
        if self._leg_ids_sensor.numel() > 0:
            leg_forces = torch.linalg.norm(
                self._contact_sensor.data.net_forces_w[:, self._leg_ids_sensor, :],
                dim=-1,
            )
            leg_collision_pen = 1.0 - torch.exp(
                -(leg_forces * leg_forces) / self.cfg.leg_collision_force_denom
            )
            rew_leg_collision = torch.mean(leg_collision_pen, dim=1)
        else:
            rew_leg_collision = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float
            )

        # --- shaped foot contact forces ---
        foot_forces = torch.norm(
            self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :],
            dim=-1,
        )
        rew_contact_forces = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        for i in range(4):
            rew_contact_forces += -(
                1.0 - self.desired_contact_states[:, i]
            ) * (1.0 - torch.exp(-foot_forces[:, i] ** 2 / 100.0))
        rew_contact_forces = rew_contact_forces / 4.0

        # --- torque regularization ---
        torque_l2 = torch.sum(self._last_torques ** 2, dim=1)

        # collect all terms with scales from cfg
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped
            * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped
            * self.cfg.yaw_rate_reward_scale,
            "rew_action_rate": rew_action_rate
            * self.cfg.action_rate_reward_scale,
            "raibert_heuristic": rew_raibert_heuristic
            * self.cfg.raibert_heuristic_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "feet_clearance": rew_feet_clearance
            * self.cfg.feet_clearance_reward_scale,
            "contact_forces": rew_contact_forces
            * self.cfg.tracking_contacts_shaped_force_reward_scale,
            "torque_l2": torque_l2 * self.cfg.torque_l2_reward_scale,
            "leg_collision": rew_leg_collision * self.cfg.leg_collision_reward_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # accumulate episode sums
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Episode termination and timeout conditions."""

        # time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # base contact (large force on base link)
        net_forces = self._contact_sensor.data.net_forces_w
        base_forces = torch.norm(net_forces[:, self._base_id, :], dim=-1)
        if base_forces.ndim > 1:
            base_forces = base_forces.max(dim=1)[0]
        cstr_base_contact = base_forces > 1.0

        # upside-down: gravity points up in body frame
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0.0

        # base too low
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min

        # persistent leg collisions
        if self.cfg.terminate_on_leg_collision and self._leg_ids_sensor.numel() > 0:
            leg_forces = torch.linalg.norm(
                self._contact_sensor.data.net_forces_w[:, self._leg_ids_sensor, :],
                dim=-1,
            )
            leg_hit = torch.any(
                leg_forces > self.cfg.leg_collision_force_threshold, dim=1
            )

            self._leg_contact_counter = torch.where(
                leg_hit,
                self._leg_contact_counter + 1,
                torch.zeros_like(self._leg_contact_counter),
            )
            cstr_leg_collision = (
                self._leg_contact_counter
                >= self.cfg.leg_collision_consecutive_steps
            )
        else:
            cstr_leg_collision = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )

        died = (
            cstr_base_contact
            | cstr_upsidedown
            | cstr_base_height_min
            | cstr_leg_collision
        )

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # reset actions and torques
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self._last_torques[env_ids] = 0.0
        self._leg_contact_counter[env_ids] = 0

        # reset gait state
        self.gait_indices[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0
        self.foot_indices[env_ids] = 0.0

        # -------- Bonus 1: randomize actuator friction per episode --------
        if self.cfg.enable_actuator_friction:
            num_envs_reset = len(env_ids)
            action_dim = self._actions.shape[1]

            # viscous coefficient mu_v ~ U(mu_v_min, mu_v_max)
            self.mu_v[env_ids] = torch.empty(
                (num_envs_reset, action_dim), device=self.device
            ).uniform_(self.cfg.mu_v_min, self.cfg.mu_v_max)

            # stiction coefficient F_s ~ U(f_s_min, f_s_max)
            self.f_s[env_ids] = torch.empty(
                (num_envs_reset, action_dim), device=self.device
            ).uniform_(self.cfg.f_s_min, self.cfg.f_s_max)
        # ------------------------------------------------------------------

        # random commands in [-1, 1]
        self._commands[env_ids] = torch.zeros_like(
            self._commands[env_ids]
        ).uniform_(-1.0, 1.0)

        # reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]

        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # logging
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)

        extras = {}
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)

    # ------------------------------------------------------------------ #
    #  Gait helpers (Raibert)
    # ------------------------------------------------------------------ #

    def _step_contact_targets(self):
        """Advance gait phase and compute desired contacts and clock inputs."""
        frequency = 3.0
        self.gait_indices = torch.remainder(
            self.gait_indices + self.step_dt * frequency, 1.0
        )

        phase_offsets = torch.tensor(
            [0.0, 0.5, 0.5, 0.0], device=self.device
        ).unsqueeze(0)
        phases = torch.remainder(
            self.gait_indices.unsqueeze(1) + phase_offsets, 1.0
        )
        self.foot_indices = phases

        stance_portion = 0.6
        self.desired_contact_states[:] = (phases < stance_portion).float()

        self.clock_inputs[:] = torch.sin(2.0 * math.pi * phases)

    def _reward_raibert_heuristic(self):
        """Squared error between actual and desired Raibert foot placement."""
        cur_footsteps_translated = (
            self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        )
        footsteps_body = torch.zeros(
            self.num_envs, 4, 3, device=self.device
        )
        for i in range(4):
            footsteps_body[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w),
                cur_footsteps_translated[:, i, :],
            )

        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [
                desired_stance_width / 2,
                -desired_stance_width / 2,
                desired_stance_width / 2,
                -desired_stance_width / 2,
            ],
            device=self.device,
        ).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [
                desired_stance_length / 2,
                desired_stance_length / 2,
                -desired_stance_length / 2,
                -desired_stance_length / 2,
            ],
            device=self.device,
        ).unsqueeze(0)

        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) - 0.5
        frequency = 3.0
        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2.0

        desired_ys_offset = phases * y_vel_des * (0.5 / frequency)
        desired_ys_offset[:, 2:4] *= -1.0
        desired_xs_offset = phases * x_vel_des * (0.5 / frequency)

        desired_ys = desired_ys_nom + desired_ys_offset
        desired_xs = desired_xs_nom + desired_xs_offset

        desired_footsteps_body = torch.stack(
            [desired_xs, desired_ys], dim=2
        )  # (N, 4, 2)

        err = desired_footsteps_body - footsteps_body[:, :, 0:2]
        return torch.sum(err * err, dim=(1, 2))

    # ------------------------------------------------------------------ #
    #  Debug visualization
    # ------------------------------------------------------------------ #

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self._commands[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )

        self.goal_vel_visualizer.visualize(
            base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale
        )
        self.current_vel_visualizer.visualize(
            base_pos_w, vel_arrow_quat, vel_arrow_scale
        )

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(
            default_scale, device=self.device
        ).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat
