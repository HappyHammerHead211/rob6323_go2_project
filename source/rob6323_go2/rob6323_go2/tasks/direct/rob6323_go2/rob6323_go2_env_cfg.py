# rob6323_go2/envs/rob6323_go2_env_cfg.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    action_scale = 0.25
    action_space = 12
    # 48 base obs + 4 gait clock inputs
    observation_space = 52
    state_space = 0
    debug_vis = True

    # base height termination
    base_height_min = 0.20

    # ---------------- Bonus 1: actuator friction params ----------------
    # enable friction model
    enable_actuator_friction = True
    # stiction coefficient range F_s ~ U(f_s_min, f_s_max)
    f_s_min = 0.0
    f_s_max = 2.5
    # viscous coefficient range mu_v ~ U(mu_v_min, mu_v_max)
    mu_v_min = 0.0
    mu_v_max = 0.3
    # -------------------------------------------------------------------

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

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
        debug_vis=False,
    )

    # custom PD control (stiffness/damping computed in env)
    Kp = 20.0
    Kd = 0.5
    torque_limits = 23.5

    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # disable built-in actuator PD, use torque control from env
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    # arrow size for debug visualization
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # command tracking
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    # action smoothness
    action_rate_reward_scale = -0.01

    # torque regularization
    torque_l2_reward_scale = -1e-5

    # gait / style shaping
    raibert_heuristic_reward_scale = -1.0
    feet_clearance_reward_scale = -10.0
    tracking_contacts_shaped_force_reward_scale = 2.0

    # base / joint shaping
    orient_reward_scale = -1.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -2e-5
    ang_vel_xy_reward_scale = -0.001

    # leg collision shaping
    leg_collision_reward_scale = -1.0
    leg_collision_force_denom = 100.0
    terminate_on_leg_collision = False
    leg_collision_force_threshold = 5.0
    leg_collision_consecutive_steps = 8
