# ROB6323 Go2 Project — Isaac Lab

This repository is the starter code for the NYU Reinforcement Learning and Optimal Control project in which students train a Unitree Go2 walking policy in Isaac Lab starting from a minimal baseline and improve it via reward shaping and robustness strategies. Please read this README fully before starting and follow the exact workflow and naming rules below to ensure your runs integrate correctly with the cluster scripts and grading pipeline.

## Repository policy

- Fork this repository and do not change the repository name in your fork.  
- Your fork must be named rob6323_go2_project so cluster scripts and paths work without modification.

### Prerequisites

- **GitHub Account:** You must have a GitHub account to fork this repository and manage your code. If you do not have one, [sign up here](https://github.com/join).

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

## Connect to Greene

- Connect to the NYU Greene HPC via SSH; if you are off-campus or not on NYU Wi‑Fi, you must connect through the NYU VPN before SSHing to Greene.  
- The official instructions include example SSH config snippets and commands for greene.hpc.nyu.edu and dtn.hpc.nyu.edu as well as VPN and gateway options: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0#h.7t97br4zzvip.

## Clone in $HOME

After logging into Greene, `cd` into your home directory (`cd $HOME`). You must clone your fork into `$HOME` only (not scratch or archive). This ensures subsequent scripts and paths resolve correctly on the cluster. Since this is a private repository, you need to authenticate with GitHub. You have two options:

### Option A: Via VS Code (Recommended)
The easiest way to avoid managing keys manually is to configure **VS Code Remote SSH**. If set up correctly, VS Code forwards your local credentials to the cluster.
- Follow the [NYU HPC VS Code guide](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code) to set up the connection.

> **Tip:** Once connected to Greene in VS Code, you can clone directly without using the terminal:
> 1. **Sign in to GitHub:** Click the "Accounts" icon (user profile picture) in the bottom-left sidebar. If you aren't signed in, click **"Sign in with GitHub"** and follow the browser prompts to authorize VS Code.
> 2. **Clone the Repo:** Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type **Git: Clone**, and select it.
> 3. **Select Destination:** When prompted, select your home directory (`/home/<netid>/`) as the clone location.
>
> For more details, see the [VS Code Version Control Documentation](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git#_clone-a-repository-locally).

### Option B: Manual SSH Key Setup
If you prefer using a standard terminal, you must generate a unique SSH key on the Greene cluster and add it to your GitHub account:
1. **Generate a key:** Run the `ssh-keygen` command on Greene (follow the official [GitHub documentation on generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)).
2. **Add the key to GitHub:** Copy the output of your public key (e.g., `cat ~/.ssh/id_ed25519.pub`) and add it to your account settings (follow the [GitHub documentation on adding a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)).

### Execute the Clone
Once authenticated, run the following commands. Replace `<your-git-ssh-url>` with the SSH URL of your fork (e.g., `git@github.com:YOUR_USERNAME/rob6323_go2_project.git`).
```
cd $HOME
git clone <your-git-ssh-url> rob6323_go2_project
```
*Note: You must ensure the target directory is named exactly `rob6323_go2_project`. This ensures subsequent scripts and paths resolve correctly on the cluster.*
## Install environment

- Enter the project directory and run the installer to set up required dependencies and cluster-side tooling.  
```
cd $HOME/rob6323_go2_project
./install.sh
```
Do not skip this step, as it configures the environment expected by the training and evaluation scripts. It will launch a job in burst to set up things and clone the IsaacLab repo inside your greene storage. You must wait until the job in burst is complete before launching your first training. To check the progress of the job, you can run `ssh burst "squeue -u $USER"`, and the job should disappear from there once it's completed. It takes around **30 minutes** to complete. 
You should see something similar to the screenshot below (captured from Greene):

![Example burst squeue output](docs/img/burst_squeue_example.png)

In this output, the **ST** (state) column indicates the job status:
- `PD` = pending in the queue (waiting for resources).
- `CF` = instance is being configured.
- `R`  = job is running.

On burst, it is common for an instance to fail to configure; in that case, the provided scripts automatically relaunch the job when this happens, so you usually only need to wait until the job finishes successfully and no longer appears in `squeue`.

## What to edit

- In this project you'll only have to modify the two files below, which define the Isaac Lab task and its configuration (including PPO hyperparameters).  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py
PPO hyperparameters are defined in source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/agents/rsl_rl_ppo_cfg.py, but you shouldn't need to modify them.

## How to edit

- Option A (recommended): Use VS Code Remote SSH from your laptop to edit files on Greene; follow the NYU HPC VS Code guide and connect to a compute node as instructed (VPN required off‑campus) (https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code). If you set it correctly, it makes the login process easier, among other things, e.g., cloning a private repo.
- Option B: Edit directly on Greene using a terminal editor such as nano.  
```
nano source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py
```
- Option C: Develop locally on your machine, push to your fork, then pull changes on Greene within your $HOME/rob6323_go2_project clone.

> **Tip:** Don't forget to regularly push your work to github

## Launch training

- From $HOME/rob6323_go2_project on Greene, submit a training job via the provided script.  
```
cd "$HOME/rob6323_go2_project"
./train.sh
```
- Check job status with SLURM using squeue on the burst head node as shown below.  
```
ssh burst "squeue -u $USER"
```
Be aware that jobs can be canceled and requeued by the scheduler or underlying provider policies when higher-priority work preempts your resources, which is normal behavior on shared clusters using preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

Use `rsync` to copy results from the cluster to your local machine. It is faster and can resume interrupted transfers. Run this on your machine (NOT on Greene):

```
rsync -avzP -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./
```

*Explanation of flags:*
- `-a`: Archive mode (preserves permissions, times, and recursive).
- `-v`: Verbose output.
- `-z`: Compresses data during transfer (faster over network).
- `-P`: Shows progress bar and allows resuming partial transfers.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rob6323_go2_project` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.

## Project scope reminder

- The assignment expects you to go beyond velocity tracking by adding principled reward terms (posture stabilization, foot clearance, slip minimization, smooth actions, contact and collision penalties), robustness via domain randomization, and clear benchmarking metrics for evaluation as described in the course guidelines.  
- Keep your repository organized, document your changes in the README, and ensure your scripts are reproducible, as these factors are part of grading alongside policy quality and the short demo video deliverable.

## Resources

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).

---
Students should only edit README.md below this line.

# ROB-GY 6323 – Unitree Go2 Locomotion in Isaac Lab

This repository contains the course project for  
**“Reinforcement Learning and Optimal Control for Autonomous Systems I”**  
(Prof. Ludovic Righetti, NYU Tandon).

We train a **Unitree Go2** quadruped locomotion policy in **Isaac Lab** using **RSL-RL PPO**, starting from the provided velocity-tracking baseline and progressively adding:

- Action-rate regularization (smooth actions)  
- Torque-level PD control (custom low-level controller)  
- Early termination on base height  
- Raibert-style gait clock and foot-placement shaping  
- Additional shaping on base orientation, vertical velocity, joint velocity, feet clearance, contact forces, and leg collisions  
- **Bonus 1:** simple actuator friction model with per-episode randomization

The final policies exhibit smoother torques, more stable base motion, and clearer trotting-like gaits on flat terrain, with an additional friction-robust variant for Bonus 1.

---

## 1. Repository Layout

Main files modified/added for the project:

- `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/`
  - `rob6323_go2_env_cfg.py`  
    Environment configuration (Go2 asset, terrain, sensors, reward scales, PD gains, termination, Bonus 1 friction parameters).
  - `rob6323_go2_env.py`  
    Task implementation:
    - Observation construction (52-dim observation including gait clock)
    - Torque-level PD control
    - Reward terms (tracking, action smoothness, orientation, feet clearance, collisions, etc.)
    - Termination conditions
    - **Bonus 1:** friction torque computation and per-episode randomization

Training is launched via the standard Isaac Lab / RSL-RL entrypoint for this task (same as in the class template), with log files written under:

- `logs/rsl_rl/go2_flat_direct/...`

---

## 2. Environment and Control

Key environment settings (see `rob6323_go2_env_cfg.py`):

- 4096 parallel environments
- Physics timestep: `dt = 1/200 s`
- Control frequency: `50 Hz` (decimation = 4)
- Flat terrain with friction coefficient 1.0
- Contact sensor on all robot links, used for feet/contact rewards
- Action space: 12-D, joint position offsets in `[-1, 1]` scaled by `action_scale = 0.25`
- Observation space: 52-D
  - Base linear / angular velocity
  - Projected gravity
  - Commanded planar velocities `(v_x, v_y, yaw rate)`
  - Joint positions/velocities
  - Previous action
  - 4-D gait clock

**Custom PD control**

- Desired joint positions = nominal stance + scaled action
- PD torque: `tau = Kp * (q_des - q) - Kd * qd`
- Gains: `Kp = 20.0`, `Kd = 0.5`
- Torque limits: `±23.5 Nm`
- Built-in actuator stiffness/damping is disabled via `ImplicitActuatorCfg`.

---

## 3. Reward Design

The total reward is a weighted sum of the following terms (all implemented in `rob6323_go2_env.py` and configured via `rob6323_go2_env_cfg.py`):

### 3.1 Command Tracking

- Linear velocity tracking in the XY plane  
  - Weighted by `lin_vel_reward_scale = 1.0`
  - Logged as `Episode_Reward/track_lin_vel_xy_exp`
- Yaw-rate tracking  
  - Weighted by `yaw_rate_reward_scale = 0.5`
  - Logged as `Episode_Reward/track_ang_vel_z_exp`

### 3.2 Action Smoothness and Torque Regularization

- Action-rate penalty on successive actions (first and second differences)  
  - Weighted by `action_rate_reward_scale = -0.01`
- Torque L2 penalty on joint torques  
  - Weighted by `torque_l2_reward_scale = -1e-5`

These terms encourage smooth, low-frequency control signals without overwhelming the tracking rewards.

### 3.3 Gait and Posture Shaping

- **Raibert heuristic** (`raibert_heuristic_reward_scale = -1.0`)  
  Penalizes deviation between actual foot positions and Raibert-style desired foot placements given gait phase and command.
- **Orientation penalty** (`orient_reward_scale = -1.0`)  
  Encourages a level base (small roll/pitch).
- **Vertical velocity penalty** (`lin_vel_z_reward_scale = -0.02`)  
  Reduces bouncing in the base height.
- **Joint velocity penalty** (`dof_vel_reward_scale = -2e-5`)
- **Roll/pitch rate penalty** (`ang_vel_xy_reward_scale = -0.001`)

### 3.4 Feet Clearance, Contacts, and Leg Collisions

- **Feet clearance** (`feet_clearance_reward_scale = -10.0`)  
  Penalizes feet that are too low during swing phase (encourages consistent swing height).
- **Contact forces** (`tracking_contacts_shaped_force_reward_scale = 2.0`)  
  Shaped reward that prefers appropriate contact timing and magnitude.
- **Leg collisions**  
  - Penalty `leg_collision_reward_scale = -1.0` applied when hip/thigh/calf links see large contact forces.
  - Shaping denominator `leg_collision_force_denom = 100.0` controls sensitivity.
  - Optional early termination switches:
    - `terminate_on_leg_collision`
    - `leg_collision_force_threshold`
    - `leg_collision_consecutive_steps`

---

## 4. Bonus 1: Actuator Friction with Randomization

Bonus 1 adds a simple actuator friction model to make joint dynamics more realistic and to test robustness of the learned gait.

### 4.1 Configuration Parameters

In `rob6323_go2_env_cfg.py`:

```python
# Bonus 1: actuator friction parameters
enable_actuator_friction = True

# Randomization ranges (sampled per environment at reset)
f_s_min = 0.0   # static friction min
f_s_max = 2.5   # static friction max
mu_v_min = 0.0  # viscous friction min
mu_v_max = 0.3  # viscous friction max
