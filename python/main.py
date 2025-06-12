import numpy as np
import matplotlib.pyplot as plt

from judo.controller import Controller, ControllerConfig
from judo.optimizers import MPPI, MPPIConfig

from safety_function import *
from certificates import *
from cost import *
from plotting import *
from planner_task import *

import yaml

with open("/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cpp/logs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Parameters
nq = 2
nv = 2
nu = 2

steps = config["steps"]
dt = config["dt"]

ocg_res = config["ocg_res"]
ocg_size = np.array([config["ocg_size_x"], config["ocg_size_y"]])
ocg_cells = np.array([int(ocg_size[0]/ocg_res) + 1, int(ocg_size[1]/ocg_res) + 1])
ocg_center = np.array([0, 0])
ocg_corner = ocg_center - (ocg_size/2)

# Obstacle data
obstacle1_pos = np.array([1.0, 0.5])
obstacle1_rad = 0.5
obstacle2_pos = np.array([0.3, 1.75])
obstacle2_rad = 0.35

# Load CSV data
rollout_csv_path = '/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cpp/logs/rollout_log.csv'
rollout_data = np.loadtxt(rollout_csv_path, delimiter=',')

occ_grid_csv_path = '/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cpp/logs/ocg_log.csv'
ocg_data = np.loadtxt(occ_grid_csv_path, delimiter=',')

hsafe_csv_path = '/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cpp/logs/hsafe_log.csv'
hsafe_data = np.loadtxt(hsafe_csv_path)
hsafe_2d = hsafe_data.reshape(ocg_cells[0], ocg_cells[1])
# Create grid in local coordinates (centered around 0,0)
x = np.linspace(-ocg_size[0]/2, ocg_size[0]/2, ocg_cells[0])
y = np.linspace(-ocg_size[1]/2, ocg_size[1]/2, ocg_cells[1])
xx, yy = np.meshgrid(x, y, indexing='ij')  # shape: (ocg_cells[0], ocg_cells[1])

# Stack into (N, 2) array of 2D positions
h_locations = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # shape: (N, 2)
h_locations += ocg_center

num_rows, num_cols = rollout_data.shape
expected_cols = 1 + steps * (nq + nv + nu)
assert num_cols == expected_cols, "Mismatch in expected CSV columns"

# Time vector
t = np.arange(steps) * dt

# Initialize data arrays
q_list = []
v_list = []
u_list = []

for row in range(num_rows):
    i_val = rollout_data[row, 0]  # Not used in this version, but available
    offset = 1  # Start after i_val

    q_flat = []
    v_flat = []
    u_flat = []

    for _ in range(steps):
        q_flat.extend(rollout_data[row, offset: offset + nq])
        offset += nq

        v_flat.extend(rollout_data[row, offset: offset + nv])
        offset += nv

    for _ in range(steps):
        u_flat.extend(rollout_data[row, offset: offset + nu])
        offset += nu

    q_list.append(np.reshape(q_flat, (steps, nq)).T)
    v_list.append(np.reshape(v_flat, (steps, nv)).T)
    u_list.append(np.reshape(u_flat, (steps, nu)).T)

# Make everything into a numpy array
q = np.stack(q_list)  # Shape: (num_rollouts, nq, steps)
v = np.stack(v_list)  # Shape: (num_rollouts, nv, steps)
u = np.stack(u_list)  # Shape: (num_rollouts, nu, steps)

# Plotting q trajectories
fig, ax = plt.subplots()
for i in range(q.shape[0]):
    ax.plot(q[i, 0, :], q[i, 1, :], linewidth=2)
    ax.scatter(q[i, 0, 0], q[i, 1, 0], color='black', zorder=5)

plot_ocg(ax, ocg_data, ocg_res, ocg_cells, ocg_center, ocg_size)
plot_heatmap_from_1d(ax, hsafe_2d, ocg_cells[0], ocg_cells[1], ocg_res)

ax.set_aspect('equal')
ax.grid(True)
ax.set_title('x vs y')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# Create and plot the CBF
cbf = create_cbf(q, hsafe_2d, ocg_cells, ocg_res, ocg_corner)

fig, ax = plt.subplots()
plot_cbf(ax, q, cbf)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('CBF Values')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# Create and plot the CLF
clf = create_clf(q, v, u, cost_function)
fig, ax = plt.subplots()
plot_clf(ax, q, clf)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('CLF Values')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# Plot h as a surface
# plot_surface_from_1d(hsafe_2d, ocg_cells, ocg_res, -(ocg_size - ocg_center)/2)

##
# Create the controller and make a plan ----- Nominal controller
##
nom_horizon = 3.
nom_opt_iters = 2
nom_sigma = 0.2
nom_temp = 0.01
nom_num_samples = 100
nom_use_noise_ramp = False
nom_num_nodes = 7
print(f"nominal horizon: {nom_horizon}\n"
      f"nominal opt iters: {nom_opt_iters}\n"
      f"nominal sigma: {nom_sigma}\n"
      f"nominal temp: {nom_temp}\n"
      f"nominal num_samples: {nom_num_samples}\n"
      f"nom_num_nodes: {nom_num_nodes}\n"
      f"nominal use_noise_ramp: {nom_use_noise_ramp}")
nom_ctrl_config = ControllerConfig(horizon=nom_horizon, spline_order="linear", max_opt_iters=nom_opt_iters, max_num_traces=1)
model_path = "/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/static_obstacle.xml"
nom_planner_task = DINavigation(model_path, model_path)
nom_planner_config = DINavigationConfig()
nom_optimizer_config = MPPIConfig(sigma=nom_sigma, temperature=nom_temp, num_rollouts=nom_num_samples,
                                  use_noise_ramp=nom_use_noise_ramp, num_nodes=nom_num_nodes)
optimizer = MPPI(nom_optimizer_config, 2)
nom_controller = Controller(nom_ctrl_config, nom_planner_task, nom_planner_config, optimizer, nom_optimizer_config)

state = np.array([1.75, -0.3, 0, 0])
time = 0.
nom_controller.update_action(state, time)
nom_action = nom_controller.action(time)
nom_states = nom_controller.states
nom_rewards = nom_controller.rewards

def _get_best_states(states, rewards):
    best_rollout = np.zeros((states.shape[1], states.shape[2]))
    best_reward = -np.inf
    for i in range(rewards.shape[0]):
        if rewards[i] > best_reward:
            best_reward = rewards[i]
            best_rollout = states[i, :, :]

    return best_rollout, best_reward

nom_rollout, nom_rew = _get_best_states(nom_states, nom_rewards)
print(f"nom_reward: {nom_rew}")

##
# Create the controller and make a plan ----- CLF/CBF controller
##
horizon = 3.
opt_iters = 2
sigma = 0.2
temp = 0.01
num_samples = 100
use_noise_ramp = False
num_nodes = 10
print(f"horizon: {horizon}\n"
      f"opt iters: {opt_iters}\n"
      f"sigma: {sigma}\n"
      f"temp: {temp}\n"
      f"num_samples: {num_samples}\n"
      f"num_nodes: {num_nodes}\n"
      f"use_noise_ramp: {use_noise_ramp}")
ctrl_config = ControllerConfig(horizon=horizon, spline_order="linear", max_opt_iters=opt_iters, max_num_traces=1)
model_path = "/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/static_obstacle.xml"
planner_task = DINavigationField(model_path, model_path)
planner_config = DINavigationFieldConfig(clf_values=clf, cbf_values=cbf, locations=q[:,:2,0],
                                         h_values=hsafe_data, h_locations=h_locations)
optimizer_config = MPPIConfig(sigma=sigma, temperature=temp, num_rollouts=num_samples, use_noise_ramp=use_noise_ramp,
                              num_nodes=num_nodes)
optimizer = MPPI(optimizer_config, 2)
controller = Controller(ctrl_config, planner_task, planner_config, optimizer, optimizer_config)

controller.update_action(state, time)
action = controller.action(time)
states = controller.states
rewards = controller.rewards

rollout, rew = _get_best_states(states, rewards)
print(f"reward: {rew}\n")

cbf_rew = planner_task.get_cbf_reward(planner_config, rollout[:,:2])

# print(f"all rewards: {rewards}")

# Plot the plan along with the obstacles
fig, ax = plt.subplots()
ax.plot(nom_rollout[:, 0], nom_rollout[:, 1], color="blue", label="nominal rollout")
ax.plot(rollout[:, 0], rollout[:, 1], color="black", label="rollout")
ax.scatter(rollout[0, 0], rollout[0, 1], color='black', zorder=5)

plot_ocg(ax, ocg_data, ocg_res, ocg_cells, ocg_center, ocg_size)

ax.set_aspect('equal')
ax.grid(True)
ax.set_title('x vs y')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

##
# Compute and plot CLF and CBF values along the trajectory
##
cbf_values = planner_task.interpolate_grid(planner_config.cbf_values, q[:, :2, 0], rollout[:, :2])
clf_values = planner_task.interpolate_grid(planner_config.clf_values, q[:, :2, 0], rollout[:, :2])

fig, ax = plt.subplots(3, 1, figsize=(8, 6))

ax[0].plot(clf_values, label="CLF over the rollout")
ax[0].grid(True)
ax[0].set_title('CLF Over Time')
ax[0].set_xlabel('Step')
ax[0].set_ylabel('CLF')

ax[1].plot(cbf_values, label="CBF over the rollout")
ax[1].grid(True)
ax[1].set_title('CBF Over Time')
ax[1].set_xlabel('Step')
ax[1].set_ylabel('CBF')

ax[2].plot(cbf_rew, label="CBF over the rollout")
ax[2].grid(True)
ax[2].set_title('CBF Reward Over Time')
ax[2].set_xlabel('Step')
ax[2].set_ylabel('CBF Reward')

plt.tight_layout()
plt.show()