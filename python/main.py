import numpy as np
import matplotlib.pyplot as plt

from safety_function import *
from certificates import *
from cost import *
from plotting import *
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

