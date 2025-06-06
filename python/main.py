import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Parameters
steps = 10
nq = 2
nv = 2
nu = 2
dt = 0.02

ocg_res = 0.05
ocg_size = np.array([5, 5])
ocg_cells = np.array([int(ocg_size[0]/ocg_res) + 1, int(ocg_size[1]/ocg_res) + 1])
ocg_center = np.array([0, 0])

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
hsafe_data = np.loadtxt(hsafe_csv_path, delimiter=',')

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

# Plotting q trajectories
fig, ax = plt.subplots()
for q in q_list:
    ax.plot(q[0, :], q[1, :], linewidth=2)
    ax.scatter(q[0, 0], q[1, 0], color='black', zorder=5)

def plot_ocg(ax, data, ocg_res, ocg_cells, ocg_center, ocg_size):
    start_pos = np.array([ocg_center[0] - ocg_size[0]/2, ocg_center[1] - ocg_size[1]/2])
    for i in range(ocg_cells[0]):
        for j in range(ocg_cells[1]):
            if data[j*ocg_cells[0] + i] == 1:
                x = start_pos[0] + i * ocg_res
                y = start_pos[1] + j * ocg_res
                square = patches.Rectangle(
                    (x, y), ocg_res, ocg_res,
                    facecolor='black', alpha=0.5, edgecolor='none'
                )
                ax.add_patch(square)

def bilinear_interpolate(grid, i, j, ocg_cells):
    JMAX = ocg_cells[1]

    i1f = np.floor(i)
    j1f = np.floor(j)
    i2f = np.ceil(i)
    j2f = np.ceil(j)

    i1 = int(i1f)
    j1 = int(j1f)
    i2 = int(i2f)
    j2 = int(j2f)

    if i1 != i2 and j1 != j2:
        f1 = (i2f - i) * grid[i1 * JMAX + j1] + (i - i1f) * grid[i2 * JMAX + j1]
        f2 = (i2f - i) * grid[i1 * JMAX + j2] + (i - i1f) * grid[i2 * JMAX + j2]
        return (j2f - j) * f1 + (j - j1f) * f2

    elif i1 != i2:
        return (i2f - i) * grid[i1 * JMAX + int(j)] + (i - i1f) * grid[i2 * JMAX + int(j)]

    elif j1 != j2:
        return (j2f - j) * grid[int(i) * JMAX + j1] + (j - j1f) * grid[int(i) * JMAX + j2]

    else:
        return grid[int(i) * JMAX + int(j)]


def get_h0(grid, rx, ry, ocg_cells, ocg_res):
    # Fractional Index Corresponding to Current Position
    ir = ry / ocg_res
    jr = rx / ocg_res

    # Saturate indices to stay within bounds
    ic = np.clip(ir, 0.0, ocg_cells[0] - 1)
    jc = np.clip(jr, 0.0, ocg_cells[1] - 1)

    return bilinear_interpolate(grid, ic, jc, ocg_cells)

def plot_surface_from_1d(grid_1d, imax, jmax, dx=1.0, dy=1.0):
    """
    Plot a 3D surface from a 1D row-major grid.

    Parameters:
        grid_1d: 1D numpy array of length imax * jmax
        imax: number of rows
        jmax: number of columns
        dx, dy: spacing in x and y directions
    """
    # Reshape to 2D grid (row-major)
    grid_2d = grid_1d.reshape((imax, jmax))

    # Create coordinate mesh
    x = np.arange(jmax) * dx
    y = np.arange(imax) * dy
    X, Y = np.meshgrid(x, y)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, grid_2d, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    plt.title('Surface Plot from 1D Grid')
    plt.tight_layout()
    plt.show()

def plot_heatmap_from_1d(ax, grid_1d, imax, jmax, ocg_res):
    """
    Plot a 2D heatmap from a 1D row-major grid.

    Parameters:
        grid_1d: 1D numpy array of length imax * jmax
        imax: number of rows
        jmax: number of columns
        dx, dy: spatial resolution (used for axis labeling)
    """
    dx = ocg_res
    dy = ocg_res

    # Reshape to 2D grid (row-major)
    grid_2d = grid_1d.reshape((imax, jmax))

    # Create coordinate extents for proper axis scaling
    extent = [-(jmax * dx)/2, (jmax * dx)/2, -(imax * dy)/2, (imax * dy/2)]  # [xmin, xmax, ymin, ymax]

    ax.imshow(grid_2d, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    # plt.colorbar(label='Value')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('2D Heatmap from 1D Grid')
    # plt.grid(False)
    # plt.tight_layout()
    # plt.show()

plot_ocg(ax, ocg_data, ocg_res, ocg_cells, ocg_center, ocg_size)
plot_heatmap_from_1d(ax, hsafe_data, ocg_cells[0], ocg_cells[1], ocg_res)

ax.set_aspect('equal')
ax.grid(True)
ax.set_title('x vs y')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# plot_surface_from_1d(hsafe_data, ocg_cells[0], ocg_cells[1], dx=0.1, dy=0.1)
