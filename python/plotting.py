import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_ocg(ax, data, ocg_res, ocg_cells, ocg_center, ocg_size):
    start_pos = np.array([ocg_center[0] - ocg_size[0]/2, ocg_center[1] - ocg_size[1]/2])
    for i in range(ocg_cells[0]):
        for j in range(ocg_cells[1]):
            if data[j*ocg_cells[0] + i] == 0:
                x = start_pos[0] + i * ocg_res
                y = start_pos[1] + j * ocg_res
                square = patches.Rectangle(
                    (x, y), ocg_res, ocg_res,
                    facecolor='black', alpha=0.5, edgecolor='none'
                )
                ax.add_patch(square)

def plot_surface_from_1d(grid, ocg_cells, ocg_res, corner):
    """
    Plot a 3D surface from a 1D row-major grid.

    Parameters:
        grid_1d: 1D numpy array of length imax * jmax
        imax: number of rows
        jmax: number of columns
        dx, dy: spacing in x and y directions
    """
    # Create coordinate mesh
    x = np.arange(ocg_cells[1]) * ocg_res + corner[0]
    y = np.arange(ocg_cells[0]) * ocg_res + corner[1]
    X, Y = np.meshgrid(x, y)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    plt.title('Surface Plot from 1D Grid')
    plt.tight_layout()
    plt.show()

def plot_heatmap_from_1d(ax, grid_2d, imax, jmax, ocg_res):
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

def plot_cbf(ax, q, cbf):
    for i in range(q.shape[0]):
        ax.plot(q[i, 0, :], q[i, 1, :], linewidth=2)
    sc = ax.scatter(q[:, 0, 0], q[:, 1, 0], c=cbf, cmap='viridis', s=220)
    plt.colorbar(sc, ax=ax, label='Value')


def plot_clf(ax, q, clf):
    for i in range(q.shape[0]):
        ax.plot(q[i, 0, :], q[i, 1, :], linewidth=2)
    sc = ax.scatter(q[:, 0, 0], q[:, 1, 0], c=clf, cmap='viridis', s=220)
    plt.colorbar(sc, ax=ax, label='Value')
