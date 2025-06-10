import numpy as np

def bilinear_interpolate(grid_2d, i, j, ocg_cells):
    # JMAX = ocg_cells[1]

    i1f = np.floor(i)
    j1f = np.floor(j)
    i2c = np.ceil(i)
    j2c = np.ceil(j)

    i1 = int(i1f)
    j1 = int(j1f)
    i2 = int(i2c)
    j2 = int(j2c)

    if i1 != i2 and j1 != j2:
        f1 = (i2c - i) * grid_2d[i1, j1] + (i - i1f) * grid_2d[i2, j1]
        f2 = (i2c - i) * grid_2d[i1, j2] + (i - i1f) * grid_2d[i2, j2]
        # f1 = (i2c - i) * grid[i1 * JMAX + j1] + (i - i1f) * grid[i2 * JMAX + j1]
        # f2 = (i2c - i) * grid[i1 * JMAX + j2] + (i - i1f) * grid[i2 * JMAX + j2]
        return (j2c - j) * f1 + (j - j1f) * f2

    elif i1 != i2:
        # return (i2c - i) * grid[i1 * JMAX + int(j)] + (i - i1f) * grid[i2 * JMAX + int(j)]
        return (i2c - i) * grid_2d[i1, int(j)] + (i - i1f) * grid_2d[i2, int(j)]

    elif j1 != j2:
        # return (j2c - j) * grid[int(i) * JMAX + j1] + (j - j1f) * grid[int(i) * JMAX + j2]
        return (j2c - j) * grid_2d[int(i), j1] + (j - j1f) * grid_2d[int(i), j2]

    else:
        # return grid[int(i) * JMAX + int(j)]
        return grid_2d[int(i), int(j)]

def get_h(grid_2d, rx, ry, ocg_cells, ocg_res, ocg_corner):
    # Fractional Index Corresponding to Current Position
    ir = (ry - ocg_corner[0]) / ocg_res
    jr = (rx - ocg_corner[1]) / ocg_res

    # Saturate indices to stay within bounds
    ic = np.clip(ir, 0.0, ocg_cells[0] - 1)
    jc = np.clip(jr, 0.0, ocg_cells[1] - 1)

    return bilinear_interpolate(grid_2d, ic, jc, ocg_cells)