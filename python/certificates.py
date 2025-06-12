from safety_function import *

def create_clf(q, v, u, cost_fcn):
    """Creates a CLF for all the points given their rollouts.
    q is of shape [rollouts, nq, steps]
    v is of shape [rollouts, nv, steps]
    u is of shape [rollouts, nu, steps]
    """

    assert q.shape[0] == v.shape[0] == u.shape[0]

    cost = np.zeros(q.shape[0])

    for i in range(q.shape[0]):
        cost[i] = cost_fcn(q[i], v[i], u[i])

    return cost

def create_cbf(q, hsafe, ocg_cells, ocg_res, ocg_corner):
    """Creates a CBF for all the points given their rollouts"""

    h = np.zeros(q.shape[0])
    # Iterate through the list
    for i in range(q.shape[0]):
        # Evaluate h at each point
        hmin = 1000
        for j in range(q.shape[2]):
            hcand = get_h(hsafe, q[i, 0, j], q[i, 1, j], ocg_cells, ocg_res, ocg_corner)
            if  hcand < hmin:
                hmin = hcand

        h[i] = hmin
    return h