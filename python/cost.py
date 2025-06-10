import numpy as np

def distance_to_goal(q, goal):
    """Returns the squared distance to the goal."""
    return np.square(q[:2, :] - goal).sum(-1).sum(-1)

def velocity_reg(v, target_v):
    """Returns the squared distance from the target to v."""
    return np.square(v[:2, :] - target_v).sum(-1).sum(-1)

def input_reg(u, target_u):
    """Returns the squared distance from the target to u."""
    return np.square(u[:, :] - target_u).sum(-1).sum(-1)

def cost_function(q, v, u):
    """Returns the value of the cost function.
    The inputs should be of shape [size, steps]
    """
    assert q.shape[1] == v.shape[1] == u.shape[1]

    q_target = np.zeros((2, q.shape[1]))
    v_target = np.zeros((2, v.shape[1]))
    u_target = np.zeros((2, u.shape[1]))

    return 5*distance_to_goal(q, q_target) + 0.1*velocity_reg(v, v_target) + 0.1*input_reg(u, u_target)