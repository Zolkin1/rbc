from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.interpolate import LinearNDInterpolator

import mujoco

from judo.tasks import TaskConfig, Task
from judo.utils.fields import np_1d_field
from judo.tasks.cost_functions import quadratic_norm

@dataclass
class DINavigationConfig(TaskConfig):
    """Reward configuration for the cylinder push task."""

    w_robot_pos: float = 0.5
    w_pusher_velocity: float = 0.1
    goal_pos: np.ndarray = np_1d_field(
        np.array([0., 0.]),
        names=["x", "y"],
        mins=[-5.0, -5.0],
        maxs=[5.0, 5.0],
        steps=[0.01, 0.01],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
        xyz_vis_defaults=[0.0, 0.0, 0.0],
    )

class DINavigation(Task[DINavigationConfig]):
    """Defines the cylinder push balancing task."""

    def __init__(self, model_path: str, sim_model_path: str | None = None) -> None:
        """Initializes the cylinder push task."""
        super().__init__(model_path, sim_model_path=sim_model_path)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: DINavigationConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the cylinder push reward from MJPC.

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The cylinder push reward has four terms:
            * `pusher_reward`, penalizing the distance between the pusher and the cart.
            * `velocity_reward` penalizing squared linear velocity of the pusher.
            * `goal_reward`, penalizing the distance from the cart to the goal.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        robot_pos = states[..., 0:2]
        robot_vel = states[..., 3:5]
        robot_goal = config.goal_pos[0:2]

        velocity_reward = -config.w_pusher_velocity * quadratic_norm(robot_vel).sum(-1)

        robot_to_goal = robot_goal - robot_pos
        goal_reward = -config.w_robot_pos * quadratic_norm(robot_to_goal).sum(-1)

        return goal_reward + velocity_reward

    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        theta = 2 * np.pi * np.random.rand(2)
        self.data.qpos = np.array(
            [
                np.cos(theta[0]),
                np.sin(theta[0]),
                # 2 * np.cos(theta[1]),
                # 2 * np.sin(theta[1]),
            ]
        )
        self.data.qvel = np.zeros(2)
        mujoco.mj_forward(self.model, self.data)

@dataclass
class DINavigationFieldConfig(TaskConfig):
    """Reward configuration for the cylinder push task."""

    clf_values: np.array
    cbf_values: np.array
    locations: np.ndarray

    h_values: np.array
    h_locations: np.ndarray

    cbf_alpha: float = 1
    w_robot_pos: float = 0.5
    w_pusher_velocity: float = 0.1
    goal_pos: np.ndarray = np_1d_field(
        np.array([0., 0.]),
        names=["x", "y"],
        mins=[-5.0, -5.0],
        maxs=[5.0, 5.0],
        steps=[0.01, 0.01],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
        xyz_vis_defaults=[0.0, 0.0, 0.0],
    )

class DINavigationField(Task[DINavigationFieldConfig]):
    """Defines the cylinder push balancing task."""

    def __init__(self, model_path: str, sim_model_path: str | None = None) -> None:
        """Initializes the cylinder push task."""
        super().__init__(model_path, sim_model_path=sim_model_path)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: DINavigationFieldConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the cylinder push reward from MJPC.

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The cylinder push reward has four terms:
            * `pusher_reward`, penalizing the distance between the pusher and the cart.
            * `velocity_reward` penalizing squared linear velocity of the pusher.
            * `goal_reward`, penalizing the distance from the cart to the goal.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        robot_pos = states[..., 0:2]
        robot_vel = states[..., 3:5]
        robot_goal = config.goal_pos[0:2]

        velocity_reward = -config.w_pusher_velocity * quadratic_norm(robot_vel).sum(-1)

        robot_to_goal = robot_goal - robot_pos
        goal_reward = -config.w_robot_pos * quadratic_norm(robot_to_goal).sum(-1)

        clf_value_reward = np.zeros(states.shape[0])
        clf_diff_reward = np.zeros(states.shape[0])
        cbf_cond_reward = np.zeros(states.shape[0])
        cbf_value_reward = np.zeros(states.shape[0])
        for i in range(clf_value_reward.shape[0]):
            # CLF Reward (minimize the CLF values)
            clf_value_reward[i] = -self.get_clf_value_reward(config, robot_pos[i, :]).sum()
            # CLF Reward (minimize the CLF increases along a trajectory)
            clf_diff_reward[i] = self.get_clf_reward(config, robot_pos[i, :]).sum()
            # CBF Reward (avoid areas with values that violate the condition)
            cbf_cond_reward[i] = -self.get_cbf_reward(config, robot_pos[i, :]).sum()
            # CBF Value Reward
            cbf_value_reward[i] = self.get_cbf_value_reward(config, robot_pos[i, :]).sum()

        # return clf_diff_reward + clf_value_reward + cbf_cond_reward
        return clf_value_reward + cbf_cond_reward + 10*cbf_value_reward + clf_diff_reward

    def get_clf_value_reward(self, task_config: DINavigationFieldConfig, positions: np.ndarray) -> np.ndarray:
        """Computes the CLF reward. For now, it is just based on the value and not the gradient."""
        clf_values = self.interpolate_grid(task_config.clf_values, task_config.locations, positions)

        return clf_values * self.model.opt.timestep

    def get_clf_reward(self, task_config: DINavigationFieldConfig, positions: np.ndarray) -> np.ndarray:
        """Penalty for a increasing CLF value."""
        clf_values = self.interpolate_grid(task_config.clf_values, task_config.locations, positions)

        # Compute the difference in the CLF values
        clf_diff = np.diff(clf_values)

        # If the difference is negative then the value increased, so penalize it
        clf_clamp = np.minimum(np.zeros(clf_diff.shape[0]), clf_diff)

        # Penalize these differences
        return clf_clamp

    def get_cbf_value_reward(self, task_config: DINavigationFieldConfig, positions: np.ndarray) -> np.ndarray:
        """Computes the CBF reward. For now, it is just based on the value and not the gradient."""
        cbf_values = np.minimum(self.interpolate_grid(task_config.clf_values, task_config.locations, positions),
                                self.interpolate_grid(task_config.h_values, task_config.h_locations, positions))

        # Only penalize negative values
        cbf_clamp = np.minimum(np.zeros(cbf_values.shape[0]), cbf_values)

        return cbf_clamp

    def get_cbf_reward(self, task_config: DINavigationFieldConfig, positions: np.ndarray) -> np.ndarray:
        """Penality for violating the CBF condition."""
        cbf_values = np.minimum(self.interpolate_grid(task_config.clf_values, task_config.locations, positions),
                                self.interpolate_grid(task_config.h_values, task_config.h_locations, positions))

        # Compute the difference in the CLF values
        cbf_diff = np.diff(cbf_values)

        cbf_rhs = -task_config.cbf_alpha * cbf_values

        # If the difference is negative then the value increased, so penalize it
        cbf_clamp = 1000*np.sign(np.maximum(np.zeros(cbf_diff.shape[0]), cbf_diff - cbf_rhs[0:-1]))

        # Penalize these differences
        return cbf_clamp

    def interpolate_grid(self, values, val_locations, positions) -> np.ndarray:
        """Interpolates the grid at the given positions.
            values: (N)
            val_locations: (N, 2)
            positions: (M, 2)

            Returns:
                (M,) array of interpolated values.
        """
        # Interpolate using linear method
        interp = LinearNDInterpolator(
            points=val_locations,
            values=values,
            fill_value=10000,
        )

        interpolated = interp(positions)

        return interpolated


    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        theta = 2 * np.pi * np.random.rand(2)
        self.data.qpos = np.array(
            [
                np.cos(theta[0]),
                np.sin(theta[0]),
                # 2 * np.cos(theta[1]),
                # 2 * np.sin(theta[1]),
            ]
        )
        self.data.qvel = np.zeros(2)
        mujoco.mj_forward(self.model, self.data)