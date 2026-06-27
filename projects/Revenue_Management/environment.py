import random
from typing import Any, Literal, NamedTuple, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


class EnvData(NamedTuple):
    service_id: int
    demand_matrix: np.ndarray
    prices: np.ndarray
    initial_capacity: int


class RevenueManagementEnv(gym.Env[np.ndarray, int]):
    """
    A custom environment for revenue management using reinforcement learning.

    The Environment can take multiple demand matrices as input and randomly draw one at the start of each episode.

    Action space:
    """

    def __init__(self, data: list[EnvData], max_steps: int = 500):
        super().__init__()

        self.data = data
        self.max_steps = max_steps

        n_buckets, n_day_xs = np.shape(self.data[0].demand_matrix)
        for n in range(1, len(data)):
            if np.shape(self.data[n].demand_matrix) != (n_buckets, n_day_xs):
                raise ValueError("All demand_matrices in data must have the same shape.")

        self.n_buckets, self.n_day_xs = n_buckets, n_day_xs

        self.action_space = None  # TODO: implement an action space!
        self.observation_space = None  # TODO: implement an observation space!

        # EPISODE-SPECIFIC VARIABLES (must be reset between episodes)
        self.current_step = 0
        self.availabilities = np.zeros((n_buckets, n_day_xs), dtype=int)
        self.initial_availabilities = self.availabilities.copy()
        self.revenue = 0.0

        self._setup_data(0)

    def _setup_data(self, n: int) -> None:
        self.service_id = self.data[n].service_id
        self.prices = self.data[n].prices
        self.demand_matrix = self.data[n].demand_matrix

        self.initial_capacity = self.data[n].initial_capacity
        self.remaining_capacity = self.initial_capacity

        self.revenue_matrix = np.multiply(self.demand_matrix, np.expand_dims(self.prices, 1))
        self.max_bucket_revenue = np.max(self.revenue_matrix)
        self.availabilities = np.zeros(np.shape(self.demand_matrix), dtype=int)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment"""
        n = random.randint(0, len(self.data) - 1)  # noqa
        self._setup_data(n)
        self.revenue = 0.0
        self.current_step = 0
        self.visited_states = {}

        return self._get_obs(), {}

    def _clip_bucket(self, bucket: int) -> int:
        return min(max(bucket, 0), self.n_buckets - 1)

    def _clip_day_x(self, day_x: int) -> int:
        return min(max(day_x, 0), self.n_day_xs - 1)

    def _clip_indexes(self, bucket: int, day_x: int) -> tuple[int, int]:
        return self._clip_bucket(bucket), self._clip_day_x(day_x)

    def _get_obs(self) -> NDArray[np.float64]:
        """This function returns the observation."""
        raise NotImplementedError("You need to implement the observation in this function")

    def _select_bucket_day_x(self, bucket: int, day_x: int) -> tuple[int, float]:
        """Select the given bucket and day_x, updating availabilities, remaining capacity, and revenue."""

        if self.demand_matrix[bucket, day_x] > self.remaining_capacity:
            raise ValueError("Invalid action: not enough remaining capacity.")

        self.availabilities[bucket, day_x] = self.demand_matrix[bucket, day_x]
        self.remaining_capacity -= self.demand_matrix[bucket, day_x]
        self.revenue += self.revenue_matrix[bucket, day_x]

        return self.demand_matrix[bucket, day_x], self.revenue_matrix[bucket, day_x]

    def get_trajectory_head(self) -> tuple[int, int]:
        """Returns the (bucket, day_x) of the trajectory head."""
        day_x = int(np.max(np.where(self.availabilities.sum(axis=0) > 0)))
        bucket = np.where(self.availabilities[:, day_x] > 0)[0][0]
        return bucket, day_x

    def step(self, action: int) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self.current_step += 1

        # TODO: implement action handling!

        truncated = self.current_step >= self.max_steps
        terminated = False  # TODO: implement termination condition

        if self.remaining_capacity < 0:
            raise ValueError("Remaining capacity cannot be negative.")

        if self.remaining_capacity != self.initial_capacity - np.sum(self.availabilities):
            raise ValueError("Remaining capacity does not match availabilities.")

        reward = 0.0  # TODO: implement a reward!

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {},  # info
        )
