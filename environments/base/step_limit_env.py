from typing import Any, Generic

import numpy as np
from numpy.typing import DTypeLike

from environments.base.environment import E, Environment


class StepLimitEnv(Environment, Generic[E]):
    def __init__(self, env: E, max_steps: int) -> None:
        self.env = env
        self.max_steps = max_steps
        self.current_steps = 0

    def reset(self) -> list[np.ndarray]:
        self.current_steps = 0
        observation = self.env.reset()
        observation.append(self._get_progress())
        return observation

    def step(self, action_idx: int) -> tuple[list[np.ndarray], float, bool, dict[str, Any]]:
        self.current_steps += 1
        observation, reward, done, info = self.env.step(action_idx)
        observation.append(self._get_progress())
        if self.current_steps >= self.max_steps:
            done = True
        return observation, reward, done, info

    def get_observation_space(self) -> list[tuple[list[int], DTypeLike]]:
        observation_space = self.env.get_observation_space()
        observation_space.append(([1], np.float32))
        return observation_space

    def visualize(self, *args, **kwargs) -> np.ndarray:
        return super().visualize(*args, **kwargs)

    def _get_progress(self) -> np.ndarray:
        return np.array([self.current_steps / self.max_steps], dtype=np.float32)
