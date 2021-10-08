import time
from collections import deque
from pathlib import Path
from typing import Any, Generic

import numpy as np
from numpy.typing import DTypeLike
from torch.utils.tensorboard import SummaryWriter

from environments.base.environment import E, Environment


class MonitorEnv(Environment, Generic[E]):
    def __init__(self, env: E, log_dir: Path):
        self.env = env
        self.log_dir = log_dir
        self.tstart = time.time()
        self.score = 0.0
        self.episode_steps = 0
        self.total_steps = 0
        self.episodes = 0
        self.last_100_scores: deque[float] = deque(maxlen=100)
        self.writer: SummaryWriter = None  # type: ignore

    def reset(self) -> list[np.ndarray]:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        self.score = 0.0
        self.episode_steps = 0
        self.episodes += 1
        return self.env.reset()

    def step(self, action_idx: int) -> tuple[list[np.ndarray], float, bool, dict[str, Any]]:
        observation, reward, done, info = self.env.step(action_idx)
        self.score += reward
        self.episode_steps += 1
        self.total_steps += 1
        if done:
            self.last_100_scores.append(self.score)
            elapsed_hours = (time.time() - self.tstart) / 3600
            smoothed_score = sum(self.last_100_scores) / len(self.last_100_scores)
            self.writer.add_scalar("elapsed_hours", elapsed_hours, self.total_steps)
            self.writer.add_scalar("episode_steps", self.episode_steps, self.total_steps)
            self.writer.add_scalar("episodes", self.episodes, self.total_steps)
            self.writer.add_scalar("score", self.score, self.total_steps)
            self.writer.add_scalar("smoothed_score", smoothed_score, self.total_steps)
            for key, value in info.items():
                if np.isscalar(value):
                    self.writer.add_scalar(key, value, self.total_steps)
        return observation, reward, done, info

    def get_observation_space(self) -> list[tuple[list[int], DTypeLike]]:
        return self.env.get_observation_space()

    def visualize(self, *args, **kwargs) -> np.ndarray:
        return super().visualize(*args, **kwargs)
