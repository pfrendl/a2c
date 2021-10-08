from multiprocessing.connection import Connection
from typing import Generic

import numpy as np
import torch
from torch import Tensor
from torch.multiprocessing import Pipe, Process

from environments.base.environment import E, Environment


class BatchEnv(Generic[E]):
    def __init__(self, envs: list[E]) -> None:
        self.envs = envs
        self.num_workers = len(envs)
        self.observation_sh: list[Tensor] = []
        self.action_idx_sh = torch.zeros((0,), dtype=torch.int32)
        self.reward_sh = torch.zeros((0, 1))
        self.done_sh = torch.zeros((0, 1), dtype=torch.bool)
        self.workers: list[tuple[Connection, Process]] = []

    def reset(self) -> list[Tensor]:
        self.stop()

        observation_space = self.envs[0].get_observation_space()
        self.observation_sh = [
            torch.zeros((self.num_workers, *shape), dtype=getattr(torch, np.dtype(dtype).name)).share_memory_()
            for shape, dtype in observation_space
        ]
        self.action_idx_sh = torch.zeros((self.num_workers,), dtype=torch.int32).share_memory_()
        self.reward_sh = torch.zeros((self.num_workers, 1)).share_memory_()
        self.done_sh = torch.ones((self.num_workers, 1), dtype=torch.bool).share_memory_()

        self.workers = []
        for worker_idx, env in enumerate(self.envs):
            parent_conn, child_conn = Pipe()
            proc_args = (
                worker_idx,
                self.observation_sh,
                self.action_idx_sh,
                self.reward_sh,
                self.done_sh,
                child_conn,
                env,
            )
            p = Process(target=work, args=proc_args)
            p.start()
            self.workers.append((parent_conn, p))

        for worker in self.workers:
            worker[0].recv()

        return self.observation_sh

    def step(self, action_idx: Tensor) -> tuple[list[Tensor], Tensor, Tensor]:
        self.action_idx_sh.copy_(action_idx)
        for worker in self.workers:
            worker[0].send(True)
        for worker in self.workers:
            worker[0].recv()
        return self.observation_sh, self.reward_sh, self.done_sh

    def stop(self) -> None:
        for worker in self.workers:
            worker[0].send(False)
        for worker in self.workers:
            worker[1].join()
        self.workers = []


def work(
    worker_idx: int,
    observation_sh: list[Tensor],
    action_idx_sh: Tensor,
    reward_sh: Tensor,
    done_sh: Tensor,
    conn: Connection,
    env: Environment,
) -> None:
    loop = True
    while loop:
        if done_sh[worker_idx]:
            observation = env.reset()
            for obs_sh, obs in zip(observation_sh, observation):
                obs_sh[worker_idx, : obs.shape[0]] = torch.from_numpy(obs)

        conn.send(True)
        loop = conn.recv()

        action_idx: int = action_idx_sh[worker_idx].item()  # type: ignore
        observation, reward, done, _ = env.step(action_idx=action_idx)
        for obs_sh, obs in zip(observation_sh, observation):
            obs_sh[worker_idx, : obs.shape[0]] = torch.from_numpy(obs)
        reward_sh[worker_idx] = reward
        done_sh[worker_idx] = done
