from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import DTypeLike

from environments.base.environment import Environment


class BlobEnv(Environment):
    def __init__(self) -> None:
        self.radii = 0.02
        self.resistance = 0.01
        self.acceleration = 1.0
        self.delta_t = 0.05

        self.num_enemies = 10
        self.num_collectibles = 10
        self.num_blobs = 1 + self.num_enemies + self.num_collectibles

        self.positions = np.zeros((0, 2))
        self.velocities = np.zeros((0, 2))
        self.types = np.zeros((0, 2))
        self.type_rewards = np.array([0.0, -1.0, 1.0])

        delta_v = self.delta_t * self.acceleration
        self.actions = np.array(
            [
                [0.0, 0.0],
                [-delta_v, 0.0],
                [0.0, -delta_v],
                [delta_v, 0.0],
                [0.0, delta_v],
            ]
        )

        self.colors = [
            [255, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
        ]

    def reset(self) -> list[np.ndarray]:
        self.positions = np.random.uniform(low=-1.0, high=1.0, size=(self.num_blobs, 2))
        self.velocities = np.zeros_like(self.positions)
        self.types = np.concatenate(
            [
                [0],
                np.full((self.num_enemies,), 1),
                np.full((self.num_collectibles,), 2),
            ],
            axis=0,
        )
        return self._create_observation()

    def step(self, action: int) -> tuple[list[np.ndarray], float, bool, dict[str, Any]]:
        action_idxs = np.concatenate(
            [
                [action],
                np.random.randint(low=0, high=5, size=(self.num_blobs - 1)),
            ],
            axis=0,
        )
        action_vecs = self.actions[action_idxs]
        self.velocities = (1 - self.resistance) * self.velocities + action_vecs
        self.positions += self.delta_t * self.velocities

        pos_mask = np.abs(self.positions) > 1 - self.radii
        vel_mask = np.sign(self.velocities) == np.sign(self.positions)
        self.velocities[pos_mask & vel_mask] *= -1

        contact_mask = np.linalg.norm(self.positions - self.positions[0:1], axis=-1) < 2 * self.radii
        contact_mask[0] = False
        num_contacts = contact_mask.sum()
        self.positions[contact_mask] = np.random.uniform(low=-1.0, high=1.0, size=(num_contacts, 2))
        self.velocities[contact_mask] = np.zeros((num_contacts, 2))

        observation = self._create_observation()
        reward = self.type_rewards[self.types[contact_mask]].sum()
        done = False

        return observation, reward, done, {}

    def get_observation_space(self) -> list[tuple[list[int], DTypeLike]]:
        return [
            ([2], np.float32),
            ([1], np.int32),
            ([self.num_blobs, 5], np.float32),
        ]

    def _create_observation(self) -> list[np.ndarray]:
        position = self.positions[0]
        relative_positions = self.positions[1:] - self.positions[0:1]
        types = np.zeros((self.num_blobs - 1, 3))
        types[range(self.num_blobs - 1), self.types[1:]] = 1
        objects = np.concatenate([relative_positions, types], axis=1)
        num_objects = np.array([objects.shape[0]], dtype=np.int32)
        return [position, num_objects, objects]

    def visualize(self, *args, **kwargs) -> np.ndarray:
        return self._visualize_impl(*args, **kwargs)

    def _visualize_impl(self, resolution: int, attention_weights: Optional[np.ndarray] = None) -> np.ndarray:
        img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        positions = ((resolution * self.positions) / 2 + resolution / 2).astype(int).tolist()
        radius = int((resolution * self.radii) / 2)
        if attention_weights is not None:
            attention_weights_list = (256 * attention_weights).clip(max=255).astype(np.uint8).tolist()
            for i in range(1, self.num_blobs):
                attention_weight = attention_weights_list[i - 1]
                cv2.circle(
                    img=img,
                    center=positions[i],
                    radius=radius + 5,
                    color=(attention_weight, attention_weight, attention_weight),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
        for i in range(self.num_blobs):
            cv2.circle(
                img=img,
                center=positions[i],
                radius=radius,
                color=self.colors[self.types[i]],
                thickness=cv2.FILLED,
                lineType=cv2.LINE_AA,
            )
        return img
