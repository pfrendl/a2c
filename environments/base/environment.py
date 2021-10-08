from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
from numpy.typing import DTypeLike


E = TypeVar("E", bound="Environment")


class Environment(ABC):
    @abstractmethod
    def reset(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def step(self, action_idx: int) -> tuple[list[np.ndarray], float, bool, dict[str, Any]]:
        pass

    @abstractmethod
    def get_observation_space(self) -> list[tuple[list[int], DTypeLike]]:
        pass

    @abstractmethod
    def visualize(self, *args, **kwargs) -> np.ndarray:
        pass
