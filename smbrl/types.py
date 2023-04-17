from typing import Any, Callable, Iterable, Optional, Protocol, Union

import jax
import numpy as np
from numpy import typing as npt
from omegaconf import DictConfig
from optax import OptState

from smbrl import logging
from smbrl.models import Model
from smbrl.trajectory import TrajectoryData

Data = tuple[jax.Array, jax.Array]

ModelUpdate = tuple[tuple[Model, OptState], jax.Array]
FloatArray = npt.NDArray[Union[np.float32, np.float64]]


class Agent(Protocol):
    logger: logging.TrainingLogger
    config: DictConfig
    episodes: int

    def __call__(self, observation: FloatArray) -> FloatArray:
        ...

    def observe(self, trajectory: TrajectoryData) -> None:
        ...

    def adapt(self, trajectory: TrajectoryData) -> None:
        ...

    def reset(self) -> None:
        ...


TaskSamplerFactory = Callable[[int, Optional[bool]], Iterable[Any]]
