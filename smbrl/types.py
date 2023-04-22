from typing import Any, Callable, Iterable, Optional, Protocol, Union

import jax
import numpy as np
import optax
from jaxtyping import PyTree
from numpy import typing as npt
from omegaconf import DictConfig

from smbrl import logging
from smbrl.agents.models import Model
from smbrl.trajectory import TrajectoryData

Data = tuple[jax.Array, jax.Array]

FloatArray = npt.NDArray[Union[np.float32, np.float64]]


class Agent(Protocol):
    logger: logging.TrainingLogger
    config: DictConfig
    episodes: int
    model: Model

    def __call__(self, observation: FloatArray) -> FloatArray:
        ...

    def observe(self, trajectory: TrajectoryData) -> None:
        ...

    def adapt(self, trajectory: TrajectoryData) -> None:
        ...

    def reset(self) -> None:
        ...


TaskSamplerFactory = Callable[[int, Optional[bool]], Iterable[Any]]

ModelUpdate = tuple[tuple[PyTree, optax.OptState], jax.Array]
