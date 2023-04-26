from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    NamedTuple,
    Optional,
    Protocol,
    Union,
)

import jax
import numpy as np
import optax
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from jaxtyping import PyTree
from numpy import typing as npt
from omegaconf import DictConfig

if TYPE_CHECKING:
    from smbrl.agents.models import Model
    from smbrl.logging import TrainingLogger
    from smbrl.trajectory import TrajectoryData

Data = tuple[jax.Array, jax.Array]

FloatArray = npt.NDArray[Union[np.float32, np.float64]]

TaskSampler = Callable[[int, Optional[bool]], Iterable[Any]]

EnvironmentFactory = Callable[[], Union[Env[Box, Box], Env[Box, Discrete]]]

MetaEnvironmentFactory = tuple[EnvironmentFactory, TaskSampler]

ModelUpdate = tuple[tuple[PyTree, optax.OptState], jax.Array]


class Agent(Protocol):
    logger: "TrainingLogger"
    config: DictConfig
    model: "Model"

    def __call__(self, observation: FloatArray) -> FloatArray:
        ...

    def observe(self, trajectory: "TrajectoryData") -> None:
        ...

    def adapt(self, trajectory: "TrajectoryData") -> None:
        ...

    def reset(self) -> None:
        ...


class Prediction(NamedTuple):
    next_state: jax.Array
    reward: jax.Array
    next_state_stddev: jax.Array
    reward_stddev: jax.Array


RolloutFn = Callable[[int, jax.Array, jax.random.KeyArray, jax.Array], Prediction]
