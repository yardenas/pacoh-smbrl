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
    from smbrl.logging import TrainingLogger
    from smbrl.trajectory import TrajectoryData

Data = tuple[jax.Array, jax.Array]

FloatArray = npt.NDArray[Union[np.float32, np.float64]]

TaskSampler = Callable[[int, Optional[bool]], Iterable[Any]]

EnvironmentFactory = Callable[[], Union[Env[Box, Box], Env[Box, Discrete]]]

MetaEnvironmentFactory = tuple[EnvironmentFactory, TaskSampler]

ModelUpdate = tuple[tuple[PyTree, optax.OptState], jax.Array]


class Model(Protocol):
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        ...

    def step(self, state: jax.Array, action: jax.Array) -> "Prediction":
        ...

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        policy: "Policy",
    ) -> "Prediction":
        ...


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


class Actor(Protocol):
    def act(
        self,
        state: FloatArray,
        key: Optional[jax.random.KeyArray] = None,
        deterministic: bool = False,
    ) -> FloatArray:
        ...


class RolloutFn(Protocol):
    def __call__(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        policy: "Policy",
    ) -> "Prediction":
        ...


class Prediction(NamedTuple):
    next_state: jax.Array
    reward: jax.Array
    cost: Optional[jax.Array] = None
    next_state_stddev: Optional[jax.Array] = None
    reward_stddev: Optional[jax.Array] = None


class Moments(NamedTuple):
    mean: jax.Array
    stddev: Optional[jax.Array] = None


Policy = Union[Callable[[jax.Array], jax.Array], jax.Array]
