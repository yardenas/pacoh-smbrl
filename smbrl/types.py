from typing import Any, Callable, Iterable, Optional, Protocol, Union

import jax
import numpy as np
from numpy import typing as npt
from optax import OptState

from smbrl.models import Model
from smbrl.utils import Learner

Data = tuple[jax.Array, jax.Array]
PRNGKey = Any


class ModelUpdateFn(Protocol):
    def __call__(
        self,
        data: Data,
        model: Model,
        learner: Learner,
        opt_state: OptState,
        key: PRNGKey,
    ) -> tuple[tuple[Model, OptState], jax.Array]:
        ...


FloatArray = npt.NDArray[Union[np.float32, np.float64]]

TaskSamplerFactory = Callable[[int, Optional[bool]], Iterable[Any]]
