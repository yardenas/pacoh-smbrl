from typing import Any, Protocol

import jax
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
        pass
