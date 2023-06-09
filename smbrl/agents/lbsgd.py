"""https://github.com/lasgroup/lbsgd-rl/blob/main/lbsgd_rl/"""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree
from optax._src import base

from smbrl.utils import Learner


class LBSGDState(NamedTuple):
    eta: jax.Array
    lr: jax.Array


def scale_by_lbsgd(
    eta: float, m_0: float, m_1: float, eta_rate: float, base_lr: float
) -> base.GradientTransformation:
    eta_rate += 1.0

    def init_fn(params):
        del params
        return LBSGDState(eta, base_lr)

    def update_fn(updates, state, params=None):
        del params
        loss_grads, constraints_grads, constraint = updates
        eta_t = state.eta
        lr = jnp.where(
            jnp.greater(constraint, 0.0),
            compute_lr(constraint, loss_grads, constraints_grads, m_0, m_1, eta_t),
            base_lr,
        )
        new_eta = jnp.where(jnp.greater(constraint, 0.0), eta_t / eta_rate, eta_t)
        updates = jax.tree_map(lambda x: x * lr, loss_grads)
        return updates, LBSGDState(new_eta, lr)

    return base.GradientTransformation(init_fn, update_fn)


def compute_lr(constraint, loss_grads, constraint_grads, m_0, m_1, eta):
    constraint_grads, _ = jax.flatten_util.ravel_pytree(constraint_grads)
    loss_grads, _ = jax.flatten_util.ravel_pytree(loss_grads)
    projection = constraint_grads.dot(loss_grads)
    lhs = (
        constraint
        / (
            2.0 * jnp.abs(projection) / jnp.linalg.norm(loss_grads)
            + jnp.sqrt(constraint * m_1 + 1e-8)
        )
        / jnp.linalg.norm(loss_grads)
    )
    m_2 = (
        m_0
        + 10.0 * eta * (m_1 / (constraint + 1e-8))
        + 8.0
        * eta
        * jnp.linalg.norm(projection) ** 2
        / ((jnp.linalg.norm(loss_grads) * constraint) ** 2)
    )
    rhs = 1.0 / (m_2 + 1e-8)
    return jnp.minimum(lhs, rhs)


class LBSGDLearner(Learner):
    def __init__(
        self,
        model: PyTree,
        optimizer_config: dict[str, Any],
        eta: float,
        m_0: float,
        m_1: float,
        eta_rate: float,
        base_lr: float,
    ):
        self.optimizer = optax.chain(
            scale_by_lbsgd(eta, m_0, m_1, eta_rate, base_lr),
            optax.scale_by_adam(eps=optimizer_config.get("eps", 1e-8)),
            optax.scale(-1.0),
        )
        self.state = self.optimizer.init(eqx.filter(model, eqx.is_array))
