from itertools import zip_longest
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree


class Learner:
    def __init__(
        self,
        model: PyTree,
        optimizer_config: dict[str, Any],
    ):
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config.get("clip", float("inf"))),
            optax.scale_by_adam(eps=optimizer_config.get("eps", 1e-8)),
            optax.scale(-optimizer_config.get("lr", 1e-3)),
        )
        self.state = self.optimizer.init(model)

    def grad_step(
        self, model: PyTree, grads: PyTree, state: optax.OptState
    ) -> tuple[PyTree, optax.OptState]:
        updates, new_opt_state = self.optimizer.update(grads, state)
        all_ok = all_finite(grads)
        updates = update_if(
            all_ok, updates, jax.tree_map(lambda x: jnp.zeros_like(x), updates)
        )
        new_opt_state = update_if(all_ok, new_opt_state, state)
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state


def all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(True)
    else:
        leaves = map(jnp.isfinite, leaves)
        leaves = map(jnp.all, leaves)
        return jnp.stack(list(leaves)).all()


def update_if(pred, update, fallback):
    return jax.tree_map(lambda x, y: jax.lax.select(pred, x, y), update, fallback)


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def grouper(iterable, n, *, incomplete="fill", fillvalue=None):
    """Collect data into non-overlapping fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n
    if incomplete == "fill":
        return zip_longest(*args, fillvalue=fillvalue)
    if incomplete == "ignore":
        return zip(*args)
    else:
        raise ValueError("Expected fill or ignore")


def inv_softplus(x):
    return jnp.where(x < 20.0, jnp.log(jnp.exp(x) - 1.0), x)
