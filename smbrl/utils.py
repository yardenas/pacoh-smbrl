from itertools import cycle, zip_longest
from typing import Any, Iterable, Optional

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import optax
from jaxtyping import PyTree

from smbrl.trajectory import TrajectoryData
from smbrl.types import TaskSampler


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
        leaves = list(map(jnp.isfinite, leaves))
        leaves = list(map(jnp.all, leaves))
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


def clip_stddev(stddev, stddev_min, stddev_max, stddev_scale=1.0):
    stddev = jnp.clip(
        (stddev + inv_softplus(0.1)) * stddev_scale,
        inv_softplus(stddev_min),
        inv_softplus(stddev_max),
    )
    return jnn.softplus(stddev)


class PRNGSequence:
    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def take_n(self, n):
        keys = jax.random.split(self.key, n + 1)
        self.key = keys[0]
        return keys[1:]


def add_to_buffer(buffer, trajectory, normalizer, reward_scale):
    results = normalizer.result
    normalize_fn = lambda x: normalize(x, results.mean, results.std)
    buffer.add(
        TrajectoryData(
            normalize_fn(trajectory.observation),
            normalize_fn(trajectory.next_observation),
            trajectory.action,
            trajectory.reward * reward_scale,
            trajectory.cost,
        )
    )


def normalize(
    observation,
    mean,
    std,
):
    diff = observation - mean
    return diff / (std + 1e-8)


def ensemble_predict(fn, in_axes=0):
    """
    A decorator that wraps (parameterized-)functions such that if they define
    an ensemble, predictions are made for each member of the ensemble individually.
    """

    def vmap_ensemble(*args, **kwargs):
        # First vmap along the batch dimension.
        ensemble_predict = lambda fn: jax.vmap(fn, in_axes=in_axes)(*args, **kwargs)
        # then vmap over members of the ensemble, such that each
        # individually computes outputs.
        ensemble_predict = eqx.filter_vmap(ensemble_predict)
        return ensemble_predict(fn)

    return vmap_ensemble


def fix_task_sampling(sampler: TaskSampler, num_tasks: int) -> TaskSampler:
    """
    Takes a task sampler and makes sure that the same tasks are being sampled.
    """
    train_tasks = cycle(list(sampler(num_tasks, True)))
    test_tasks = cycle(list(sampler(num_tasks, False)))

    def sample(batch_size: int, train: Optional[bool] = False) -> Iterable[Any]:
        train_tasks_it = iter(train_tasks)
        test_tasks_it = iter(test_tasks)
        for _ in range(batch_size):
            yield next(train_tasks_it) if train else next(test_tasks_it)

    return sample


class Count:
    def __init__(self, n: int):
        self.count = 0
        self.n = n

    def __call__(self):
        bingo = (self.count + 1) == self.n
        self.count = (self.count + 1) % self.n
        return bingo
