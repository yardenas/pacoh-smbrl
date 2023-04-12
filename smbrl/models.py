from typing import Callable, NamedTuple, Optional

import distrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import PyTree

from smbrl.utils import clip_stddev, inv_softplus


class Prediction(NamedTuple):
    next_state: jax.Array
    reward: jax.Array
    next_state_stddev: jax.Array
    reward_stddev: jax.Array


class Model(eqx.Module):
    layers: list[eqx.nn.Linear]
    encoder: eqx.nn.Linear
    state_decoder: eqx.nn.Linear
    reward_decoder: eqx.nn.Linear
    state_stddev_clip: tuple[float, float]
    reward_stddev_clip: tuple[float, float]

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        state_stddev_clip: tuple[float, float] = (
            float(inv_softplus(0.5)),
            float(inv_softplus(1.0)),
        ),
        reward_stddev_clip: tuple[float, float] = (
            float(inv_softplus(0.5)),
            float(inv_softplus(1.0)),
        ),
        *,
        key: jax.Array
    ):
        keys = jax.random.split(key, 3 + n_layers)
        self.layers = [
            eqx.nn.Linear(hidden_size, hidden_size, key=k) for k in keys[:n_layers]
        ]
        self.encoder = eqx.nn.Linear(state_dim + action_dim, hidden_size, key=keys[1])
        self.state_decoder = eqx.nn.Linear(hidden_size, state_dim * 2, key=keys[2])
        self.reward_decoder = eqx.nn.Linear(hidden_size, 2, key=keys[3])
        self.state_stddev_clip = state_stddev_clip
        self.reward_stddev_clip = reward_stddev_clip

    def __call__(
        self, state_sequence: jax.Array, action_sequence: jax.Array
    ) -> Prediction:
        x = jax.vmap(self.encoder)(
            jnp.concatenate([state_sequence, action_sequence], -1)
        )
        for layer in self.layers:
            x = jnn.relu(jax.vmap(layer)(x))
        next_state = jax.vmap(self.state_decoder)(x)
        next_state, next_state_stddev = jnp.split(next_state, 2, -1)
        reward = jax.vmap(self.reward_decoder)(x)
        reward, reward_stddev = jnp.split(reward, 2, -1)
        next_state_stddev = clip_stddev(next_state_stddev, *self.state_stddev_clip)
        reward_stddev = clip_stddev(reward_stddev, *self.reward_stddev_clip)
        return Prediction(next_state, reward, next_state_stddev, reward_stddev)

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.Array,
        action_sequence: Optional[jax.Array] = None,
        policy=None,
    ) -> Prediction:
        def f(carry, x):
            prev_state = carry
            action, key = x
            if action is None:
                assert policy is not None
                action = policy(prev_state).sample(key)
            out = self(
                prev_state[None],
                action[None],
            )
            return out.next_state[0], out

        if action_sequence is None:
            assert action_sequence is not None
            action_sequence = [None] * horizon
        else:
            assert len(action_sequence) == horizon
        init = initial_state
        inputs = (action_sequence, jax.random.split(key, horizon))
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        out = jax.tree_map(lambda x: x.squeeze(1), out)
        return out


class ParamsDistribution(eqx.Module):
    mus: PyTree
    stddev: PyTree

    def log_prob(self, other: PyTree) -> jax.Array:
        dist, *_ = self._to_dist()
        flat_params, _ = jax.flatten_util.ravel_pytree(other)
        return dist.log_prob(flat_params) / len(flat_params)

    def sample(self, seed: jax.Array) -> PyTree:
        dist, _, pytree_def = self._to_dist()
        samples = dist.sample(seed=seed)
        return pytree_def(samples)

    def _to_dist(
        self,
    ) -> tuple[distrax.Distribution, jax.Array, Callable[[jax.Array], PyTree]]:
        self_flat_mus, pytree_def = jax.flatten_util.ravel_pytree(self.mus)
        self_flat_stddevs, _ = jax.flatten_util.ravel_pytree(self.stddev)
        self_flat_stddevs = jnp.exp(self_flat_stddevs)
        dist = distrax.MultivariateNormalDiag(self_flat_mus, self_flat_stddevs)
        return dist, self_flat_mus, pytree_def
