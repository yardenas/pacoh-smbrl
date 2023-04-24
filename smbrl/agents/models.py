# mypy: disable-error-code="attr-defined"
from typing import Callable

import distrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import PyTree

from smbrl.types import Prediction
from smbrl.utils import clip_stddev


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
            0.5,
            1.0,
        ),
        reward_stddev_clip: tuple[float, float] = (
            0.5,
            1.0,
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

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jax.vmap(self.encoder)(x)
        for layer in self.layers:
            x = jnn.relu(jax.vmap(layer)(x))
        next_state = jax.vmap(self.state_decoder)(x)
        next_state, next_state_stddev = jnp.split(next_state, 2, -1)
        reward = jax.vmap(self.reward_decoder)(x)
        reward, reward_stddev = jnp.split(reward, 2, -1)
        next_state_stddev = clip_stddev(next_state_stddev, *self.state_stddev_clip)
        reward_stddev = clip_stddev(reward_stddev, *self.reward_stddev_clip)
        return (
            to_outs(next_state, reward.squeeze(-1)),
            to_outs(next_state_stddev, reward_stddev.squeeze(-1)),
        )

    def step(self, state: jax.Array, action: jax.Array) -> Prediction:
        batched = True
        if state.ndim == 1:
            state, action = state[None], action[None]
            batched = False
        x = to_ins(state, action)
        mus, stddevs = self(x)
        if not batched:
            mus, stddevs = mus.squeeze(0), stddevs.squeeze(0)
        split = lambda x: jnp.split(x, [self.state_decoder.out_features // 2], axis=-1)  # type: ignore # noqa: E501
        state_mu, reward_mu = split(mus)
        state_stddev, reward_stddev = split(stddevs)
        reward_mu = reward_mu.squeeze(-1)
        reward_stddev = reward_stddev.squeeze(-1)
        return Prediction(state_mu, reward_mu, state_stddev, reward_stddev)

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        action_sequence: jax.Array,
    ) -> Prediction:
        def f(carry, x):
            prev_state = carry
            action, key = x
            out = self.step(
                prev_state,
                action,
            )
            return out.next_state, out

        init = initial_state
        inputs = (action_sequence, jax.random.split(key, horizon))
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        return out  # type: ignore


def to_ins(observation, action):
    return jnp.concatenate([observation, action], -1)


def to_outs(next_state, reward):
    return jnp.concatenate([next_state, reward[..., None]], -1)


class ParamsDistribution(eqx.Module):
    mus: PyTree
    stddev: PyTree

    def log_prob(self, other: PyTree) -> jax.Array:
        dist, *_ = self._to_dist()
        flat_params, _ = jax.flatten_util.ravel_pytree(other)
        logprobs: jax.Array = dist.log_prob(flat_params) / len(flat_params)
        return logprobs

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
