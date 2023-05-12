# mypy: disable-error-code="attr-defined"
from typing import Callable

import distrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import PyTree

from smbrl.agents.s4 import SequenceBlock
from smbrl.types import Moments, Prediction


class FeedForwardModel(eqx.Module):
    layers: list[eqx.nn.Linear]
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    stddev: jax.Array

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        *,
        key: jax.Array
    ):
        keys = jax.random.split(key, 2 + n_layers)
        self.layers = [
            eqx.nn.Linear(hidden_size, hidden_size, key=k) for k in keys[:n_layers]
        ]
        self.encoder = eqx.nn.Linear(state_dim + action_dim, hidden_size, key=keys[-2])
        self.decoder = eqx.nn.Linear(hidden_size, state_dim + 1, key=keys[-1])
        self.stddev = jnp.ones((state_dim + 1,))

    def __call__(self, x: jax.Array) -> Moments:
        x = jax.vmap(self.encoder)(x)
        for layer in self.layers:
            x = jnn.relu(jax.vmap(layer)(x))
        split = lambda x: jnp.split(x, [self.decoder.out_features - 1], -1)  # type: ignore # noqa E501
        pred = jax.vmap(self.decoder)(x)
        state, reward = split(pred)
        state_stddev, reward_stddev = split(self.stddev)
        state_stddev = jnp.exp(jnp.ones_like(state) * state_stddev)
        reward_stddev = jnp.exp(jnp.ones_like(reward) * reward_stddev)
        return Moments(
            to_outs(state, reward.squeeze(-1)),
            to_outs(state_stddev, reward_stddev.squeeze(-1)),
        )

    def step(self, state: jax.Array, action: jax.Array) -> Prediction:
        batched = True
        if state.ndim == 1:
            state, action = state[None], action[None]
            batched = False
        x = to_ins(state, action)
        mus, stddevs = self(x)
        assert stddevs is not None
        if not batched:
            mus, stddevs = mus.squeeze(0), stddevs.squeeze(0)
        split = lambda x: jnp.split(x, [self.decoder.out_features - 1], -1)  # type: ignore # noqa E501
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
        inputs = (action_sequence, jax.random.split(key, action_sequence.shape[0]))
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


SSM = tuple[jax.Array, jax.Array, jax.Array]


class S4Model(eqx.Module):
    layers: list[SequenceBlock]
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_layers: int,
        hippo_n: int,
        hidden_size: int,
        sequence_length: int,
        *,
        key: jax.random.KeyArray
    ):
        keys = jax.random.split(key, n_layers + 2)
        self.layers = [
            SequenceBlock(hidden_size, hippo_n, sequence_length, key=key)
            for key in keys[:n_layers]
        ]
        self.encoder = eqx.nn.Linear(state_dim + action_dim, hidden_size, key=keys[-2])
        self.decoder = eqx.nn.Linear(hidden_size, state_dim + 1, key=keys[-1])

    def __call__(
        self,
        x: jax.Array,
    ) -> Moments:
        x = jax.vmap(self.encoder)(x)
        for layer in self.layers:
            x = layer(x, convolve=True)[1]
        outs = jax.vmap(self.decoder)(x)
        return Moments(outs)

    def step(
        self,
        state: jax.Array,
        action: jax.Array,
        layers_ssm: list[SSM],
        layers_hidden: list[jax.Array],
    ) -> tuple[list[jax.Array], Prediction]:
        x = to_ins(state, action)
        x = self.encoder(x)
        x = x[None]
        out_hiddens = []
        for layer, ssm, hidden in zip(self.layers, layers_ssm, layers_hidden):
            hidden, x = layer(x, ssm=ssm, hidden=hidden)
            out_hiddens.append(hidden)
        outs = self.decoder(x[0])
        state, reward = jnp.split(
            outs, [self.decoder.out_features - 1], -1  # type: ignore
        )
        return out_hiddens, Prediction(state, reward.squeeze(-1))

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        action_sequence: jax.Array,
        layers_ssm: list[SSM],
        layers_hidden: list[jax.Array],
    ) -> Prediction:
        def f(carry, x):
            prev_hidden, prev_state = carry
            action, key = x
            out_hidden, out = self.step(
                prev_state,
                action,
                layers_ssm,
                prev_hidden,
            )
            return (out_hidden, out.next_state), out

        assert horizon == action_sequence.shape[0]
        assert all(x.ndim == 2 for x in layers_hidden)
        init = (layers_hidden, initial_state)
        inputs = (action_sequence, jax.random.split(key, horizon))
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        return out

    @property
    def init_state(self):
        return [layer.cell.init_state for layer in self.layers]

    @property
    def ssm(self):
        return [layer.cell.ssm for layer in self.layers]
