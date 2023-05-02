# mypy: disable-error-code="attr-defined"

from typing import Callable

import distrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import PyTree

from smbrl.agents.s4 import SequenceBlock
from smbrl.types import Prediction


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
        self.stddev = jnp.ones((state_dim + 1,)) * -10.0

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jax.vmap(self.encoder)(x)
        for layer in self.layers:
            x = jnn.relu(jax.vmap(layer)(x))
        split = lambda x: jnp.split(x, jnp.asarray([self.decoder.out_features - 1]), -1)
        pred = jax.vmap(self.decoder)(x)
        state, reward = split(pred)
        state_stddev, reward_stddev = split(self.stddev)
        state_stddev = jnp.exp(jnp.ones_like(state) * state_stddev) * 0.1
        reward_stddev = jnp.exp(jnp.ones_like(reward) * reward_stddev) * 0.1
        return (
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
        if not batched:
            mus, stddevs = mus.squeeze(0), stddevs.squeeze(0)
        split = lambda x: jnp.split(x, jnp.asarray([self.decoder.out_features - 1]), -1)
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


SSM = tuple[jax.Array, jax.Array, jax.Array]


class S4Model(eqx.Module):
    layers: list[SequenceBlock]
    encoder: eqx.nn.Linear
    state_decoder: eqx.nn.Linear
    reward_decoder: eqx.nn.Linear

    def __init__(
        self,
        n_layers,
        state_dim,
        action_dim,
        hippo_n,
        hidden_size,
        sequence_length,
        *,
        key
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
        layers_ssm: list[SSM],
        layers_hidden: list[jax.Array],
        convolve: bool = False,
    ) -> tuple[list[jax.Array], jax.Array]:
        x = jax.vmap(self.encoder)(x)
        hidden_states = []
        for layer_ssm, layer_hidden, layer in zip(
            layers_ssm, layers_hidden, self.layers
        ):
            layer_hidden, x = layer(
                x, convolve=convolve, ssm=layer_ssm, hidden=layer_hidden
            )
            hidden_states.append(layer_hidden)
        outs = jax.vmap(self.decoder)(x)
        if convolve:
            return [], outs
        else:
            return hidden_states, outs

    def step(
        self,
        state: jax.Array,
        action: jax.Array,
        layers_ssm: list[SSM],
        layers_hidden: list[jax.Array],
        convolve: bool = False,
    ) -> tuple[list[jax.Array], Prediction]:
        batched = True
        if state.ndim == 1:
            state, action = state[None], action[None]
            batched = False
        x = to_ins(state, action)
        hidden, outs = self(x, layers_ssm, layers_hidden, convolve=convolve)
        if not batched:
            squeeze = lambda x: jax.tree_map(lambda x: x.squeeze(0), x)
            outs = squeeze(outs)
            hidden = squeeze(hidden)
        state, reward = jnp.split(
            outs, jnp.asarray([self.decoder.out_features - 1]), -1
        )
        reward = reward.squeeze(-1)
        return hidden, Prediction(state, reward)

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        ssm: list[SSM],
        initial_hidden: list[jax.Array],
        key,
        action_sequence: jax.Array,
    ):
        def f(carry, x):
            prev_hidden, prev_state = carry
            action, key = x
            out_hidden, out = self.step(
                prev_state[None],
                action[None],
                convolve=False,
                ssm=ssm,
                hidden=prev_hidden,
            )
            return (out_hidden, out.next_state[0]), out

        assert all(x.ndim == 2 for x in initial_hidden)
        init = (initial_hidden, initial_state)
        inputs = (action_sequence, jax.random.split(key, horizon))
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        out = jax.tree_map(lambda x: x.squeeze(1), out)
        return out

    @property
    def init_state(self):
        return [layer.cell.init_state for layer in self.layers]

    @property
    def ssm(self):
        return [layer.cell.ssm for layer in self.layers]
