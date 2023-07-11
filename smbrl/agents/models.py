# mypy: disable-error-code="attr-defined"
from typing import Callable

import distrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import PyTree

from smbrl.types import Moments, Policy, Prediction


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
        key: jax.random.KeyArray,
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
        policy: Policy,
    ) -> Prediction:
        def f(carry, x):
            prev_state = carry
            if callable_policy:
                key = x
                action = policy(jax.lax.stop_gradient(prev_state), key)
            else:
                action, key = x
            out = self.step(
                prev_state,
                action,
            )
            return out.next_state, out

        init = initial_state
        callable_policy = False
        if isinstance(policy, jax.Array):
            inputs: tuple[jax.Array, jax.random.KeyArray] | jax.random.KeyArray = (
                policy,
                jax.random.split(key, policy.shape[0]),
            )
        else:
            callable_policy = True
            inputs = jax.random.split(key, horizon)
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


class RecurrentModel(eqx.Module):
    cell: eqx.nn.GRUCell
    encoder: eqx.nn.Linear
    decoder: eqx.nn.MLP

    def __init__(self, state_dim, action_dim, n_layers, hidden_size, *, key):
        cell_key, encoder_key, decoder_key = jax.random.split(key, 3)
        self.encoder = eqx.nn.Linear(
            state_dim + action_dim, hidden_size, key=encoder_key
        )
        self.cell = eqx.nn.GRUCell(hidden_size, hidden_size, key=cell_key)
        self.decoder = eqx.nn.MLP(
            hidden_size, state_dim + 1, hidden_size, n_layers, key=decoder_key
        )

    def __call__(
        self,
        x: jax.Array,
    ) -> Moments:
        def f(c, i):
            hidden = self.cell(i, c)
            return hidden, hidden

        x = jax.vmap(self.encoder)(x)
        hidden = self.init_state
        _, x = jax.lax.scan(f, hidden, x)
        outs = jax.vmap(self.decoder)(x)
        return Moments(outs)

    def step(
        self,
        state: jax.Array,
        action: jax.Array,
        hidden: jax.Array,
    ) -> tuple[jax.Array, Prediction]:
        x = to_ins(state, action)
        x = self.encoder(x)
        hidden = self.cell(x, hidden)
        x = self.decoder(hidden)
        state, reward = jnp.split(x, [self.decoder.out_size - 1], -1)  # type: ignore
        return hidden, Prediction(state, reward.squeeze(-1))

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        action_sequence: jax.Array,
        hidden: jax.Array,
    ) -> Prediction:
        def f(carry, x):
            prev_hidden, prev_state = carry
            action, key = x
            out_hidden, out = self.step(
                prev_state,
                action,
                prev_hidden,
            )
            return (out_hidden, out.next_state), out

        assert horizon == action_sequence.shape[0]
        init = (hidden, initial_state)
        inputs = (action_sequence, jax.random.split(key, horizon))
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        return out

    @property
    def init_state(self):
        return jnp.zeros(self.cell.hidden_size)
