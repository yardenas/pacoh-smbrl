from typing import NamedTuple

import distrax as dtx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from smbrl.types import Prediction


class State(NamedTuple):
    stochastic: jax.Array
    deterministic: jax.Array

    def flatten(self):
        return jnp.concatenate([self.stochastic, self.deterministic], axis=-1)


class Features(NamedTuple):
    observation: jax.Array
    reward: jax.Array
    cost: jax.Array

    def flatten(self):
        return jnp.concatenate([self.observation, self.reward, self.cost], axis=-1)

    @classmethod
    def from_flat(cls, flat):
        return cls(*jnp.split(flat, [-1, 1, 1], axis=-1))


class ShiftScale(NamedTuple):
    loc: jax.Array
    scale: jax.Array


def to_ins(stochastic: jax.Array, features: Features) -> jax.Array:
    return jnp.concatenate([stochastic, *features], axis=-1)


class Prior(eqx.Module):
    cell: eqx.nn.GRUCell
    encoder: eqx.nn.Linear
    decoder1: eqx.nn.Linear
    decoder2: eqx.nn.Linear

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        action_dim: int,
        key: jax.random.KeyArray,
    ):
        encoder_key, cell_key, decoder1_key, decoder2_key = jax.random.split(key, 4)
        self.encoder = eqx.nn.Linear(
            stochastic_size + action_dim + 2, deterministic_size, key=encoder_key
        )
        self.cell = eqx.nn.GRUCell(deterministic_size, deterministic_size, key=cell_key)
        self.decoder1 = eqx.nn.Linear(deterministic_size, hidden_size, key=decoder1_key)
        self.decoder2 = eqx.nn.Linear(
            hidden_size, stochastic_size * 2, key=decoder2_key
        )

    def __call__(
        self, prev_state: State, action: jax.Array
    ) -> tuple[dtx.Normal, jax.Array]:
        x = to_ins(prev_state.stochastic, action)
        x = jnn.elu(self.encoder(x))
        hidden = self.cell(x, prev_state.deterministic)
        x = jnn.elu(self.decoder1(hidden))
        x = self.decoder2(x)
        mu, stddev = jnp.split(x, 2, -1)
        stddev = jnn.softplus(stddev) + 0.1
        return ShiftScale(mu, stddev), hidden


class Posterior(eqx.Module):
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        observation_dim: int,
        key: jax.random.KeyArray,
    ):
        encoder_key, decoder_key = jax.random.split(key)
        self.encoder = eqx.nn.Linear(
            deterministic_size + observation_dim, hidden_size, key=encoder_key
        )
        self.decoder = eqx.nn.Linear(hidden_size, stochastic_size * 2, key=decoder_key)

    def __call__(self, prev_state: State, embedding: jax.Array):
        x = jnp.concatenate([prev_state.deterministic, embedding], -1)
        x = jnn.elu(self.encoder(x))
        x = self.decoder(x)
        mu, stddev = jnp.split(x, 2, -1)
        stddev = jnn.softplus(stddev) + 0.1
        return ShiftScale(mu, stddev)


class RSSM(eqx.Module):
    prior: Prior
    posterior: Posterior

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        observation_dim: int,
        action_dim: int,
        key: jax.random.KeyArray,
    ):
        prior_key, posterior_key = jax.random.split(key)
        self.prior = Prior(
            deterministic_size, stochastic_size, hidden_size, action_dim, prior_key
        )
        self.posterior = Posterior(
            deterministic_size,
            stochastic_size,
            hidden_size,
            observation_dim,
            posterior_key,
        )
        self.deterministic_size = deterministic_size
        self.stochastic_size = stochastic_size

    def predict(
        self, prev_state: State, action: jax.Array, key: jax.random.KeyArray
    ) -> State:
        prior, deterministic = self.prior(prev_state, action)
        stochastic = dtx.Normal(*prior).sample(seed=key)
        return State(stochastic, deterministic)

    def filter(
        self,
        prev_state: State,
        embeddings: jax.Array,
        action: jax.Array,
        key: jax.random.KeyArray,
    ) -> tuple[State, ShiftScale, ShiftScale]:
        prior, deterministic = self.prior(prev_state, action)
        state = State(prev_state.stochastic, deterministic)
        posterior = self.posterior(state, embeddings)
        stochastic = dtx.Normal(*posterior).sample(seed=key)
        return State(stochastic, deterministic), prior, posterior

    @property
    def init(self):
        return jnp.zeros(self.deterministic_size + self.stochastic_size)


class WorldModel(eqx.Module):
    cell: RSSM
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        state_dim,
        action_dim,
        n_layers,
        deterministic_size,
        stochastic_size,
        hidden_size,
        *,
        key
    ):
        cell_key, encoder_key, decoder_key = jax.random.split(key, 3)
        self.cell = RSSM(
            deterministic_size,
            stochastic_size,
            hidden_size,
            state_dim,
            action_dim,
            cell_key,
        )
        self.encoder = eqx.nn.Linear(
            state_dim + action_dim, hidden_size, key=encoder_key
        )
        self.decoder = eqx.nn.Linear(hidden_size, state_dim + 1, key=decoder_key)

    def __call__(
        self, features: Features, actions: jax.Array, key: jax.random.KeyArray
    ) -> tuple[State, jax.Array, ShiftScale, ShiftScale]:
        obs_embeddings = jax.vmap(self.encoder)(features.flatten())

        def fn(carry, inputs):
            prev_state = carry
            embedding, prev_action, key = inputs
            state, prior, posterior = self.cell.filter(
                prev_state, embedding, prev_action, key
            )
            return state, (state, prior, posterior)

        keys = jax.random.split(key, obs_embeddings.shape[0])
        last_state, (states, priors, posteriors) = jax.lax.scan(
            fn,
            self.cell.init,
            (obs_embeddings, actions, keys),
        )
        outs = jax.vmap(self.decoder)(states.flatten())
        return last_state, outs, priors, posteriors

    def step(
        self,
        state: State,
        features: Features,
        action: jax.Array,
        key: jax.random.KeyArray,
    ) -> State:
        obs_embeddings = self.encoder(features.flatten())
        state, *_ = self.cell.filter(state, obs_embeddings, action, key)
        return state

    def sample(
        self,
        horizon: int,
        state: State,
        action_sequence: jax.Array,
        key: jax.random.KeyArray,
    ) -> Prediction:
        def f(carry, inputs):
            prev_state = carry
            prev_action, key = inputs
            state = self.cell.predict(prev_state, prev_action, key)
            out = self.decoder(state.flatten())
            out = Prediction(out[:-1], out[-1])
            return state, out

        assert horizon == action_sequence.shape[0]
        keys = jax.random.split(key, horizon)
        _, out = jax.lax.scan(f, state, (action_sequence, keys))
        return out
