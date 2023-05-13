from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import distrax as dtx

from smbrl.types import Moments, Prediction


class State(NamedTuple):
    stochastic: jax.Array
    deterministic: jax.Array


class Features(NamedTuple):
    action: jax.Array
    reward: jax.Array
    cost: jax.Array


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
        self, prev_state: State, features: Features
    ) -> tuple[dtx.Normal, jax.Array]:
        x = to_ins(prev_state.stochastic, features)
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

    def __call__(self, prev_state: State, observation: jax.Array):
        x = jnp.concatenate([prev_state.deterministic, observation], -1)
        x = jnn.elu(self.encoder(x))
        x = self.decoder(x)
        mu, stddev = jnp.split(x, 2, -1)
        stddev = jnn.softplus(stddev) + 0.1
        return ShiftScale(mu, stddev)


class AdaM(eqx.Module):
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
        self, prev_state: State, features: Features, key: jax.random.KeyArray
    ) -> State:
        prior, deterministic = self.prior(prev_state, features)
        stochastic = dtx.Normal(*prior).sample(seed=key)
        return State(stochastic, deterministic)

    def filter(
        self,
        prev_state: State,
        features: Features,
        observation: jax.Array,
        key: jax.random.KeyArray,
    ) -> tuple[State, ShiftScale, ShiftScale]:
        prior, deterministic = self.prior(prev_state, features)
        state = State(prev_state.stochastic, deterministic)
        posterior = self.posterior(state, observation)
        stochastic = dtx.Normal(*posterior).sample(seed=key)
        return State(stochastic, deterministic), prior, posterior

    @property
    def init(self):
        return jnp.zeros(self.deterministic_size + self.stochastic_size)


class WorldModel(eqx.Module):
    cell: AdaM
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
        self.cell = AdaM(
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
        self,
        x: jax.Array,
    ) -> Moments:
        x = jax.vmap(self.encoder)(x)
        self.cell()
        outs = jax.vmap(self.decoder)(x)
        return Moments(outs)

    def step(
        self,
        state: jax.Array,
        action: jax.Array,
        hidden_state: State,
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
        hidden_state: State,
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
        assert all(x.ndim == 2 for x in layers_hidden)
        init = (layers_hidden, initial_state)
        inputs = (action_sequence, jax.random.split(key, horizon))
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        return out
