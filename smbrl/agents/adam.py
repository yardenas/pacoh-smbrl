from typing import NamedTuple, Optional

import distrax as dtx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from optax import OptState, l2_loss

from smbrl.types import Prediction
from smbrl.utils import Learner


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


class ShiftScale(NamedTuple):
    loc: jax.Array
    scale: jax.Array


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
            stochastic_size + action_dim, deterministic_size, key=encoder_key
        )
        self.cell = eqx.nn.GRUCell(deterministic_size, deterministic_size, key=cell_key)
        self.decoder1 = eqx.nn.Linear(deterministic_size, hidden_size, key=decoder1_key)
        self.decoder2 = eqx.nn.Linear(
            hidden_size, stochastic_size * 2, key=decoder2_key
        )

    def __call__(
        self, prev_state: State, action: jax.Array
    ) -> tuple[dtx.Normal, jax.Array]:
        x = jnp.concatenate([prev_state.stochastic, action], -1)
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
        embedding_size: int,
        key: jax.random.KeyArray,
    ):
        encoder_key, decoder_key = jax.random.split(key)
        self.encoder = eqx.nn.Linear(
            deterministic_size + embedding_size, hidden_size, key=encoder_key
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
    deterministic_size: int
    stochastic_size: int

    def __init__(
        self,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        embedding_size: int,
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
            embedding_size,
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
    def init(self) -> State:
        return State(
            jnp.zeros(self.stochastic_size), jnp.zeros(self.deterministic_size)
        )


class WorldModel(eqx.Module):
    cell: RSSM
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        state_dim,
        action_dim,
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
            hidden_size,
            action_dim,
            cell_key,
        )
        self.encoder = eqx.nn.Linear(state_dim + 1 + 1, hidden_size, key=encoder_key)
        self.decoder = eqx.nn.Linear(
            hidden_size + stochastic_size, state_dim + 1 + 1, key=decoder_key
        )

    def __call__(
        self,
        features: Features,
        actions: jax.Array,
        key: jax.random.KeyArray,
        init_state: Optional[State] = None,
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
            init_state if init_state is not None else self.cell.init,
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
            # FIXME (yarden): don't hardcode this.
            out = Prediction(out[:-2], out[-2])
            return state, out

        assert horizon == action_sequence.shape[0]
        keys = jax.random.split(key, horizon)
        _, out = jax.lax.scan(f, state, (action_sequence, keys))
        return out


@eqx.filter_jit
def variational_step(
    features: Features,
    actions: jax.Array,
    model: WorldModel,
    learner: Learner,
    opt_state: OptState,
    key: jax.random.KeyArray,
    beta: float = 1.0,
):
    def loss_fn(model):
        infer = lambda features, actions: model(features, actions, key)
        _, outs, priors, posteriors = eqx.filter_vmap(infer)(features, actions)
        priors = dtx.MultivariateNormalDiag(*priors)
        posteriors = dtx.MultivariateNormalDiag(*posteriors)
        reconstruction_loss = l2_loss(outs, features.flatten()).mean()
        kl_loss = balanced_kl_loss(posteriors, priors, 1.0, 0.8).mean()
        return reconstruction_loss + beta * kl_loss, dict(
            reconstruction_loss=reconstruction_loss, kl_loss=kl_loss
        )

    (loss, rest), model_grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    new_model, new_opt_state = learner.grad_step(model, model_grads, opt_state)
    return (new_model, new_opt_state), (loss, rest)


# https://github.com/danijar/dreamerv2/blob/259e3faa0e01099533e29b0efafdf240adeda4b5/common/nets.py#L130
def balanced_kl_loss(
    posterior: dtx.Distribution, prior: dtx.Distribution, free_nats: float, mix: float
) -> jnp.ndarray:
    # sg = lambda x: jax.tree_map(jax.lax.stop_gradient, x)
    # lhs = posterior.kl_divergence(sg(prior)).astype(jnp.float32).mean()
    # rhs = sg(posterior).kl_divergence(prior).astype(jnp.float32).mean()
    # return (1.0 - mix) * jnp.maximum(lhs, free_nats) +
    # mix * jnp.maximum(rhs, free_nats)
    return jnp.maximum(posterior.kl_divergence(prior), free_nats)
