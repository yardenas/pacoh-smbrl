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
        context_dim: int,
        key: jax.random.KeyArray,
    ):
        prior_key, posterior_key = jax.random.split(key)
        self.prior = Prior(
            deterministic_size,
            stochastic_size,
            hidden_size,
            action_dim + context_dim,
            prior_key,
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
        self,
        prev_state: State,
        action: jax.Array,
        key: jax.random.KeyArray,
        context: Optional[jax.Array] = None,
    ) -> State:
        if context is None:
            context = jnp.zeros_like(action[..., :-1])
        action = jnp.concatenate([action, context], -1)
        prior, deterministic = self.prior(prev_state, action)
        stochastic = dtx.Normal(*prior).sample(seed=key)
        return State(stochastic, deterministic)

    def filter(
        self,
        prev_state: State,
        embeddings: jax.Array,
        action: jax.Array,
        key: jax.random.KeyArray,
        context: Optional[jax.Array],
    ) -> tuple[State, ShiftScale, ShiftScale]:
        if context is None:
            context = jnp.zeros_like(action[..., :-1])
        action = jnp.concatenate([action, context], -1)
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


class DomainContext(eqx.Module):
    norm: eqx.nn.LayerNorm
    encoder: eqx.nn.MultiheadAttention
    temporal_summary: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        input_size: int,
        attention_size: int,
        context_size: int,
        sequence_length: int,
        key: jax.random.KeyArray,
    ):
        self.norm = eqx.nn.LayerNorm(input_size)
        encoder_key, temporal_summary_key, decoder_key = jax.random.split(key, 3)
        self.encoder = eqx.nn.MultiheadAttention(
            num_heads,
            input_size,
            output_size=attention_size,
            inference=True,
            key=encoder_key,
        )
        self.temporal_summary = eqx.nn.Linear(
            sequence_length, 1, key=temporal_summary_key
        )
        self.decoder = eqx.nn.Linear(attention_size, context_size * 2, key=decoder_key)

    def __call__(self, features: Features):
        x = jax.vmap(jax.vmap(self.norm))(features.flatten())
        causal_mask = jnp.tril(
            jnp.ones((self.encoder.num_heads, x.shape[1], x.shape[1]))
        )
        encode = lambda x: self.encoder(x, x, x, mask=causal_mask)
        x = jax.vmap(encode)(x)
        x = x.mean(0)
        x = jnn.elu(jax.vmap(self.temporal_summary, 1, 1)(x).squeeze(0))
        x = self.decoder(x)
        mu, stddev = jnp.split(x, 2, -1)
        stddev = jnn.softplus(stddev) + 0.1
        return ShiftScale(mu, stddev)


class WorldModelOuts(NamedTuple):
    context_prior: ShiftScale
    context_posterior: ShiftScale
    dynamics_prior: ShiftScale
    dynamics_posterior: ShiftScale
    last_state: State
    outs: jax.Array


class WorldModel(eqx.Module):
    cell: RSSM
    context: DomainContext
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        sequence_length: int,
        *,
        key
    ):
        cell_key, context_key, encoder_key, decoder_key = jax.random.split(key, 4)
        context_size = 32
        self.cell = RSSM(
            deterministic_size,
            stochastic_size,
            hidden_size,
            hidden_size,
            action_dim,
            context_size,
            cell_key,
        )
        self.context = DomainContext(
            1, state_dim + 1 + 1, 64, context_size, sequence_length, key=context_key
        )
        self.encoder = eqx.nn.Linear(state_dim + 1 + 1, hidden_size, key=encoder_key)
        self.decoder = eqx.nn.Linear(
            hidden_size + stochastic_size, state_dim + 1 + 1, key=decoder_key
        )

    def __call__(
        self,
        features: Features,
        actions: jax.Array,
        context: jax.Array,
        key: jax.random.KeyArray,
        init_state: Optional[State] = None,
    ) -> tuple[State, jax.Array, ShiftScale, ShiftScale]:
        obs_embeddings = jax.vmap(self.encoder)(features.flatten())

        def fn(carry, inputs):
            prev_state = carry
            embedding, prev_action, key = inputs
            state, prior, posterior = self.cell.filter(
                prev_state, embedding, prev_action, key, context
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
        context: jax.Array,
        key: jax.random.KeyArray,
    ) -> State:
        obs_embeddings = self.encoder(features.flatten())
        state, *_ = self.cell.filter(state, obs_embeddings, action, key, context)
        return state

    def infer_context(self, features: Features) -> ShiftScale:
        return self.context(features)

    def sample(
        self,
        horizon: int,
        state: State,
        action_sequence: jax.Array,
        context: jax.Array,
        key: jax.random.KeyArray,
    ) -> Prediction:
        def f(carry, inputs):
            prev_state = carry
            prev_action, key = inputs
            state = self.cell.predict(prev_state, prev_action, key, context)
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
        infer_fn = eqx.filter_vmap(lambda f, a: infer(f, a, model, key))
        outs, context_posterior, context_prior, posteriors, priors = infer_fn(
            features, actions
        )
        reconstruction_loss = l2_loss(outs, features.flatten()).mean()
        dynamics_kl_loss = kl_divergence(posteriors, priors, 0.5).mean()
        context_kl_loss = kl_divergence(context_posterior, context_prior, 0.5).mean()
        kl_loss = dynamics_kl_loss + context_kl_loss
        extra = dict(
            reconstruction_loss=reconstruction_loss,
            dynamics_kl_loss=dynamics_kl_loss,
            context_kl_loss=context_kl_loss,
        )
        return reconstruction_loss + beta * kl_loss, extra

    (loss, rest), model_grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    new_model, new_opt_state = learner.grad_step(model, model_grads, opt_state)
    return (new_model, new_opt_state), (loss, rest)


def infer(
    features: Features, actions: jax.Array, model: WorldModel, key: jax.random.KeyArray
):
    context_posterior = model.infer_context(features)
    prior_features = jax.tree_map(lambda x: x[:-1], features)
    context_prior = model.infer_context(prior_features)
    infer_key, context_key = jax.random.split(key)
    context = dtx.Normal(*context_posterior).sample(seed=context_key)
    context = jnp.zeros_like(context)
    infer_fn = lambda features, actions: model(features, actions, context, infer_key)
    outs = eqx.filter_vmap(infer_fn)(features, actions)
    _, outs, priors, posteriors = outs
    return outs, priors, posteriors, context_prior, context_posterior


def kl_divergence(
    posterior: ShiftScale, prior: ShiftScale, free_nats: float = 0.0
) -> jax.Array:
    prior_dist = dtx.MultivariateNormalDiag(*prior)
    posterior_dist = dtx.MultivariateNormalDiag(*posterior)
    return jnp.maximum(posterior_dist.kl_divergence(prior_dist), free_nats)
