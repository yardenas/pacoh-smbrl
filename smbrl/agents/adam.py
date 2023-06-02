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


class FeedForward(eqx.Module):
    norm: eqx.nn.LayerNorm
    hidden: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(self, attention_size: int, hidden_size: int, key: jax.random.KeyArray):
        key1, key2 = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(attention_size)
        self.hidden = eqx.nn.Linear(attention_size, hidden_size, key=key1)
        self.out = eqx.nn.Linear(hidden_size, attention_size, key=key2)

    def __call__(self, x: jax.Array) -> jax.Array:
        skip = x
        x = jnn.relu(self.hidden(x))
        x = skip + self.out(x)
        x = self.norm(x)
        return x


class SequenceFeatures(eqx.Module):
    norm: eqx.nn.LayerNorm
    mha: eqx.nn.MultiheadAttention
    ff: FeedForward

    def __init__(
        self,
        num_heads: int,
        attention_size: int,
        hidden_size: int,
        key: jax.random.KeyArray,
    ):
        key1, key2 = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(attention_size)
        self.mha = eqx.nn.MultiheadAttention(
            num_heads,
            attention_size,
            inference=True,
            key=key1,
        )
        self.ff = FeedForward(attention_size, hidden_size, key2)

    def __call__(self, x: jax.Array, mask: jax.Array = None) -> jax.Array:
        skip = x
        causal_mask = jnp.tril(
            jnp.ones((self.encoder.num_heads, x.shape[1], x.shape[1]))
        )
        if mask is not None:
            mask = mask[None, None]
        else:
            mask = jnp.ones_like(causal_mask)
        mask = causal_mask * mask
        x = skip + self.mha(x, x, x, mask=causal_mask)
        x = jax.vmap(self.norm)(x)
        x = jax.vmap(self.ff)(x)
        return x


class DomainContext(eqx.Module):
    encoder: eqx.nn.Linear
    sequence_features: eqx.nn.Sequential
    temporal_summary: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        input_size: int,
        attention_size: int,
        hidden_size: int,
        context_size: int,
        sequence_length: int,
        key: jax.random.KeyArray,
    ):
        (
            encoder_key,
            *sequence_keys,
            temporal_summary_key,
            decoder_key,
        ) = jax.random.split(key, 2 + num_layers)
        self.encoder = eqx.nn.Linear(input_size, attention_size, key=encoder_key)
        self.sequence_features = eqx.nn.Sequential(
            [
                SequenceFeatures(num_heads, attention_size, hidden_size, key)
                for key in sequence_keys
            ]
        )
        self.temporal_summary = eqx.nn.Linear(
            sequence_length - 1, 1, key=temporal_summary_key
        )
        self.decoder = eqx.nn.Linear(attention_size, context_size * 2, key=decoder_key)

    def __call__(self, features: Features, actions: jax.Array) -> ShiftScale:
        x = jax.vmap(self.task_features)(features, actions)
        x = x.mean((0, 1))
        mu, stddev = jnp.split(x, 2, -1)
        # TODO (yarden): maybe constrain the stddev like here:
        # https://arxiv.org/pdf/1901.05761.pdf to enforce the size of it.
        stddev = jnn.softplus(stddev) + 0.1
        return ShiftScale(mu, stddev)

    def task_features(self, features: Features, actions: jax.Array) -> jax.Array:
        x = jnp.concatenate([features.flatten(), actions], -1)
        x = jax.vmap(self.encoder)(x)
        x = self.sequence_features(x)
        x = jax.vmap(self.temporal_summary, 1, 1)(x)
        x = jnn.relu(x)
        x = jax.vmap(self.decoder)(x)
        return x


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
            1,
            8,
            state_dim + 1 + 1 + action_dim,
            64,
            64,
            context_size,
            sequence_length,
            key=context_key,
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
        obs_embeddings = jnn.elu(jax.vmap(self.encoder)(features.flatten()))

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
        obs_embeddings = jnn.elu(self.encoder(features.flatten()))
        state, *_ = self.cell.filter(state, obs_embeddings, action, key, context)
        return state

    def infer_context(self, features: Features, actions: jax.Array) -> ShiftScale:
        c_features = jax.tree_map(lambda x: x[:, :-1], features)
        c_actions = actions[:, 1:]
        return self.context(c_features, c_actions)

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
        context_kl_loss = (
            kl_divergence(context_posterior, context_prior, 0.5).mean() * 1e-5
        )
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
    context_posterior = model.infer_context(features, actions)
    context_prior = ShiftScale(
        jnp.zeros_like(context_posterior.loc), jnp.ones_like(context_posterior.scale)
    )
    infer_key, context_key = jax.random.split(key)
    context = dtx.Normal(*context_posterior).sample(seed=context_key)
    infer_fn = lambda features, actions: model(features, actions, context, infer_key)
    outs = eqx.filter_vmap(infer_fn)(features, actions)
    _, outs, priors, posteriors = outs
    return outs, context_prior, context_posterior, priors, posteriors


def kl_divergence(
    posterior: ShiftScale, prior: ShiftScale, free_nats: float = 0.0
) -> jax.Array:
    prior_dist = dtx.MultivariateNormalDiag(*prior)
    posterior_dist = dtx.MultivariateNormalDiag(*posterior)
    return jnp.maximum(posterior_dist.kl_divergence(prior_dist), free_nats)
