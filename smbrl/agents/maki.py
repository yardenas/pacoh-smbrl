from typing import NamedTuple

import distrax as dtx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from optax import OptState, l2_loss

from smbrl.agents.models import FeedForwardModel
from smbrl.types import Prediction
from smbrl.utils import Learner


class Features(NamedTuple):
    observation: jax.Array
    reward: jax.Array
    cost: jax.Array
    terminal: jax.Array

    def flatten(self):
        return jnp.concatenate(
            [self.observation, self.reward, self.cost, self.terminal], axis=-1
        )


class ShiftScale(NamedTuple):
    shift: jax.Array
    scale: jax.Array


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

    def __call__(self, features: Features, actions: jax.Array) -> ShiftScale:
        x = jnp.concatenate([features.flatten(), actions], -1)
        x = jax.vmap(jax.vmap(self.norm))(x)
        causal_mask = jnp.tril(
            jnp.ones((self.encoder.num_heads, x.shape[1], x.shape[1]))
        )
        encode = lambda x: self.encoder(x, x, x, mask=causal_mask)
        x = jnn.elu(jax.vmap(encode)(x))
        x = x.mean(0)
        x = jnn.elu(jax.vmap(self.temporal_summary, 1, 1)(x).squeeze(0))
        x = self.decoder(x)
        mu, stddev = jnp.split(x, 2, -1)
        stddev = jnn.softplus(stddev) + 0.1
        return ShiftScale(mu, stddev)


class WorldModel(eqx.Module):
    context: DomainContext
    dynamics: FeedForwardModel

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_layers: int,
        hippo_n: int,
        hidden_size: int,
        sequence_length: int,
        context_dim=32,
        *,
        key: jax.random.KeyArray
    ):
        context_key, encoder_key, decoder_key = jax.random.split(key, 3)
        context_size = 32
        self.context = DomainContext(
            1,
            state_dim + 1 + 1 + 1 + action_dim,
            64,
            context_size,
            sequence_length,
            key=context_key,
        )
        self.dynamics = FeedForwardModel(
            2, state_dim, action_dim + context_dim, 128, key=encoder_key
        )

    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
        context: jax.Array,
    ) -> jax.Array:
        context = jnp.repeat(context[None], state.shape[0], 0)
        x = jnp.concatenate([state, action, context], -1)
        outs = self.dynamics(x)
        return outs.mean

    def infer_context(self, features: Features, actions: jax.Array) -> ShiftScale:
        return self.context(features, actions)

    def sample(
        self,
        horizon: int,
        state: jax.Array,
        action_sequence: jax.Array,
        context: jax.Array,
        key: jax.random.KeyArray,
    ) -> Prediction:
        context = jnp.repeat(context[None], action_sequence.shape[0], 0)
        action_sequence = jnp.concatenate([action_sequence, context], -1)
        return self.dynamics.sample(horizon, state, key, action_sequence)


@eqx.filter_jit
def variational_step(
    features: Features,
    actions: jax.Array,
    next_states: jax.Array,
    model: WorldModel,
    learner: Learner,
    opt_state: OptState,
    key: jax.random.KeyArray,
    beta: float = 1.0,
    free_nats: float = 0.0,
):
    def loss_fn(model):
        infer_fn = eqx.filter_vmap(lambda f, a: infer(f, a, model, key))
        y_hat, context_posterior, context_prior = infer_fn(features, actions)
        y = jnp.concatenate([next_states, features.reward], -1)
        reconstruction_loss = l2_loss(y_hat, y).mean()
        kl_loss = kl_divergence(context_posterior, context_prior, free_nats).mean()
        extra = dict(
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
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
        jnp.zeros_like(context_posterior.shift), jnp.ones_like(context_posterior.scale)
    )
    context = dtx.Normal(*context_posterior).sample(seed=key)
    context = jnp.zeros_like(context)
    infer_fn = lambda features, actions: model(features.observation, actions, context)
    outs = eqx.filter_vmap(infer_fn)(features, actions)
    return outs, context_prior, context_posterior


def kl_divergence(
    posterior: ShiftScale, prior: ShiftScale, free_nats: float = 0.0
) -> jax.Array:
    prior_dist = dtx.MultivariateNormalDiag(*prior)
    posterior_dist = dtx.MultivariateNormalDiag(*posterior)
    return jnp.maximum(posterior_dist.kl_divergence(prior_dist), free_nats)
