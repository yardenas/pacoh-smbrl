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
    encoder: eqx.nn.Linear
    # layers: list[SequenceBlock]
    cell = eqx.nn.GRUCell
    decoder: eqx.nn.Linear

    def __init__(
        self,
        state_dim,
        action_dim,
        n_layers,
        hippo_n,
        hidden_size,
        sequence_length,
        context_size,
        *,
        key
    ):
        keys = jax.random.split(key, n_layers + 2)
        # self.layers = [
        #     SequenceBlock(hidden_size, hippo_n, sequence_length, key=key)
        #     for key in keys[:n_layers]
        # ]
        self.cell = eqx.nn.GRUCell(state_dim + action_dim + 3, hidden_size)
        self.encoder = eqx.nn.Linear(
            state_dim + action_dim + 3, hidden_size, key=keys[-2]
        )
        self.decoder = eqx.nn.Linear(hidden_size, context_size * 2, key=keys[-1])

    def __call__(self, features: Features, actions: jax.Array) -> ShiftScale:
        x = jnp.concatenate([features.flatten(), actions], -1)
        x = jax.vmap(self.encoder)(x)

        def fn(carry, inputs):
            out = self.cell(inputs, carry)
            return out, out

        x, _ = jax.lax.scan(fn, jnp.zeros_like(x), x)
        x = self.decoder(x[-1])
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
        self.context = DomainContext(
            state_dim,
            action_dim,
            n_layers,
            hippo_n,
            hidden_size,
            sequence_length,
            context_dim,
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
        # TODO (uarden): this should actually be outside of worldmodel
        flatten = lambda x: x.reshape(-1, x.shape[-1])
        features, actions = jax.tree_map(flatten, (features, actions))
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
