from typing import NamedTuple, Optional

import distrax as dtx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from optax import OptState, l2_loss

from smbrl.agents.models import FeedForwardModel
from smbrl.types import Policy, Prediction
from smbrl.utils import Learner, contextualize


class Features(NamedTuple):
    observation: jax.Array
    reward: jax.Array
    cost: jax.Array
    terminal: jax.Array
    done: jax.Array

    def flatten(self):
        return jnp.concatenate(
            [self.observation, self.reward, self.cost, self.terminal, self.done],
            axis=-1,
        )


class ShiftScale(NamedTuple):
    shift: jax.Array
    scale: jax.Array


class BeliefAndState(NamedTuple):
    belief: ShiftScale
    state: jax.Array


class FeedForwardBlock(eqx.Module):
    norm: eqx.nn.LayerNorm
    hidden: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(
        self, hidden_size: int, intermediate_size: int, key: jax.random.KeyArray
    ):
        key1, key2 = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(hidden_size)
        self.hidden = eqx.nn.Linear(hidden_size, intermediate_size, key=key1)
        self.out = eqx.nn.Linear(intermediate_size, hidden_size, key=key2)

    def __call__(self, x: jax.Array) -> jax.Array:
        skip = x
        x = jnn.gelu(self.hidden(x))
        x = skip + self.out(x)
        x = self.norm(x)
        return x


class AttentionBlock(eqx.Module):
    norm: eqx.nn.LayerNorm
    mha: eqx.nn.MultiheadAttention

    def __init__(self, hidden_size: int, num_heads: int, key: jax.random.KeyArray):
        self.mha = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            inference=True,
            key=key,
        )
        self.norm = eqx.nn.LayerNorm(shape=hidden_size)

    def __call__(
        self, inputs: jax.Array, mask: Optional[jax.Array] = None
    ) -> jax.Array:
        if mask is not None:
            mask = self.make_self_attention_mask(mask)
        attention_output = self.mha(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            inference=True,
        )
        result = attention_output
        result = result + inputs
        result = jax.vmap(self.norm)(result)
        return result

    def make_self_attention_mask(self, mask: jax.Array) -> jax.Array:
        """Create self-attention mask from sequence-level mask."""
        mask = jnp.multiply(
            jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2)
        )
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)


class TransformerLayer(eqx.Module):
    attention: AttentionBlock
    ff: FeedForwardBlock

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        key: jax.random.KeyArray,
    ):
        key1, key2 = jax.random.split(key)
        self.attention = AttentionBlock(hidden_size, num_heads, key1)
        self.ff = FeedForwardBlock(hidden_size, intermediate_size, key2)

    def __call__(
        self,
        inputs: jax.Array,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        attention_output = self.attention(inputs, mask)
        output = jax.vmap(self.ff)(attention_output)
        return output


class DomainContext(eqx.Module):
    encoder: eqx.nn.Linear
    sequence_features: list[TransformerLayer]
    decoder: eqx.nn.Linear

    def __init__(
        self,
        num_layers: int,
        input_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        context_size: int,
        key: jax.random.KeyArray,
    ):
        (
            encoder_key,
            *sequence_keys,
            decoder_key,
        ) = jax.random.split(key, 2 + num_layers)
        self.encoder = eqx.nn.Linear(input_size, hidden_size, key=encoder_key)
        self.sequence_features = [
            TransformerLayer(hidden_size, intermediate_size, num_heads, key)
            for key in sequence_keys
        ]
        self.decoder = eqx.nn.Linear(hidden_size, context_size * 2, key=decoder_key)

    def __call__(self, features: Features, actions: jax.Array) -> ShiftScale:
        x = jnp.concatenate([features.flatten(), actions], -1)
        x = x.reshape(-1, *x.shape[2:])
        x = jax.vmap(self.encoder)(x)
        for layer in self.sequence_features:
            x = layer(x)
        # Global average pooling
        x = x.mean(0)
        x = self.decoder(x)
        mu, stddev = jnp.split(x, 2, -1)
        # TODO (yarden): maybe constrain the stddev like here:
        # https://arxiv.org/pdf/1901.05761.pdf to enforce the size of it.
        stddev = jnn.softplus(stddev) + 0.001
        return ShiftScale(mu, stddev)


class WorldModel(eqx.Module):
    context: DomainContext
    dynamics: FeedForwardModel

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_dynamics_layers: int,
        dynamics_size: int,
        num_context_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        context_size: int,
        *,
        key: jax.random.KeyArray,
    ):
        context_key, encoder_key = jax.random.split(key, 2)
        aux_dim = 4  # terminal, done, cost, reward
        self.context = DomainContext(
            num_context_layers,
            state_dim + aux_dim + action_dim,
            hidden_size,
            intermediate_size,
            num_heads,
            context_size,
            key=context_key,
        )
        self.dynamics = FeedForwardModel(
            num_dynamics_layers,
            state_dim,
            action_dim + context_size,
            dynamics_size,
            key=encoder_key,
        )

    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
        context: jax.Array,
    ) -> jax.Array:
        state = contextualize(state, context)
        x = jnp.concatenate([state, action], -1)
        outs = self.dynamics(x)
        return outs.mean

    def infer_context(self, features: Features, actions: jax.Array) -> ShiftScale:
        return self.context(features, actions)

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        policy: Policy,
        context_belief: ShiftScale,
    ) -> Prediction:
        def f(carry, x):
            prev_belief_state = carry
            if callable_policy:
                key = x
                action = policy(jax.lax.stop_gradient(prev_belief_state), key)
            else:
                action, key = x
            out = self.dynamics.step(
                contextualize(prev_belief_state.state, prev_belief_state.belief.shift),
                action,
            )
            next_belief_state = BeliefAndState(context_belief, out.next_state)
            out = Prediction(next_belief_state, out.reward)
            return next_belief_state, out

        init = BeliefAndState(context_belief, initial_state)
        callable_policy = False
        if isinstance(policy, jax.Array):
            inputs: tuple[jax.Array, jax.random.KeyArray] | jax.random.KeyArray = (
                policy,
                jax.random.split(key, policy.shape[0]),
            )
            assert policy.shape[0] <= horizon
        else:
            callable_policy = True
            inputs = jax.random.split(key, horizon)
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        return out  # type: ignore


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
            posterior=context_posterior,
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
    pred_fn = lambda features, actions: model(features.observation, actions, context)
    outs = eqx.filter_vmap(pred_fn)(features, actions)
    return outs, context_posterior, context_prior


def kl_divergence(
    posterior: ShiftScale, prior: ShiftScale, free_nats: float = 0.0
) -> jax.Array:
    prior_dist = dtx.MultivariateNormalDiag(*prior)
    posterior_dist = dtx.MultivariateNormalDiag(*posterior)
    return jnp.maximum(posterior_dist.kl_divergence(prior_dist), free_nats)
