from typing import NamedTuple, Optional

import distrax as dtx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from optax import OptState, l2_loss

from smbrl.agents.rssm import Features, ShiftScale, State, WorldModel, kl_divergence
from smbrl.types import Policy, Prediction
from smbrl.utils import Learner


class BeliefAndState(NamedTuple):
    belief: ShiftScale
    state: State


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


def repeat_context(context: jax.Array, num_repeat: int) -> jax.Array:
    return jnp.repeat(context[None], num_repeat, 0)


def contextualize_action(action, context):
    if action.ndim - context.ndim == 1:
        context = repeat_context(context, action.shape[0])
    return jnp.concatenate([action, context], -1)


class ContextualWorldModel(eqx.Module):
    context: DomainContext
    world_model: WorldModel
    context_encoder: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_context_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        context_size: int,
        deterministic_size: int,
        stochastic_size: int,
        *,
        key: jax.random.KeyArray,
    ):
        context_key, model_key, encoder_key = jax.random.split(key, 3)
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
        self.context_encoder = eqx.nn.Linear(
            context_size + action_dim, action_dim, key=encoder_key
        )
        self.world_model = WorldModel(
            state_dim,
            action_dim,
            deterministic_size,
            stochastic_size,
            hidden_size,
            key=model_key,
        )

    def __call__(
        self,
        features: Features,
        actions: jax.Array,
        context: jax.Array,
        key: jax.random.KeyArray,
    ) -> tuple[State, jax.Array, ShiftScale, ShiftScale]:
        actions = jax.vmap(self.context_encoder)(contextualize_action(actions, context))
        init_state = self.world_model.cell.init
        return self.world_model(features, actions, key, init_state)

    def infer_context(self, features: Features, actions: jax.Array) -> ShiftScale:
        return self.context(features, actions)

    def step(
        self,
        state: State,
        observation: jax.Array,
        action: jax.Array,
        context: jax.Array,
        key: jax.random.KeyArray,
    ) -> State:
        action = self.context_encoder(contextualize_action(action, context))
        return self.world_model.step(state, observation, action, key)

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        policy: Policy,
        context_belief: ShiftScale,
    ) -> Prediction:
        def f(carry, x):
            prev_state = carry
            if callable_policy:
                key = x
                key, p_key = jax.random.split(key)
                prev_belief_state = BeliefAndState(context_belief, prev_state)
                action = policy(jax.lax.stop_gradient(prev_belief_state), p_key)
            else:
                action, key = x
            action = self.context_encoder(
                contextualize_action(action, context_belief.shift)
            )
            next_state = self.world_model.cell.predict(prev_state, action, key)
            return next_state, next_state

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
        flat_init_state = State.from_flat(
            initial_state, self.world_model.cell.stochastic_size
        )
        _, state = jax.lax.scan(
            f,
            flat_init_state,
            inputs,
        )
        out = jax.vmap(self.world_model.decoder)(state.flatten())
        reward, cost = out[:, -2], out[:, -1]
        context_belief = jax.tree_map(
            lambda x: repeat_context(x, state.stochastic.shape[0]), context_belief
        )
        out = Prediction(BeliefAndState(context_belief, state), reward, cost)
        return out


class InferenceOutputs(NamedTuple):
    states: State
    flat_preds: jax.Array
    posterior: ShiftScale
    prior: ShiftScale
    context_posterior: ShiftScale
    context_prior: ShiftScale


@eqx.filter_jit
def variational_step(
    features: Features,
    actions: jax.Array,
    next_states: jax.Array,
    model: WorldModel,
    learner: Learner,
    opt_state: OptState,
    key: jax.random.KeyArray,
    beta_context: float = 1.0,
    beta_model: float = 1.0,
    free_nats_context: float = 0.0,
    free_nats_model: float = 0.0,
):
    def loss_fn(model):
        infer_fn = eqx.filter_vmap(lambda f, a: infer(f, a, model, key))
        outs = infer_fn(features, actions)
        y = jnp.concatenate([next_states, features.reward, features.cost], -1)
        reconstruction_loss = l2_loss(outs.flat_preds, y).mean()
        context_kl_loss = kl_divergence(
            outs.context_posterior, outs.context_prior, free_nats_context
        ).mean()
        transition_kl_loss = kl_divergence(
            outs.posterior, outs.prior, free_nats_model
        ).mean()
        extra = dict(
            reconstruction_loss=reconstruction_loss,
            context_kl_loss=context_kl_loss,
            transition_kl_loss=transition_kl_loss,
            states=outs.states,
            context_posterior=outs.context_posterior,
        )
        return (
            reconstruction_loss
            + beta_context * context_kl_loss
            + beta_model * transition_kl_loss,
            extra,
        )

    (loss, rest), model_grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    new_model, new_opt_state = learner.grad_step(model, model_grads, opt_state)
    return (new_model, new_opt_state), (loss, rest)


def infer(
    features: Features,
    actions: jax.Array,
    model: ContextualWorldModel,
    key: jax.random.KeyArray,
) -> InferenceOutputs:
    context_posterior = model.infer_context(features, actions)
    context_prior = ShiftScale(
        jnp.zeros_like(context_posterior.shift), jnp.ones_like(context_posterior.scale)
    )
    context_key, transition_key = jax.random.split(key)
    context = dtx.Normal(*context_posterior).sample(seed=context_key)
    pred_fn = lambda features, actions: model(
        features, actions, context, transition_key
    )
    state, flat_preds, posterior, prior = eqx.filter_vmap(pred_fn)(features, actions)
    return InferenceOutputs(
        state, flat_preds, posterior, prior, context_posterior, context_prior
    )
