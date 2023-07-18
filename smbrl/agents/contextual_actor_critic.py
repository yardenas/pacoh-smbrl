from functools import partial
from typing import Any, Optional, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from optax import OptState

from smbrl import types
from smbrl.agents import actor_critic as ac
from smbrl.agents import maki
from smbrl.utils import Learner, contextualize


class ContextualContinuousActor(ac.ContinuousActor):
    def act(
        self,
        state: maki.BeliefAndState,
        key: Optional[jax.random.KeyArray] = None,
        deterministic: bool = False,
    ) -> jax.Array:
        flat_state = jnp.concatenate(
            [state.belief.shift, state.belief.scale, state.state], axis=-1
        )
        return super().act(flat_state, key, deterministic)


class ContextualCritic(ac.Critic):
    def __call__(self, state: maki.BeliefAndState) -> jax.Array:
        flat_state = contextualize(state.state, state.belief.shift)
        return super().__call__(flat_state)


class ContextualModelBasedActorCritic(ac.ModelBasedActorCritic):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_config: dict[str, Any],
        critic_config: dict[str, Any],
        actor_optimizer_config: dict[str, Any],
        critic_optimizer_config: dict[str, Any],
        horizon: int,
        discount: float,
        lambda_: float,
        key: jax.random.KeyArray,
        belief: maki.ShiftScale,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_config=actor_config,
            critic_config=critic_config,
            actor_optimizer_config=actor_optimizer_config,
            critic_optimizer_config=critic_optimizer_config,
            horizon=horizon,
            discount=discount,
            lambda_=lambda_,
            key=key,
        )
        actor_key, critic_key = jax.random.split(key, 2)
        context_size = belief.shift.shape[-1]
        self.actor = ContextualContinuousActor(
            state_dim=state_dim + context_size * 2,
            action_dim=action_dim,
            **actor_config,
            key=actor_key,
        )
        self.critic = ContextualCritic(
            state_dim=state_dim + context_size, **critic_config, key=critic_key
        )
        self.actor_learner = Learner(self.actor, actor_optimizer_config)
        self.critic_learner = Learner(self.critic, critic_optimizer_config)
        self.belief = belief
        self.update_fn = contextual_update_actor_critic

    def update(
        self,
        sample: types.RolloutFn,
        initial_states: types.FloatArray,
        key: jax.random.KeyArray,
    ):
        actor_critic_fn = partial(self.update_fn, sample)
        results = actor_critic_fn(
            self.horizon,
            initial_states,
            self.actor,
            self.critic,
            self.actor_learner.state,
            self.critic_learner.state,
            self.actor_learner,
            self.critic_learner,
            key,
            self.discount,
            self.lambda_,
            self.belief,
        )
        self.actor = results.new_actor
        self.critic = results.new_critic
        self.actor_learner.state = results.new_actor_learning_state
        self.critic_learner.state = results.new_critic_learning_state
        return {
            "agent/actor/loss": results.actor_loss.item(),
            "agent/critic/loss": results.critic_loss.item(),
        }

    def contextualize(self, belief: maki.ShiftScale):
        self.belief = belief


class ContextualRolloutFn(Protocol):
    def __call__(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        policy: types.Policy,
        context_belief: maki.ShiftScale,
    ) -> types.Prediction:
        ...


@eqx.filter_jit
def contextual_update_actor_critic(
    rollout_fn: ContextualRolloutFn,
    horizon: int,
    initial_states: jax.Array,
    actor: ContextualContinuousActor,
    critic: ContextualCritic,
    actor_learning_state: OptState,
    critic_learning_state: OptState,
    actor_learner: Learner,
    critic_learner: Learner,
    key: jax.random.KeyArray,
    discount: float,
    lambda_: float,
    context: maki.ShiftScale,
):
    vmapped_rollout_fn = jax.vmap(rollout_fn, (None, 0, None, None, 0))
    contextualized_rollout_fn = lambda h, i, k, p: vmapped_rollout_fn(
        h, i, k, p, context
    )
    return ac.update_actor_critic(
        contextualized_rollout_fn,
        horizon,
        initial_states,
        actor,
        critic,
        actor_learning_state,
        critic_learning_state,
        actor_learner,
        critic_learner,
        key,
        discount,
        lambda_,
    )
