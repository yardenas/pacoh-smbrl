from functools import partial
from typing import Any

import equinox as eqx
import jax
from optax import OptState

from smbrl.agents import contextual_actor_critic as cac
from smbrl.agents import maki
from smbrl.agents import safe_actor_critic as sac
from smbrl.utils import Learner


class SafeContextualModelBasedActorCritic(
    sac.SafeModelBasedActorCritic, cac.ContextualModelBasedActorCritic
):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_config: dict[str, Any],
        critic_config: dict[str, Any],
        actor_optimizer_config: dict[str, Any],
        critic_optimizer_config: dict[str, Any],
        safety_critic_optimizer_config,
        horizon: int,
        discount: float,
        safety_discount: float,
        lambda_: float,
        safety_budget: float,
        key: jax.random.KeyArray,
        penalizer: sac.Penalizer,
        belief: maki.ShiftScale,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_config=actor_config,
            critic_config=critic_config,
            actor_optimizer_config=actor_optimizer_config,
            critic_optimizer_config=critic_optimizer_config,
            safety_critic_optimizer_config=safety_critic_optimizer_config,
            horizon=horizon,
            discount=discount,
            safety_discount=safety_discount,
            lambda_=lambda_,
            safety_budget=safety_budget,
            penalizer=penalizer,
            key=key,
        )
        actor_key, critic_key = jax.random.split(key, 3)
        context_size = belief.shift.shape[-1]
        self.actor = cac.ContextualContinuousActor(
            state_dim=state_dim + context_size * 2,
            action_dim=action_dim,
            **actor_config,
            key=actor_key,
        )
        self.critic = cac.ContextualCritic(
            state_dim=state_dim + context_size, **critic_config, key=critic_key
        )
        self.safety_critic = cac.ContextualCritic(
            state_dim=state_dim, **critic_config, key=critic_key
        )
        self.safety_critic_learner = Learner(
            self.safety_critic, critic_optimizer_config
        )
        self.update_fn = safe_contextual_update_actor_critic

    def contextualize(self, belief: maki.ShiftScale):
        self.update_fn = partial(safe_contextual_update_actor_critic, context=belief)


@eqx.filter_jit
def safe_contextual_update_actor_critic(
    rollout_fn: cac.ContextualRolloutFn,
    horizon: int,
    initial_states: jax.Array,
    actor: cac.ContextualContinuousActor,
    critic: cac.ContextualCritic,
    safety_critic: cac.ContextualCritic,
    actor_learning_state: OptState,
    critic_learning_state: OptState,
    safety_critic_learning_state: OptState,
    actor_learner: Learner,
    critic_learner: Learner,
    safety_critic_learner: Learner,
    key: jax.random.KeyArray,
    discount: float,
    safety_discount: float,
    lambda_: float,
    safety_budget: float,
    eta: float,
    backup_lr: float,
    context: maki.ShiftScale,
):
    vmapped_rollout_fn = jax.vmap(rollout_fn, (None, 0, None, None, 0))
    contextualized_rollout_fn = lambda h, i, k, p: vmapped_rollout_fn(
        h, i, k, p, context
    )
    return sac.safe_update_actor_critic(
        contextualized_rollout_fn,
        horizon,
        initial_states,
        actor,
        critic,
        safety_critic,
        actor_learning_state,
        critic_learning_state,
        safety_critic_learning_state,
        actor_learner,
        critic_learner,
        safety_critic_learner,
        key,
        discount,
        safety_discount,
        lambda_,
        safety_budget,
        eta,
        backup_lr,
    )
