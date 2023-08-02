from functools import partial
from typing import Any

import equinox as eqx
import jax
from optax import OptState

from smbrl import types
from smbrl.agents import contextual_actor_critic as cac
from smbrl.agents import maki
from smbrl.agents import safe_actor_critic as sac
from smbrl.utils import Learner


class SafeContextualModelBasedActorCritic(
    sac.SafeModelBasedActorCritic, cac.ContextualContinuousActor
):
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
        safety_discount: float,
        lambda_: float,
        safety_budget: float,
        eta: float,
        m_0: float,
        m_1: float,
        eta_rate: float,
        base_lr: float,
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
            safety_discount=safety_discount,
            lambda_=lambda_,
            safety_budget=safety_budget,
            eta=eta,
            m_0=m_0,
            m_1=m_1,
            eta_rate=eta_rate,
            base_lr=base_lr,
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
        self.actor_learner = sac.LBSGDLearner(
            self.actor,
            actor_optimizer_config,
            eta,
            m_0,
            m_1,
            eta_rate,
            base_lr,
        )
        self.safety_discount = safety_discount
        self.safety_budget = safety_budget
        self.backup_lr = base_lr
        self.update_fn = safe_contextual_update_actor_critic

    def update(
        self,
        model: types.Model,
        initial_states: jax.Array,
        key: jax.random.KeyArray,
    ) -> dict[str, float]:
        actor_critic_fn = partial(self.update_fn, model.sample)
        results: sac.SafeActorCriticStepResults = actor_critic_fn(
            self.horizon,
            initial_states,
            self.actor,
            self.critic,
            self.safety_critic,
            self.actor_learner.state,
            self.critic_learner.state,
            self.safety_critic_learner.state,
            self.actor_learner,
            self.critic_learner,
            self.safety_critic_learner,
            key,
            self.discount,
            self.safety_discount,
            self.lambda_,
            self.safety_budget,
            self.actor_learner.state[0].eta,
            self.backup_lr,
            self.belief,
        )
        self.actor = results.new_actor
        self.critic = results.new_critic
        self.safety_critic = results.new_safety_critic
        self.actor_learner.state = results.new_actor_learning_state
        self.critic_learner.state = results.new_critic_learning_state
        self.safety_critic_learner.state = results.new_safety_critic_learning_state
        return {
            "agent/actor/loss": results.actor_loss.item(),
            "agent/critic/loss": results.critic_loss.item(),
            "agent/safety_critic/loss": results.safety_critic_loss.item(),
            "agent/safety_critic/safe": results.safe.item(),
            "agent/safety_critic/constraint": results.constraint.item(),
            "agent/lbsgd/lr": results.new_actor_learning_state[0].lr.item(),
            "agent/lbsgd/eta": results.new_actor_learning_state[0].eta.item(),
        }


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
