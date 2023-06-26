from typing import Any, Optional

import jax
import jax.numpy as jnp

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
        self.actor = ContextualContinuousActor(
            state_dim=state_dim,
            action_dim=action_dim,
            **actor_config,
            key=actor_key,
        )
        self.critic = ContextualCritic(
            state_dim=state_dim, **critic_config, key=critic_key
        )
        self.actor_learner = Learner(self.actor, actor_optimizer_config)
        self.critic_learner = Learner(self.critic, critic_optimizer_config)
        self.belief = belief

    def contextualize(self, belief: maki.ShiftScale):
        self.belief = belief
