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
        observation: maki.BeliefAndState,
        key: Optional[jax.random.KeyArray] = None,
        deterministic: bool = False,
    ) -> jax.Array:
        flat_state = jnp.concatenate(
            [
                observation.belief.shift,
                observation.belief.scale,
                observation.state.flatten(),
            ],
            axis=-1,
        )
        return super().act(flat_state, key, deterministic)


class ContextualCritic(ac.Critic):
    def __call__(self, observation: maki.BeliefAndState) -> jax.Array:
        # TODO (yarden): forward pass from multiple samples.
        # Rank by size and use something like wang's to make robust
        # This can be then used for TD learning of the value function.
        # Another alternative: https://arxiv.org/pdf/2102.05371.pdf
        # (which uses quantile critics) &
        # https://proceedings.neurips.cc/paper/2020/file/0b6ace9e8971cf36f1782aa982a708db-Paper.pdf
        # (which uses mixes values based on cvar and not wang's).
        # From this perspective, isn't it just as being robust to an
        # aleatoric uncertainty of the augmented MDP?
        # here is another approach for empirical estimation of risk measures:
        # https://www.tandfonline.com/doi/abs/10.1080/10920277.2003.10596117
        flat_state = contextualize(
            observation.state.flatten(), observation.belief.shift
        )
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
        self.update_fn = contextual_update_actor_critic

    def contextualize(self, belief: maki.ShiftScale):
        self.update_fn = partial(contextual_update_actor_critic, context=belief)


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
