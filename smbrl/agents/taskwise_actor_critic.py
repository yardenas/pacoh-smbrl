from typing import Any

import equinox as eqx
import jax
from optax import OptState

from smbrl import types
from smbrl.agents.actor_critic import (
    ContinuousActor,
    Critic,
    ModelBasedActorCritic,
    update_actor_critic,
)
from smbrl.utils import Learner, ensemble_predict


class TaskwiseModelBasedActorCritic(ModelBasedActorCritic):
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
        task_batch_size: int = 1,
    ) -> None:
        actor_key, critic_key = jax.random.split(key, 2)
        if task_batch_size > 1:
            batched = True
            actor_factory = eqx.filter_vmap(
                lambda key: ContinuousActor(
                    state_dim=state_dim, action_dim=action_dim, **actor_config, key=key
                )
            )
            self.actor = actor_factory(jax.random.split(actor_key, task_batch_size))
            critic_factory = eqx.filter_vmap(
                lambda key: Critic(state_dim=state_dim, **critic_config, key=key)
            )
            self.critic = critic_factory(jax.random.split(critic_key, task_batch_size))
        else:
            batched = False
            self.actor = ContinuousActor(
                state_dim=state_dim,
                action_dim=action_dim,
                **actor_config,
                key=actor_key,
            )
            self.critic = Critic(state_dim=state_dim, **critic_config, key=critic_key)
        self.actor_learner = Learner(self.actor, actor_optimizer_config, batched)
        self.critic_learner = Learner(self.critic, critic_optimizer_config, batched)
        self.horizon = horizon
        self.discount = discount
        self.lambda_ = lambda_
        if task_batch_size > 1:
            self.update_fn = taskwise_update_actor_critic
        else:
            self.update_fn = update_actor_critic


def sample_per_task(sample_fn):
    # We get an ensemble of models which are specialized for a specific task.
    # 1. Use each member of the ensemble to make predictions.
    # 2. Average over these predictions to (approximately) marginalize
    #    over posterior parameters.
    sample_fn = jax.vmap(sample_fn, (None, 0, None, None))
    ensemble_sample = lambda sample_fn, h, o, k, pi: ensemble_predict(
        sample_fn, (None, 0, None, None)
    )(
        h,
        o[None],
        k,
        pi,
    )
    return lambda h, o, k, pi: jax.tree_map(
        lambda x: x.squeeze(1).mean(0), ensemble_sample(sample_fn, h, o, k, pi)
    )


@eqx.filter_jit
def taskwise_update_actor_critic(
    rollout_fn: types.RolloutFn,
    horizon: int,
    initial_states: jax.Array,
    actor: ContinuousActor,
    critic: Critic,
    actor_learning_state: OptState,
    critic_learning_state: OptState,
    actor_learner: Learner,
    critic_learner: Learner,
    key: jax.random.KeyArray,
    discount: float,
    lambda_: float,
):
    update_actor_critic_fn = eqx.filter_vmap(
        lambda i, a, c, m, s_a, c_a: update_actor_critic(
            sample_per_task(m),
            horizon,
            i,
            a,
            c,
            s_a,
            c_a,
            actor_learner,
            critic_learner,
            key,
            discount,
            lambda_,
        )
    )
    return update_actor_critic_fn(
        initial_states,
        actor,
        critic,
        rollout_fn,
        actor_learning_state,
        critic_learning_state,
    )
