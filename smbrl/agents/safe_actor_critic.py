from functools import partial
from typing import Any, Callable, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from optax import OptState

from smbrl import types
from smbrl.agents import actor_critic as ac
from smbrl.utils import Learner


class ActorEvaluation(NamedTuple):
    trajectories: types.Prediction
    loss: jax.Array
    lambda_values: jax.Array
    safety_lambda_values: jax.Array
    constraint: jax.Array
    safe: jax.Array


class Penalizer(Protocol):
    state: PyTree

    def __call__(
        self,
        evaluate: Callable[[ac.ContinuousActor], ActorEvaluation],
        state: Any,
        actor: ac.ContinuousActor,
    ) -> tuple[PyTree, Any, ActorEvaluation, dict[str, jax.Array]]:
        ...


class SafeModelBasedActorCritic(ac.ModelBasedActorCritic):
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
        key: jax.random.KeyArray,
        penalizer: Penalizer,
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
        *_, critic_key = jax.random.split(key, 3)
        self.safety_critic = ac.Critic(
            state_dim=state_dim, **critic_config, key=critic_key
        )
        self.safety_critic_learner = Learner(
            self.safety_critic, critic_optimizer_config
        )
        self.actor_learner = Learner(
            self.actor,
            actor_optimizer_config,
        )
        self.safety_discount = safety_discount
        self.safety_budget = safety_budget
        self.update_fn = safe_update_actor_critic
        self.penalizer = penalizer

    def update(
        self,
        model: types.Model,
        initial_states: jax.Array,
        key: jax.random.KeyArray,
        cost_normalizer: float,
    ) -> dict[str, float]:
        actor_critic_fn = partial(self.update_fn, model.sample)
        results: SafeActorCriticStepResults = actor_critic_fn(
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
            self.penalizer,
            self.penalizer.state,
            cost_normalizer,
        )
        self.actor = results.new_actor
        self.critic = results.new_critic
        self.safety_critic = results.new_safety_critic
        self.actor_learner.state = results.new_actor_learning_state
        self.critic_learner.state = results.new_critic_learning_state
        self.safety_critic_learner.state = results.new_safety_critic_learning_state
        self.penalizer.state = results.new_penalty_state
        return {
            "agent/actor/loss": results.actor_loss.item(),
            "agent/critic/loss": results.critic_loss.item(),
            "agent/safety_critic/loss": results.safety_critic_loss.item(),
            "agent/safety_critic/safe": float(results.safe.item()),
            "agent/safety_critic/constraint": results.constraint.item(),
            "agent/safety_critic/safety": results.safety.item(),
            **{k: v.item() for k, v in results.metrics.items()},
        }


class SafeActorCriticStepResults(NamedTuple):
    new_actor: ac.ContinuousActor
    new_critic: ac.Critic
    new_safety_critic: ac.Critic
    new_actor_learning_state: OptState
    new_critic_learning_state: OptState
    new_safety_critic_learning_state: OptState
    actor_loss: jax.Array
    critic_loss: jax.Array
    safety_critic_loss: jax.Array
    safe: jax.Array
    constraint: jax.Array
    safety: jax.Array
    new_penalty_state: Any
    metrics: dict[str, jax.Array]


def evaluate_actor(
    actor: ac.ContinuousActor,
    critic: ac.Critic,
    safety_critic: ac.Critic,
    rollout_fn: types.RolloutFn,
    horizon: int,
    initial_states: jax.Array,
    key: jax.random.KeyArray,
    discount: float,
    safety_discount: float,
    lambda_: float,
    safety_budget: float,
    cost_normalizer: float,
) -> ActorEvaluation:
    loss, (trajectories, lambda_values) = ac.actor_loss_fn(
        actor, critic, rollout_fn, horizon, initial_states, key, discount, lambda_
    )
    bootstrap_safety_values = jax.vmap(jax.vmap(safety_critic))(trajectories.next_state)
    safety_lambda_values = eqx.filter_vmap(ac.compute_lambda_values)(
        bootstrap_safety_values * cost_normalizer,
        trajectories.cost,
        safety_discount,
        lambda_,
    )
    constraint = safety_budget - safety_lambda_values.mean()
    return ActorEvaluation(
        trajectories,
        loss,
        lambda_values,
        safety_lambda_values,
        constraint,
        jnp.greater(constraint, 0.0),
    )


@eqx.filter_jit
def safe_update_actor_critic(
    rollout_fn: types.RolloutFn,
    horizon: int,
    initial_states: jax.Array,
    actor: ac.ContinuousActor,
    critic: ac.Critic,
    safety_critic: ac.Critic,
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
    penalty_fn: Penalizer,
    penalty_state: Any,
    cost_normalizer: float,
) -> SafeActorCriticStepResults:
    vmapped_rollout_fn = jax.vmap(rollout_fn, (None, 0, None, None))
    actor_grads, new_penalty_state, evaluation, metrics = penalty_fn(
        lambda a: evaluate_actor(
            a,
            critic,
            safety_critic,
            vmapped_rollout_fn,
            horizon,
            initial_states,
            key,
            discount,
            safety_discount,
            lambda_,
            safety_budget,
            cost_normalizer,
        ),
        penalty_state,
        actor,
    )
    new_actor, new_actor_state = actor_learner.grad_step(
        actor, actor_grads, actor_learning_state
    )
    critic_loss, grads = eqx.filter_value_and_grad(ac.critic_loss_fn)(
        critic, evaluation.trajectories, evaluation.lambda_values
    )
    new_critic, new_critic_state = critic_learner.grad_step(
        critic, grads, critic_learning_state
    )
    scaled_safety = evaluation.safety_lambda_values / cost_normalizer
    safety_critic_loss, grads = eqx.filter_value_and_grad(ac.critic_loss_fn)(
        safety_critic,
        evaluation.trajectories,
        scaled_safety,
    )
    new_safety_critic, new_safety_critic_state = safety_critic_learner.grad_step(
        safety_critic, grads, safety_critic_learning_state
    )
    return SafeActorCriticStepResults(
        new_actor,
        new_critic,
        new_safety_critic,
        new_actor_state,
        new_critic_state,
        new_safety_critic_state,
        evaluation.loss,
        critic_loss,
        safety_critic_loss,
        evaluation.safe,
        evaluation.constraint,
        scaled_safety.mean(),
        new_penalty_state,
        metrics,
    )
