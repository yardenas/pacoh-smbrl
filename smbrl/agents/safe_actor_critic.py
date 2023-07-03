from functools import partial
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from optax import OptState

from smbrl import types
from smbrl.agents import actor_critic as ac
from smbrl.agents.lbsgd import LBSGDLearner
from smbrl.utils import Learner, pytrees_unstack


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
        eta: float,
        m_0: float,
        m_1: float,
        eta_rate: float,
        base_lr: float,
        key: jax.random.KeyArray,
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
        # self.actor_learner = LBSGDLearner(
        #     self.actor,
        #     actor_optimizer_config,
        #     eta,
        #     m_0,
        #     m_1,
        #     eta_rate,
        #     base_lr,
        # )
        self.safety_discount = safety_discount
        self.safety_budget = safety_budget
        self.backup_lr = base_lr
        self.update_fn = safe_update_actor_critic

    def update(
        self,
        model: types.Model,
        initial_states: types.FloatArray,
        key: jax.random.KeyArray,
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
            # self.actor_learner.state[0].eta,
            0.1,
            self.backup_lr,
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
        }


class ActorLossOuts(NamedTuple):
    trajectories: types.Prediction
    lambda_values: jax.Array
    safety_lambda_values: jax.Array
    loss: jax.Array
    constraint: jax.Array
    safe: jax.Array


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


def actor_loss_fn(
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
    eta: float,
    backup_lr: float,
) -> tuple[jax.Array, ActorLossOuts]:
    loss, (trajectories, lambda_values) = ac.actor_loss_fn(
        actor, critic, rollout_fn, horizon, initial_states, key, discount, lambda_
    )
    bootstrap_safety_values = jax.vmap(jax.vmap(safety_critic))(trajectories.next_state)
    safety_lambda_values = eqx.filter_vmap(ac.compute_lambda_values)(
        bootstrap_safety_values, trajectories.cost, safety_discount, lambda_
    )
    constraint = safety_budget - safety_lambda_values.mean()
    loss = jnp.where(
        jnp.greater(constraint, 0.0),
        loss - eta * jnp.log(constraint + 1e-3),
        -backup_lr * constraint,
    )
    outs = jnp.stack([loss, constraint])
    outs = loss
    return outs, ActorLossOuts(
        trajectories,
        lambda_values,
        safety_lambda_values,
        loss,
        constraint,
        jnp.greater(constraint, 0.0),
    )


def jacrev(f, has_aux=False):
    def jacfn(x):
        y, vjp_fn, aux = eqx.filter_vjp(f, x, has_aux=has_aux)
        (J,) = eqx.filter_vmap(vjp_fn, in_axes=0)(jnp.eye(len(y)))
        return J, aux

    return jacfn


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
    eta: float,
    backup_lr: float,
):
    vmapped_rollout_fn = jax.vmap(rollout_fn, (None, 0, None, None))
    # Take gradients with respect to the loss function and the constraint in one go.
    # jacobian, rest = jacrev(
    #     lambda actor: actor_loss_fn(
    #         actor,
    #         critic,
    #         safety_critic,
    #         vmapped_rollout_fn,
    #         horizon,
    #         initial_states,
    #         key,
    #         discount,
    #         safety_discount,
    #         lambda_,
    #         safety_budget,
    #         eta,
    #         backup_lr,
    #     ),
    #     has_aux=True,
    # )(actor)
    grads, rest = eqx.filter_grad(
        lambda actor: actor_loss_fn(
            actor,
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
            eta,
            backup_lr,
        ),
        has_aux=True,
    )(actor)
    new_actor, new_actor_state = actor_learner.grad_step(
        actor, grads, actor_learning_state
    )
    # loss_grads, constraint_grads = pytrees_unstack(jacobian)
    # new_actor, new_actor_state = actor_learner.grad_step(
    # actor, (loss_grads, constraint_grads, rest.constraint), actor_learning_state
    # )
    critic_loss, grads = eqx.filter_value_and_grad(ac.critic_loss_fn)(
        critic, rest.trajectories, rest.lambda_values
    )
    new_critic, new_critic_state = critic_learner.grad_step(
        critic, grads, critic_learning_state
    )
    safety_critic_loss, grads = eqx.filter_value_and_grad(ac.critic_loss_fn)(
        safety_critic, rest.trajectories, rest.safety_lambda_values
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
        rest.loss,
        critic_loss,
        safety_critic_loss,
        rest.safe,
        rest.constraint,
    )
