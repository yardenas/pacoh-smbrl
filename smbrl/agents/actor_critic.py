from functools import partial
from typing import Any, NamedTuple, Optional

import distrax as trx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from optax import OptState, l2_loss

from smbrl import types
from smbrl.utils import Learner, inv_softplus


class ContinuousActor(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        *,
        key: jax.random.KeyArray,
    ):
        self.net = eqx.nn.MLP(state_dim, action_dim * 2, hidden_size, n_layers, key=key)

    def __call__(self, state: jax.Array) -> trx.Normal:
        x = self.net(state)
        mu, stddev = jnp.split(x, 2, axis=-1)
        init_std = inv_softplus(5.0)
        stddev = jnn.softplus(stddev + init_std) + 0.1
        dist = trx.Normal(mu, stddev)
        dist = trx.Transformed(dist, trx.Tanh())
        return dist

    def act(
        self,
        state: Any,
        key: Optional[jax.random.KeyArray] = None,
        deterministic: bool = False,
    ) -> jax.Array:
        if deterministic:
            return self(state).mean()
        else:
            assert key is not None
            return self(state).sample(seed=key)


class Critic(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        hidden_size: int,
        *,
        key: jax.random.KeyArray,
    ):
        self.net = eqx.nn.MLP(state_dim, 1, hidden_size, n_layers, key=key)

    def __call__(self, state: Any) -> jax.Array:
        x = self.net(state)
        return x.squeeze(-1)


class ModelBasedActorCritic:
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
    ) -> None:
        actor_key, critic_key = jax.random.split(key, 2)
        self.actor = ContinuousActor(
            state_dim=state_dim,
            action_dim=action_dim,
            **actor_config,
            key=actor_key,
        )
        self.critic = Critic(state_dim=state_dim, **critic_config, key=critic_key)
        self.actor_learner = Learner(self.actor, actor_optimizer_config)
        self.critic_learner = Learner(self.critic, critic_optimizer_config)
        self.horizon = horizon
        self.discount = discount
        self.lambda_ = lambda_
        self.update_fn = vanilla_update_actor_critic

    def update(
        self,
        model: types.Model,
        initial_states: types.FloatArray,
        key: jax.random.KeyArray,
    ):
        actor_critic_fn = partial(self.update_fn, model.sample)
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
        )
        self.actor = results.new_actor
        self.critic = results.new_critic
        self.actor_learner.state = results.new_actor_learning_state
        self.critic_learner.state = results.new_critic_learning_state
        return results.actor_loss, results.critic_loss


def discounted_cumsum(x: jax.Array, discount: float) -> jax.Array:
    # Divide by discount to have the first discount value from 1: [1, discount,
    # discount^2 ...]
    scales = jnp.cumprod(jnp.ones_like(x) * discount) / discount
    # Flip scales since jnp.convolve flips it as default.
    return jnp.convolve(x, scales[::-1])[-x.shape[0] :]


def compute_lambda_values(
    next_values: jax.Array, rewards: jax.Array, discount: float, lambda_: float
) -> jax.Array:
    tds = rewards + (1.0 - lambda_) * discount * next_values
    tds = tds.at[-1].add(lambda_ * discount * next_values[-1])
    return discounted_cumsum(tds, lambda_ * discount)


class ActorCriticStepResults(NamedTuple):
    new_actor: ContinuousActor
    new_critic: Critic
    new_actor_learning_state: OptState
    new_critic_learning_state: OptState
    actor_loss: jax.Array
    critic_loss: jax.Array


@eqx.filter_jit
def update_actor_critic(
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
) -> ActorCriticStepResults:
    def actor_loss_fn(
        actor: ContinuousActor,
    ) -> tuple[jax.Array, tuple[types.Prediction, jax.Array]]:
        traj_key, policy_key = jax.random.split(key, 2)
        policy = lambda state: actor.act(state, key=policy_key)
        trajectories = rollout_fn(horizon, initial_states, traj_key, policy)
        # vmap over batch and time axes.
        bootstrap_values = jax.vmap(jax.vmap(critic))(trajectories.next_state)
        lambda_values = eqx.filter_vmap(compute_lambda_values)(
            bootstrap_values, trajectories.reward, discount, lambda_
        )
        return -lambda_values.mean(), (trajectories, lambda_values)

    rest, grads = eqx.filter_value_and_grad(actor_loss_fn, has_aux=True)(actor)
    actor_loss, (trajectories, lambda_values) = rest
    new_actor, new_actor_state = actor_learner.grad_step(
        actor, grads, actor_learning_state
    )

    def critic_loss_fn(critic: Critic) -> jax.Array:
        values = jax.vmap(jax.vmap(critic))(trajectories.next_state)
        return l2_loss(values[:, :-1], lambda_values[:, 1:]).mean()

    critic_loss, grads = eqx.filter_value_and_grad(critic_loss_fn)(critic)
    new_critic, new_critic_state = critic_learner.grad_step(
        critic, grads, critic_learning_state
    )
    return ActorCriticStepResults(
        new_actor,
        new_critic,
        new_actor_state,
        new_critic_state,
        actor_loss,
        critic_loss,
    )


@eqx.filter_jit
def vanilla_update_actor_critic(
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
    vmapped_rollout_fn = jax.vmap(rollout_fn, (None, 0, None, None))
    contextualized_rollout_fn = lambda h, i, k, p: vmapped_rollout_fn(h, i, k, p)
    return update_actor_critic(
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