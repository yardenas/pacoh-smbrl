from functools import partial
from typing import NamedTuple, Optional

import distrax as trx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from optax import OptState, l2_loss
from pyparsing import Any

from smbrl import types
from smbrl.utils import Learner, ensemble_predict, inv_softplus


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
        state: jax.Array,
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

    def __call__(self, state: jax.Array) -> jax.Array:
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
        self.task_batch_size = task_batch_size

    def update(
        self,
        rollout_fn: types.RolloutFn,
        initial_states: types.FloatArray,
        key: jax.random.KeyArray,
    ):
        if self.task_batch_size > 1:
            actor_critic_fn = taskwise_update_actor_critic
        else:
            actor_critic_fn = update_actor_critic
        actor_critic_fn = partial(actor_critic_fn, rollout_fn)
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
    rollout: types.RolloutFn,
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
        trajectories = rollout(horizon, initial_states, traj_key, policy)
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
        values = jax.vmap(jax.vmap(critic))(trajectories.next_state[:, :-1])
        return l2_loss(values, lambda_values[:, 1:]).mean()

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


def sample_per_task(sample_fn):
    # We get an ensemble of models which are specialized for a specific task.
    # 1. Use each member of the ensemble to make predictions.
    # 2. Average over these predictions to (approximately) marginalize
    #    over posterior parameters.
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
