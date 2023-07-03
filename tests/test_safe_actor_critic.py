import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smbrl import types
from smbrl.agents.actor_critic import ModelBasedActorCritic
from smbrl.agents.safe_actor_critic import SafeModelBasedActorCritic

STATE_DIM = 2
ACTION_DIM = 1
DT = 0.1
TIME_HORIZON = 50
BATCH_SIZE = 1024
GOAL_X = 6.25
CONSTRAINT_X = 5.5


def sharp_sigmoid(x, k=1.0):
    return 1.0 / (1.0 + jnp.exp(-k * x))


def rollout(
    horizon: int,
    initial_state: jax.Array,
    policy: types.Policy,
) -> types.Prediction:
    def f(carry, x):
        prev_state = carry
        action = policy(jax.lax.stop_gradient(prev_state))
        out = step(
            prev_state,
            action,
        )
        return out.next_state, out

    init = initial_state
    _, out = jax.lax.scan(f, init, None, horizon)
    return out  # type: ignore


class DummmyModel(types.Model):
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return x, x

    def step(self, state: jax.Array, action: jax.Array) -> types.Prediction:
        return step(state, action)

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        policy: types.Policy,
    ) -> types.Prediction:
        return rollout(horizon, initial_state, policy)


def step(state, action):
    pos, vel = state
    delta = jnp.linalg.norm(pos - GOAL_X)
    distance_to_constraint = pos - CONSTRAINT_X
    new_vel = vel + action[0] * DT
    new_pos = pos + new_vel * DT
    new_delta = jnp.linalg.norm(new_pos - GOAL_X)
    new_state = jnp.stack([new_pos, new_vel])
    # Reach the goal as fast as possible.
    reward = delta - new_delta
    cost = sharp_sigmoid(distance_to_constraint, 10.0)
    return types.Prediction(new_state, reward, cost)


def optimal_policy(state, safe=False):
    if safe:
        d = CONSTRAINT_X
    else:
        d = GOAL_X
    a = jnp.where(state[0] < (d / 2.0), 1.0, -1.0)
    return jnp.asarray([a])


def evaluate(policy):
    trajectories = jax.vmap(lambda s: rollout(TIME_HORIZON, s, policy))(
        jnp.zeros((BATCH_SIZE, STATE_DIM))
    )
    objective = trajectories.reward.mean(0).sum()
    constraint = trajectories.cost.mean(0).sum()
    return objective, constraint


def safe_actor_critic():
    actor_config = {"n_layers": 2, "hidden_size": 32}
    critic_config = actor_config.copy()
    actor_optimizer_config = {"lr": 8e-5, "eps": 1e-5, "clip": 0.5}
    critic_optimizer_config = {"lr": 3e-4, "eps": 1e-5, "clip": 0.5}
    discount = 0.99
    safety_discount = 0.99
    lambda_ = 0.97
    safety_budget = 0.0
    eta = 0.1
    m_0 = 1e4
    m_1 = 1e4
    eta_rate = 8e-6
    base_lr = 3e-4
    key = jax.random.PRNGKey(0)
    return SafeModelBasedActorCritic(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        actor_config=actor_config,
        critic_config=critic_config,
        actor_optimizer_config=actor_optimizer_config,
        critic_optimizer_config=critic_optimizer_config,
        horizon=TIME_HORIZON,
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


def actor_critic():
    actor_config = {"n_layers": 2, "hidden_size": 32}
    critic_config = actor_config.copy()
    actor_optimizer_config = {"lr": 8e-5, "eps": 1e-5, "clip": 0.5}
    critic_optimizer_config = {"lr": 3e-4, "eps": 1e-5, "clip": 0.5}
    discount = 0.99
    lambda_ = 0.97
    key = jax.random.PRNGKey(0)
    return ModelBasedActorCritic(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        actor_config=actor_config,
        critic_config=critic_config,
        actor_optimizer_config=actor_optimizer_config,
        critic_optimizer_config=critic_optimizer_config,
        horizon=TIME_HORIZON,
        discount=discount,
        lambda_=lambda_,
        key=key,
    )


@pytest.mark.parametrize(
    "actor_critic,safe", [(actor_critic, False), (safe_actor_critic, True)]
)
def test_safe_model_based_actor_critic(actor_critic, safe):
    actor_critic = actor_critic()
    model = DummmyModel()
    key = jax.random.PRNGKey(0)
    for i in range(625):
        key, n_key = jax.random.split(key)
        outs = actor_critic.update(model, np.zeros((BATCH_SIZE, STATE_DIM)), n_key)
        if i % 10 == 0:
            for k, v in outs.items():
                print(k, v)
            policy = lambda state: actor_critic.actor.act(state, deterministic=True)
            objective, constraint = evaluate(policy)
            print(f"------Objective: {objective}, constraint: {constraint}------")
    policy = lambda state: actor_critic.actor.act(state, deterministic=True)
    objective, constraint = evaluate(policy)
    solution_objective, solution_constraint = evaluate(
        lambda s: optimal_policy(s, safe)
    )
    assert np.isclose(objective, solution_objective, 1e-1, 1e-1)
    if safe:
        assert np.isclose(constraint, solution_constraint, 1e-4, 1e-6)
