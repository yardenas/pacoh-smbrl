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
CONSTRAINT_X = 5.5


def sharp_sigmoid(x, k=1.0):
    return 1.0 / (1.0 + jnp.exp(-k * x))


def rollout(
    horizon: int,
    initial_state: jax.Array,
    key: jax.random.KeyArray,
    policy: types.Policy,
) -> types.Prediction:
    def f(carry, x):
        prev_state = carry
        key = x
        action = policy(jax.lax.stop_gradient(prev_state), key)
        out = step(
            prev_state,
            action,
        )
        return out.next_state, out

    init = initial_state
    inputs = jax.random.split(key, horizon)
    _, out = jax.lax.scan(f, init, inputs, horizon)
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
        return rollout(horizon, initial_state, key, policy)


def step(state, action):
    pos, vel = state
    distance_to_constraint = pos - CONSTRAINT_X
    new_vel = vel + action[0] * DT
    new_pos = pos + new_vel * DT
    new_state = jnp.stack([new_pos, new_vel])
    reward = new_pos * DT
    cost = sharp_sigmoid(distance_to_constraint, 5.0)
    return types.Prediction(new_state, reward, cost)


def optimal_policy(state, safe=False):
    if safe:
        d = CONSTRAINT_X
    else:
        d = jnp.inf
    a = jnp.where(state[0] < (d / 2.0), 1.0, -1.0)
    return jnp.asarray([a])


def evaluate(policy, key):
    trajectories = jax.vmap(lambda s: rollout(TIME_HORIZON, s, key, policy))(
        jnp.zeros((BATCH_SIZE, STATE_DIM))
    )
    objective = trajectories.reward.mean(0).sum()
    constraint = trajectories.cost.mean(0).sum()
    return objective, constraint


def safe_actor_critic(safe):
    actor_config = {"n_layers": 2, "hidden_size": 32, "init_stddev": 1.5}
    critic_config = {"n_layers": 2, "hidden_size": 32}
    actor_optimizer_config = {"lr": 8e-5, "eps": 1e-5, "clip": 0.5}
    critic_optimizer_config = {"lr": 3e-4, "eps": 1e-5, "clip": 0.5}
    discount = 0.99
    safety_discount = 0.99
    lambda_ = 0.97
    # this corresponds to the sharpness parameter of sigmoid, so change them together
    safety_budget = 2.5
    eta = 0.5 if safe else 0.0
    m_0 = 5e3
    m_1 = 5e6
    eta_rate = 1e-3
    base_lr = 1e-3
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


@pytest.fixture
def actor_critic():
    actor_config = {"n_layers": 2, "hidden_size": 32, "init_stddev": 2.5}
    critic_config = {"n_layers": 2, "hidden_size": 32}
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


@pytest.mark.parametrize("safe", [(False), (True)])
def test_safe_model_based_actor_critic(safe):
    actor_critic = safe_actor_critic(safe)
    model = DummmyModel()
    key = jax.random.PRNGKey(0)
    for i in range(625):
        key, n_key = jax.random.split(key)
        outs = actor_critic.update(model, np.zeros((BATCH_SIZE, STATE_DIM)), n_key)
        if i % 10 == 0:
            for k, v in outs.items():
                print(k, v)
            key, n_key = jax.random.split(key)
            policy = lambda state, key: actor_critic.actor.act(
                state, key, deterministic=True
            )
            objective, constraint = evaluate(policy, n_key)
            print(f"------Objective: {objective}, constraint: {constraint}------")
    policy = lambda state, _: actor_critic.actor.act(state, deterministic=True)
    objective, constraint = evaluate(policy, key)
    solution_objective, solution_constraint = evaluate(
        lambda s, _: optimal_policy(s, safe), key
    )
    assert np.isclose(objective, solution_objective, 0.5, 0.5)
    if safe:
        assert (constraint <= solution_constraint + 0.1).all()


def test_model_based_actor_critic(actor_critic):
    model = DummmyModel()
    key = jax.random.PRNGKey(0)
    for i in range(625):
        key, n_key = jax.random.split(key)
        outs = actor_critic.update(model, np.zeros((BATCH_SIZE, STATE_DIM)), n_key)
        if i % 10 == 0:
            for k, v in outs.items():
                print(k, v)
            policy = lambda state, key: actor_critic.actor.act(
                state, key, deterministic=True
            )
            key, n_key = jax.random.split(key)
            objective, constraint = evaluate(policy, key)
            print(f"------Objective: {objective}, constraint: {constraint}------")
    policy = lambda state, _: actor_critic.actor.act(state, deterministic=True)
    objective, constraint = evaluate(policy, key)
    solution_objective, _ = evaluate(lambda s, _: optimal_policy(s, False), key)
    assert np.isclose(objective, solution_objective, 1e-1, 1e-1)
