import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smbrl import types
from smbrl.agents.safe_actor_critic import SafeModelBasedActorCritic

STATE_DIM = 2
ACTION_DIM = 1
DT = 0.01
TIME_HORIZON = 50


def rollout(
    horizon: int,
    initial_state: jax.Array,
    key: jax.random.KeyArray,
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
        return rollout(horizon, initial_state, key, policy)


def step(state, action):
    pos, vel = state
    pos = pos + vel * DT
    vel = action
    new_state = jnp.stack([pos, vel])
    delta = pos - 0.25
    reward = delta + (delta < 0.025).astype(jnp.float32)
    cost = (delta > 0.05).astype(jnp.float32)
    return types.Prediction(new_state, reward, cost)


def evaluate(policy):
    trajectory = rollout(
        TIME_HORIZON, jnp.zeros(STATE_DIM), jax.random.PRNGKey(0), policy
    )
    objective = trajectory.reward.sum()
    constraint = trajectory.cost.sum()
    return objective, constraint


@pytest.fixture
def safe_actor_critic():
    state_dim = 1
    action_dim = 1
    actor_config = {"n_layers": 1, "hidden_size": 32}
    critic_config = actor_config.copy()
    actor_optimizer_config = {"lr": 3e-4, "eps": 1e-5, "clip": 0.5}
    critic_optimizer_config = actor_config.copy()
    horizon = 15
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


def test_safe_model_based_actor_critic(safe_actor_critic: SafeModelBasedActorCritic):
    for _ in range(100):
        safe_actor_critic.update(
            DummmyModel(), np.zeros(STATE_DIM), jax.random.PRNGKey(0)
        )
    policy = lambda state: safe_actor_critic.actor.act(state, deterministic=True)
    objective, constraint = evaluate(policy)
    assert objective > 0.15
    assert np.isclose(constraint, 0.0, 1e-4, 1e-6)
