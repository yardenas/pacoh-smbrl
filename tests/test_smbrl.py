# type: ignore
from typing import Iterable, Optional

import pytest
from hydra import compose, initialize

from smbrl import acting
from smbrl.trainer import Trainer
from smbrl.trajectory import TrajectoryData
from smbrl.utils import normalize


@pytest.mark.parametrize("agent", ["asmbrl", "smbrl"], ids=["asmbrl", "smbrl"])
def test_training(agent):
    with initialize(version_base=None, config_path="../smbrl/"):
        cfg = compose(
            config_name="config",
            overrides=[
                "training.time_limit=32",
                "training.episodes_per_task=1",
                "training.task_batch_size=5",
                "training.parallel_envs=5",
                "training.eval_every=1",
                "training.action_repeat=4",
                f"agents={agent}",
                f"agents.{agent}.model.n_layers=1",
                f"agents.{agent}.model.hidden_size=32",
                f"agents.{agent}.update_steps=1",
                f"agents.{agent}.replay_buffer.sequence_length=16",
            ],
        )

    def make_env():
        import gymnasium as gym

        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env._max_episode_steps = cfg.training.time_limit

        return env

    def task_sampler(dummy: int, dummy2: Optional[bool] = False) -> Iterable[int]:
        for _ in range(cfg.training.task_batch_size):
            yield 1

    with Trainer(cfg, make_env, task_sampler) as trainer:
        trainer.train(epochs=1)


@pytest.mark.parametrize("agent", ["asmbrl", "smbrl"], ids=["asmbrl", "smbrl"])
def test_model_learning(agent):
    import jax
    import numpy as np

    with initialize(version_base=None, config_path="../smbrl/"):
        cfg = compose(
            config_name="config",
            overrides=[
                "training.time_limit=32",
                "training.episodes_per_task=1",
                "training.task_batch_size=5",
                "training.parallel_envs=5",
                "training.render_episodes=0",
                "training.scale_reward=0.1",
                f"agents={agent}",
                f"agents.{agent}.model.n_layers=2",
                f"agents.{agent}.model.hidden_size=64",
                f"agents.{agent}.update_steps=100",
                f"agents.{agent}.replay_buffer.sequence_length=30",
            ],
        )

    def make_env():
        import gymnasium as gym

        env = gym.make("Pendulum-v1")
        env._max_episode_steps = cfg.training.time_limit

        return env

    def task_sampler(dummy: int, dummy2: Optional[bool] = False) -> Iterable[int]:
        for _ in range(cfg.training.task_batch_size):
            yield 1

    with Trainer(cfg, make_env, task_sampler) as trainer:
        assert trainer.agent is not None and trainer.env is not None
        import importlib

        agent_module = importlib.import_module(f"smbrl.agents.{agent}")
        agent_class = getattr(agent_module, agent.upper())

        rs = np.random.RandomState(0)
        agent_class.__call__ = lambda self, observation: np.tile(
            rs.uniform(-1.0, 1.0, trainer.env.action_space.shape),
            (
                cfg.training.task_batch_size,
                1,
            ),
        )
        trainer.train(epochs=3)
    agent = trainer.agent
    assert agent is not None
    trajectories = acting.interact(agent, trainer.env, 1, 1, False)[0].as_numpy()
    normalize_fn = lambda x: normalize(
        x, agent.obs_normalizer.result.mean, agent.obs_normalizer.result.std
    )
    trajectories = TrajectoryData(
        normalize_fn(trajectories.observation),
        normalize_fn(trajectories.next_observation),
        trajectories.action,
        trajectories.reward,
        trajectories.cost,
    )
    onestep_predictions = jax.vmap(agent.model.step)(
        trajectories.observation, trajectories.action
    )
    context = 10
    horizon = trajectories.observation.shape[1] - context
    sample = lambda obs, acs: agent.model.sample(
        horizon,
        obs,
        action_sequence=acs,
        key=jax.random.PRNGKey(0),
    )
    multistep_predictions = jax.vmap(sample)(
        trajectories.observation[:, context], trajectories.action[:, context:]
    )
    onestep_reward_mse = np.mean(
        (onestep_predictions.reward - trajectories.reward) ** 2
    )
    onestep_obs_mse = np.mean(
        (onestep_predictions.next_state - trajectories.next_observation) ** 2
    )
    print(f"One step Reward MSE: {onestep_reward_mse}")
    print(f"One step Observation MSE: {onestep_obs_mse}")
    multistep_reward_mse = np.mean(
        (multistep_predictions.reward - trajectories.reward[:, context:]) ** 2
    )
    multistep_obs_mse = np.mean(
        (multistep_predictions.next_state - trajectories.next_observation[:, context:])
        ** 2
    )
    print(f"Multistep step Reward MSE: {multistep_reward_mse}")
    print(f"Multistep Observation MSE: {multistep_obs_mse}")
    plot(trajectories.next_observation, multistep_predictions.next_state, context)


def plot(y, y_hat, context):
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.arange(y.shape[1])

    plt.figure(figsize=(10, 5), dpi=600)
    for i in range(5):
        plt.subplot(3, 4, i + 1)
        plt.plot(t, y[i, :, 2], "b.", label="observed")
        plt.plot(
            t[context:],
            y_hat[i, :, 2],
            "r",
            label="prediction",
            linewidth=1.0,
        )
        ax = plt.gca()
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(context, color="k", linestyle="--", linewidth=1.0)
    plt.tight_layout()
    plt.show()
