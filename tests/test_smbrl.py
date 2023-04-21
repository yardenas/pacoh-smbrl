# type: ignore
import importlib
from typing import Iterable, Optional

import jax
import numpy as np
import pytest
from hydra import compose, initialize

from smbrl import acting
from smbrl.trainer import Trainer
from smbrl.trajectory import TrajectoryData
from smbrl.utils import normalize, ensemble_predict


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
    sampler = lambda *args, **kwargs: task_sampler(cfg, *args, **kwargs)
    env = lambda: make_env(cfg)
    with Trainer(cfg, env, sampler) as trainer:
        trainer.train(epochs=1)


def make_env(cfg):
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    env._max_episode_steps = cfg.training.time_limit

    return env


def task_sampler(cfg, dummy: int, dummy2: Optional[bool] = False) -> Iterable[int]:
    for _ in range(cfg.training.task_batch_size):
        yield 1


def smbrl_predictions(agent, horizon):
    # FIXME (yarden): this doesn't support multi-task setting.
    step = jax.vmap(agent.model.step)
    sample = lambda obs, acs: agent.model.sample(
        horizon,
        obs,
        jax.random.PRNGKey(0),
        acs,
    )
    vmaped_sample = jax.vmap(sample)
    return step, vmaped_sample


def asmbrl_predictions(agent, horizon):
    step = jax.vmap(lambda m, o, a: ensemble_predict(m.step)(o, a))
    partial_step = lambda o, a: step(agent.model, o, a)
    sample = lambda m, o, a: ensemble_predict(m.sample, (None, 0, None, 0))(
        horizon,
        o,
        jax.random.PRNGKey(0),
        a,
    )
    vmaped_sample = jax.vmap(sample)
    partial_sample = lambda o, a: vmaped_sample(agent.model, o, a)
    return partial_step, partial_sample


@pytest.mark.parametrize(
    "agent, pred_fn_factory",
    [("asmbrl", asmbrl_predictions), ("smbrl", smbrl_predictions)],
    ids=["asmbrl", "smbrl"],
)
def test_model_learning(agent, pred_fn_factory):
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
    sampler = lambda *args, **kwargs: task_sampler(cfg, *args, **kwargs)
    env = lambda: make_env(cfg)
    with Trainer(cfg, env, sampler) as trainer:
        assert trainer.agent is not None and trainer.env is not None
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
        trainer.train(epochs=1)
    agent = trainer.agent
    assert agent is not None
    trainer.env.reset(options={"task": list(trainer.tasks(True))})
    summary = acting.interact(agent, trainer.env, 1, 1, True)
    trajectories = summary[0].as_numpy()
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
    trajectories = TrajectoryData(*map(lambda x: x[:, None], trajectories))
    context = 10
    horizon = trajectories.observation.shape[2] - context
    step, sample = pred_fn_factory(agent, horizon)
    onestep_predictions = step(trajectories.observation, trajectories.action)
    multistep_predictions = sample(
        trajectories.observation[:, :, context], trajectories.action[:, :, context:]
    )
    evaluate(onestep_predictions, multistep_predictions, trajectories, context)
    plot(trajectories.next_observation, multistep_predictions.next_state, context)


def evaluate(onestep_predictions, multistep_predictions, trajectories, context):
    l2 = lambda x, y: ((x - y) ** 2).mean()
    onestep_reward_mse = l2(onestep_predictions.reward, trajectories.reward)
    onestep_obs_mse = l2(onestep_predictions.next_state, trajectories.next_observation)
    print(f"One step Reward MSE: {onestep_reward_mse}")
    print(f"One step Observation MSE: {onestep_obs_mse}")
    multistep_reward_mse = l2(
        multistep_predictions.reward, trajectories.reward[:, context:]
    )
    multistep_obs_mse = l2(
        multistep_predictions.next_state, trajectories.next_observation[:, context:]
    )
    print(f"Multistep step Reward MSE: {multistep_reward_mse}")
    print(f"Multistep Observation MSE: {multistep_obs_mse}")


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
