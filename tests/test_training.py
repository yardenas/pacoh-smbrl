from typing import Iterable, Optional

from hydra import compose, initialize

from smbrl import acting
from smbrl.trainer import Trainer
from smbrl.trajectory import TrajectoryData


def test_training():
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
                "smbrl.model.n_layers=1",
                "smbrl.model.hidden_size=32",
                "smbrl.model.hippo_n=8",
                "smbrl.update_steps=1",
                "smbrl.replay_buffer.sequence_length=16",
            ],
        )
    if not cfg.training.jit:
        from jax.config import config as jax_config  # pyright: ignore

        jax_config.update("jax_disable_jit", True)

    def make_env():
        import gymnasium as gym

        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env._max_episode_steps = cfg.training.time_limit  # type: ignore

        return env

    def task_sampler(dummy: int, dummy2: Optional[bool] = False) -> Iterable[int]:
        for _ in range(cfg.training.task_batch_size):
            yield 1

    with Trainer(cfg, make_env, task_sampler) as trainer:
        trainer.train(epochs=1)


def test_model_learning():
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
                "smbrl.model.n_layers=2",
                "smbrl.model.hidden_size=64",
                "smbrl.model.hippo_n=16",
                "smbrl.update_steps=100",
                "smbrl.replay_buffer.sequence_length=30",
            ],
        )
    if not cfg.training.jit:
        from jax.config import config as jax_config  # pyright: ignore

        jax_config.update("jax_disable_jit", True)

    def make_env():
        import gymnasium as gym

        env = gym.make("Pendulum-v1")
        env._max_episode_steps = cfg.training.time_limit  # type: ignore

        return env

    def task_sampler(dummy: int, dummy2: Optional[bool] = False) -> Iterable[int]:
        for _ in range(cfg.training.task_batch_size):
            yield 1

    with Trainer(cfg, make_env, task_sampler) as trainer:
        assert trainer.agent is not None and trainer.env is not None
        from smbrl.smbrl import smbrl, _normalize

        rs = np.random.RandomState(0)
        smbrl.__call__ = lambda self, observation: np.tile(
            rs.uniform(-1.0, 1.0, trainer.env.action_space.shape),  # type: ignore
            (
                cfg.training.task_batch_size,
                1,
            ),
        )
        trainer.train(epochs=3)
    agent = trainer.agent
    assert agent is not None
    trajectories = acting.interact(agent, trainer.env, 1, False)[0].as_numpy()
    normalize = lambda x: _normalize(
        x, agent.obs_normalizer.result.mean, agent.obs_normalizer.result.std
    )
    trajectories = TrajectoryData(
        normalize(trajectories.observation),
        normalize(trajectories.next_observation),
        trajectories.action,
        trajectories.reward,
        trajectories.cost,
    )
    hidden, onestep_predictions = jax.vmap(agent.model)(
        trajectories.observation, trajectories.action
    )
    assert hidden is not None
    context = 10
    horizon = trajectories.observation.shape[1] - context
    sample = lambda obs, hidden, acs: agent.model.sample(
        horizon,
        obs,
        hidden,
        action_sequence=acs,
        ssm=agent.model.ssm,
        key=jax.random.PRNGKey(0),
    )
    multistep_predictions = jax.vmap(sample)(
        trajectories.observation[:, context], hidden, trajectories.action[:, context:]
    )
    onestep_reward_mse = np.mean(
        (onestep_predictions.reward.squeeze(-1) - trajectories.reward) ** 2
    )
    onestep_obs_mse = np.mean(
        (onestep_predictions.next_state - trajectories.next_observation) ** 2
    )
    print(f"One step Reward MSE: {onestep_reward_mse}")
    print(f"One step Observation MSE: {onestep_obs_mse}")
    multistep_reward_mse = np.mean(
        (multistep_predictions.reward.squeeze(-1) - trajectories.reward[:, context:])
        ** 2
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
