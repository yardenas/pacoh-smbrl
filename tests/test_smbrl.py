# type: ignore
import jax
import numpy as np
import pytest
from hydra import compose, initialize

from smbrl import acting, tasks
from smbrl.trainer import Trainer
from smbrl.trajectory import TrajectoryData
from smbrl.utils import ensemble_predict, normalize


@pytest.mark.parametrize(
    "agent",
    ["smbrl", "mmbrl"],
    ids=["smbrl", "mmbrl"],
)
def test_training(agent):
    with initialize(version_base=None, config_path="../smbrl/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "log_dir=/dev/null",
                "training.time_limit=32",
                "training.episodes_per_task=1",
                "training.task_batch_size=5",
                "training.parallel_envs=5",
                "training.eval_every=1",
                "training.action_repeat=4",
                f"agent={agent}",
                "agent.model.hidden_size=32",
                "agent.update_steps=1",
                "agent.replay_buffer.sequence_length=16",
                "agent.replay_buffer.num_shots=1",
            ],
        )
    make_env, task_sampler = tasks.make(cfg)
    with Trainer(cfg, make_env, task_sampler) as trainer:
        trainer.train(epochs=1)


def smbrl_predictions(agent, horizon):
    # vmap twice, once for the task batch size and second for num shots.
    step = jax.vmap(jax.vmap(agent.model.step))
    sample = lambda obs, acs: agent.model.sample(
        horizon,
        obs[0],
        jax.random.PRNGKey(0),
        acs,
    )
    vmaped_sample = jax.vmap(jax.vmap(sample))
    return step, vmaped_sample


def asmbrl_predictions(agent, horizon):
    step = jax.vmap(lambda m, o, a: ensemble_predict(m.step)(o, a))
    partial_step = lambda o, a: step(agent.model, o, a)
    mean_step = lambda o, a: jax.tree_map(lambda x: x.mean(1), partial_step(o, a))
    sample = lambda m, o, a: ensemble_predict(m.sample, (None, 0, None, 0))(
        horizon,
        o,
        jax.random.PRNGKey(0),
        a,
    )
    vmaped_sample = jax.vmap(sample)
    partial_sample = lambda o, a: vmaped_sample(agent.model, o, a)
    mean_sample = lambda o, a: jax.tree_map(
        lambda x: x.mean(1), partial_sample(o[:, :, 0], a)
    )
    return mean_step, mean_sample


NUM_TASKS = 5

COMMON = [
    "training.episodes_per_task=1",
    f"training.task_batch_size={NUM_TASKS}",
    f"training.parallel_envs={NUM_TASKS}",
    "training.render_episodes=0",
    "training.scale_reward=0.1",
    "training.time_limit=100",
    "agent.replay_buffer.num_shots=1",
    "log_dir=/dev/null",
]

SMBRL_CFG = COMMON.copy()

ASMBRL_CFG = [
    "agent=asmbrl",
    "agent.update_steps=100",
    "agent.posterior.update_steps=100",
] + COMMON


def collect_trajectories(cfg_overrides, agent_class):
    cfg_overrides, agent_class
    with initialize(version_base=None, config_path="../smbrl/configs"):
        cfg = compose(config_name="config", overrides=cfg_overrides)
    make_env, task_sampler = tasks.make(cfg)
    with Trainer(cfg, make_env, task_sampler) as trainer:
        assert trainer.agent is not None and trainer.env is not None
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
    trainer.env.reset(options={"task": list(trainer.tasks(False))})
    summary, _ = acting.interact(agent, trainer.env, 1, 1, False, 0)
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
    context = 35
    sequence_length = 84
    trajectories = TrajectoryData(
        *map(lambda x: x[:, None, :sequence_length], trajectories)
    )
    return trajectories, context, sequence_length, agent
