# type: ignore
import jax
import numpy as np
import pytest
from hydra import compose, initialize

from smbrl import acting, tasks
from smbrl.agents.asmbrl import ASMBRL
from smbrl.agents.fsmbrl import fSMBRL
from smbrl.agents.smbrl import SMBRL
from smbrl.trainer import Trainer
from smbrl.trajectory import TrajectoryData
from smbrl.utils import ensemble_predict, normalize


@pytest.mark.parametrize(
    "agent",
    ["asmbrl", "smbrl", "fsmbrl"],
    ids=["asmbrl", "smbrl", "fsmbrl"],
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
                "agent.model.n_layers=1",
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


def fsmbrl_predicitions(agent, horizon):
    ssm = agent.model.ssm

    def unroll_step(o, a):
        def f(carry, x):
            prev_hidden = carry
            observation, action = x
            hidden, out = agent.model.step(observation, action, ssm, prev_hidden)
            return hidden, out

        init_hidden = agent.model.init_state
        return jax.lax.scan(f, init_hidden, (o, a))

    step = jax.vmap(jax.vmap(lambda o, a: unroll_step(o, a)[1]))

    def sample(o, a):
        context = a.shape[0] - horizon
        hidden, out_context = unroll_step(o[:context], a[:context])
        out = agent.model.sample(
            horizon,
            o[context],
            jax.random.PRNGKey(0),
            a[context:],
            ssm,
            hidden,
        )
        out = jax.tree_map(
            lambda x, y: jax.numpy.concatenate([x, y], axis=0), out_context, out
        )
        return out

    vmaped_sample = jax.vmap(jax.vmap(sample))
    return step, vmaped_sample


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

FSMBRL_CFG = [
    "agent=fsmbrl",
    "agent.update_steps=100",
    "agent.model.n_layers=4",
    "agent.model.hidden_size=128",
    "agent.model.hippo_n=64",
    "agent.replay_buffer.sequence_length=84",
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


@pytest.mark.parametrize(
    "pred_fn_factory, overrides, agent_class",
    [
        (asmbrl_predictions, ASMBRL_CFG, ASMBRL),
        (smbrl_predictions, SMBRL_CFG, SMBRL),
        (fsmbrl_predicitions, FSMBRL_CFG, fSMBRL),
    ],
    ids=["asmbrl", "smbrl", "fsmbrl"],
)
def test_model_learning(pred_fn_factory, overrides, agent_class):
    trajectories, context, sequence_length, agent = collect_trajectories(
        overrides, agent_class
    )
    horizon = trajectories.observation.shape[2] - context
    step, sample = pred_fn_factory(agent, horizon)
    onestep_predictions = step(trajectories.observation, trajectories.action)
    multistep_predictions = sample(trajectories.observation, trajectories.action)
    evaluate(onestep_predictions, multistep_predictions, trajectories, context)
    plot(trajectories.next_observation, multistep_predictions.next_state, context)


def evaluate(onestep_predictions, multistep_predictions, trajectories, context):
    l2 = lambda x, y: ((x - y) ** 2).mean()
    onestep_reward_mse = l2(onestep_predictions.reward, trajectories.reward)
    onestep_obs_mse = l2(onestep_predictions.next_state, trajectories.next_observation)
    print(f"One step Reward MSE: {onestep_reward_mse}")
    print(f"One step Observation MSE: {onestep_obs_mse}")
    multistep_reward_mse = l2(multistep_predictions.reward, trajectories.reward)
    multistep_obs_mse = l2(
        multistep_predictions.next_state, trajectories.next_observation
    )
    print(f"Multistep step Reward MSE: {multistep_reward_mse}")
    print(f"Multistep Observation MSE: {multistep_obs_mse}")


def plot(y, y_hat, context):
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.arange(y.shape[2])

    plt.figure(figsize=(10, 5), dpi=600)
    for i in range(NUM_TASKS):
        plt.subplot(3, 4, i + 1)
        plt.plot(t, y[i, 0, :, 2], "b.", label="observed")
        plt.plot(
            t,
            y_hat[i, 0, :, 2],
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
    plt.show(block=False)
    plt.pause(100)
    plt.close()
