import pytest
from gymnasium.wrappers.time_limit import TimeLimit
from hydra import compose, initialize

from smbrl import tasks


@pytest.fixture
def cfg():
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
                "agent=smbrl",
                "agent.model.hidden_size=32",
                "agent.update_steps=1",
                "agent.replay_buffer.sequence_length=16",
                "agent.replay_buffer.num_shots=1",
                "environment=dm_cartpole",
            ],
        )
        return cfg


def test_basic_ops(cfg):
    make_env, _ = tasks.make(cfg)
    env = TimeLimit(make_env(), cfg.training.time_limit)
    env.seed(1)
    env.reset()
    count = 0
    while True:
        count += 1
        *_, terminal, truncated, _ = env.step(env.action_space.sample())
        if truncated:
            assert not terminal
            break
        assert count < cfg.training.time_limit
    _ = env.reset()
    assert count == cfg.training.time_limit
