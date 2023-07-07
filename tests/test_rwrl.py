import pytest
from hydra import compose, initialize

from smbrl import tasks

NO_PERTURB = {
    "enable": False,
    "period": 0,
    "scheduler": "constant",
}

PERTURB = {
    "enable": True,
    "period": 1,
    "scheduler": "uniform",
}


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
                "environment=cartpole_swingup",
            ],
        )
        return cfg


def test_same_params(cfg):
    n_tasks = 10
    cfg.training.num_tasks = n_tasks
    cfg.environment.cartpole.perturb_spec = PERTURB
    _, task_sampler = tasks.make(cfg)
    params_a = [p for p in task_sampler(n_tasks)]
    params_b = [p for p in task_sampler(n_tasks)]
    assert all(p == o for p, o in zip(params_a, params_b))


def test_different_params(cfg):
    n_tasks = 10
    cfg.environment.cartpole.perturb_spec = PERTURB
    _, task_sampler = tasks.make(cfg)
    params_a = [p for p in task_sampler(n_tasks)]
    params_b = [p for p in task_sampler(n_tasks)]
    assert any(p != o for p, o in zip(params_a, params_b))


def test_no_pertubations(cfg):
    cfg.environment.cartpole.perturb_spec = NO_PERTURB
    _, task_sampler = tasks.make(cfg)
    params_a = [p for p in task_sampler(5)]
    assert all(p is None for p in params_a)
