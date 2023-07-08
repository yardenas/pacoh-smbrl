import pytest
from gymnasium.wrappers.time_limit import TimeLimit
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
    "param": "pole_length",
    "min": 0.75,
    "max": 1.25,
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


def test_min_max(cfg):
    cfg.environment.cartpole.perturb_spec = PERTURB
    _, task_sampler = tasks.make(cfg)
    params = [params for params in task_sampler(1000)]
    min_, max_ = min(params), max(params)
    print(f"min^: {min_}, min: {PERTURB['min']}. max^: {max_}, max: {PERTURB['max']}")
    assert min_ >= PERTURB["min"]
    assert max_ <= PERTURB["max"]


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
    assert any(
        p != o for p, o in zip(params_a, params_b)
    ), f"params_a: {params_a}, params_b: {params_b}"


def test_not_all_ones(cfg):
    n_tasks = 10
    cfg.training.num_tasks = n_tasks
    cfg.environment.cartpole.perturb_spec = PERTURB
    _, task_sampler = tasks.make(cfg)
    params_a = [p for p in task_sampler(n_tasks)]
    params_b = [1] * n_tasks
    assert any(
        p != o for p, o in zip(params_a, params_b)
    ), f"params_a: {params_a}, params_b: {params_b}"


def test_no_pertubations(cfg):
    cfg.environment.cartpole.perturb_spec = NO_PERTURB
    _, task_sampler = tasks.make(cfg)
    params_a = [p for p in task_sampler(5)]
    assert all(p is None for p in params_a)


def test_basic_ops(cfg):
    make_env, _ = tasks.make(cfg)
    env = TimeLimit(make_env(), cfg.training.time_limit)
    env.seed(1)
    _ = env.reset()
    count = 0
    while True:
        count += 1
        *_, terminal, truncated, _ = env.step(env.action_space.sample())
        if truncated:
            assert not terminal
            break
    _ = env.reset()
    assert count == cfg.training.time_limit


def test_external_perturb(cfg):
    cfg.environment.cartpole.perturb_spec = PERTURB
    make_env, task_sampler = tasks.make(cfg)
    env = make_env()
    param = next(task_sampler(1))
    env.reset(options={"task": param})
    assert env.env.env._task._perturb_cur == param
    env.env.env._task.update_physics()
    assert env.env.env._task._perturb_cur == param
