from omegaconf import DictConfig
from realworldrl_suite.environments import ALL_TASKS as rlwr_tasks

from smbrl.tasks import pendulum
from smbrl.tasks.rlwr import env_factory
from smbrl.types import MetaEnvironmentFactory
from smbrl.utils import fix_task_sampling


def make(cfg: DictConfig) -> MetaEnvironmentFactory:
    assert len(cfg.environment.keys()) == 1
    env = list(cfg.environment.keys())[0]
    match env:
        case "pendulum":
            make_env, make_sampler = pendulum.make(cfg)
        case env if any(env in t_group for t_group in rlwr_tasks):
            make_env, make_sampler = env_factory.make(cfg)
        case _:
            raise NotImplementedError
    if cfg.training.num_tasks is not None:
        make_sampler = fix_task_sampling(make_sampler, cfg.training.num_tasks)
    return make_env, make_sampler


__all__ = ["make"]
