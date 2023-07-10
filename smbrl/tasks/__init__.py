from omegaconf import DictConfig

from smbrl.tasks.rlwr import TASKS as rwrl_tasks
from smbrl.types import MetaEnvironmentFactory
from smbrl.utils import fix_task_sampling


def make(cfg: DictConfig) -> MetaEnvironmentFactory:
    assert len(cfg.environment.keys()) == 1
    env = list(cfg.environment.keys())[0]
    match env:
        case "pendulum":
            from smbrl.tasks import pendulum

            make_env, make_sampler = pendulum.make(cfg)
        case _ if any(env in t_group for t_group in rwrl_tasks):
            from smbrl.tasks.rlwr import env_factory as rwrl_factory

            make_env, make_sampler = rwrl_factory.make(cfg)
        case _:
            raise NotImplementedError
    if cfg.training.num_tasks is not None:
        make_sampler = fix_task_sampling(make_sampler, cfg.training.num_tasks)
    return make_env, make_sampler


__all__ = ["make"]
