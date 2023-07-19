from omegaconf import DictConfig

from smbrl.tasks.dm_control import ENVIRONMENTS as dm_control_envs
from smbrl.tasks.rlwr import ENVIRONMENTS as rwrl_envs
from smbrl.types import MetaEnvironmentFactory
from smbrl.utils import fix_task_sampling


def make(cfg: DictConfig) -> MetaEnvironmentFactory:
    assert len(cfg.environment.keys()) == 1
    domain_name, task_cfg = list(cfg.environment.items())[0]
    task_name = task_cfg.get("task", None)
    match domain_name, task_name:
        case "pendulum", _:
            from smbrl.tasks import pendulum

            make_env, make_sampler = pendulum.make(cfg)
        case _ if any(
            domain_name == domain and task_name == task for domain, task in rwrl_envs
        ):
            from smbrl.tasks.rlwr import env_factory as rwrl_factory

            make_env, make_sampler = rwrl_factory.make(cfg)
        case _ if any(
            domain_name == domain and task_name == task
            for domain, task in dm_control_envs
        ):
            from smbrl.tasks.dm_control import make

            make_env, make_sampler = make(cfg)
        case _:
            raise NotImplementedError
    if cfg.training.num_tasks is not None:
        make_sampler = fix_task_sampling(make_sampler, cfg.training.num_tasks)
    return make_env, make_sampler


__all__ = ["make"]
