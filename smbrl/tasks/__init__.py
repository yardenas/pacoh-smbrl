from omegaconf import DictConfig

from smbrl.tasks import pendulum
from smbrl.types import MetaEnvironmentFactory
from smbrl.utils import fix_task_sampling


def make(cfg: DictConfig) -> MetaEnvironmentFactory:
    if "pendulum" in cfg.tasks:
        make_env = pendulum.make_env_factory(cfg)
        make_sampler = pendulum.make_gravity_sampler(
            cfg.training.seed, cfg.tasks.pendulum.limits
        )
    else:
        raise NotImplementedError
    if cfg.training.num_tasks is not None:
        make_sampler = fix_task_sampling(make_sampler, cfg.training.num_tasks)
    return make_env, make_sampler


__all__ = ["make"]
