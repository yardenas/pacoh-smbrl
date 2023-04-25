from omegaconf import DictConfig

from smbrl.tasks import pendulum
from smbrl.types import MetaEnvironmentFactory


def make(cfg: DictConfig) -> MetaEnvironmentFactory:
    if "pendulum" in cfg.tasks:
        return pendulum.make_env_factory(cfg), pendulum.make_gravity_sampler(
            cfg.training.seed, cfg.tasks.pendulum.limits
        )
    else:
        raise NotImplementedError


__all__ = ["make"]
