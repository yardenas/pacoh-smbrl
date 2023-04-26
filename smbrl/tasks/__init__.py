from omegaconf import DictConfig

from smbrl.tasks import pendulum
from smbrl.types import MetaEnvironmentFactory
from smbrl.utils import fix_task_sampling


def make(cfg: DictConfig) -> MetaEnvironmentFactory:
    assert len(cfg.tasks.keys()) == 1
    env = list(cfg.tasks.keys())[0]
    match env:
        case "pendulum":
            make_env, make_sampler = pendulum.make(cfg)
        case _:
            raise NotImplementedError
    if cfg.training.num_tasks is not None:
        make_sampler = fix_task_sampling(make_sampler, cfg.training.num_tasks)
    return make_env, make_sampler


__all__ = ["make"]
