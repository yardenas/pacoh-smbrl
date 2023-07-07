from typing import Callable, Iterable, Optional

from gymnasium import Env
from gymnasium.spaces import Box
from omegaconf import DictConfig
from realworldrl_suite.environments import realworld_env as rwrl

from smbrl.tasks.rlwr.wrappers import RWRLWrapper
from smbrl.types import MetaEnvironmentFactory, TaskSampler


def alter_task(env: RWRLWrapper, param: float):
    env.env._task._perturb_cur = param
    env.env._physics = env.env._task.update_physics()


def sampler_factory(
    surrogate_env: rwrl.Base,
) -> TaskSampler:
    def sampler(
        batch_size: int, train: Optional[bool] = False
    ) -> Iterable[float] | Iterable[tuple[float, float, float]]:
        for _ in range(batch_size):
            if surrogate_env.env.task.perturb_enabled:
                surrogate_env.env.task._generate_parameter()
            param = surrogate_env.env.task._perturb_cur
            yield param

    return sampler


def make_env_factory(cfg: DictConfig) -> Callable[[], Env[Box, Box]]:
    def make_env() -> Env[Box, Box]:
        from smbrl.tasks.rlwr.wrappers import RWRLWrapper
        from smbrl.wrappers import MetaEnv

        # Create an environment that does not perturb variables on reset.
        # (https://github.com/google-research/realworldrl_suite/blob/be7a51cffa7f5f9cb77a387c16bad209e0f851f8/realworldrl_suite/utils/wrappers.py#L97)
        # Pertubation parameters are sampled via the surrogate, and are applied
        # from within the alter_task function.
        env = RWRLWrapper(
            list(cfg.environment.keys())[0],
            cfg.environment.task,
            cfg.environment.safety_spec,
            {"enable": False, "period": 0, "scheduler": "constant"},
        )
        # Make `generate_parameter` a no-op function. This allows calls to
        # `update_physics` to bypass the paramerter sampling and use the
        # samples from the surrogate.
        env.env._task._generate_parameter = lambda *args: None
        env.seed(cfg.trainig.seed)
        meta_env = MetaEnv(env, alter_task)
        return meta_env

    return make_env


def make(cfg: DictConfig) -> MetaEnvironmentFactory:
    doman_name, task_cfg = list(cfg.environment.items())[0]
    surrogate = RWRLWrapper(
        doman_name,
        task_cfg.task,
        dict(task_cfg.safety_spec),
        dict(task_cfg.perturb_spec),
    )
    surrogate.seed(cfg.training.seed)
    sampler = sampler_factory(surrogate)
    make_env = make_env_factory(cfg)
    return make_env, sampler
