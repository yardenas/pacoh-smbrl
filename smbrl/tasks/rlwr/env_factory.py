from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional

import realworldrl_suite.environments as rwrl
from gymnasium import Env
from gymnasium.spaces import Box
from omegaconf import DictConfig
from realworldrl_suite.environments.realworld_env import Base

from smbrl.tasks.rlwr.wrappers import RWRLWrapper
from smbrl.types import MetaEnvironmentFactory, TaskSampler

NO_OP = lambda *args: None

# https://github.com/jqueeney/robust-safe-rl/blob/main/robust_safe_rl/envs/wrappers/rwrl_wrapper.py
rwrl_constraints_cartpole = {
    "slider_pos_constraint": rwrl.cartpole.slider_pos_constraint,
    "balance_velocity_constraint": rwrl.cartpole.balance_velocity_constraint,
    "slider_accel_constraint": rwrl.cartpole.slider_accel_constraint,
}

rwrl_constraints_walker = {
    "joint_angle_constraint": rwrl.walker.joint_angle_constraint,
    "joint_velocity_constraint": rwrl.walker.joint_velocity_constraint,
    "dangerous_fall_constraint": rwrl.walker.dangerous_fall_constraint,
    "torso_upright_constraint": rwrl.walker.torso_upright_constraint,
}

rwrl_constraints_quadruped = {
    "joint_angle_constraint": rwrl.quadruped.joint_angle_constraint,
    "joint_velocity_constraint": rwrl.quadruped.joint_velocity_constraint,
    "upright_constraint": rwrl.quadruped.upright_constraint,
    "foot_force_constraint": rwrl.quadruped.foot_force_constraint,
}

rwrl_constraints_humanoid = {
    "joint_angle_constraint": rwrl.humanoid.joint_angle_constraint,
    "joint_velocity_constraint": rwrl.humanoid.joint_velocity_constraint,
    "upright_constraint": rwrl.humanoid.upright_constraint,
    "dangerous_fall_constraint": rwrl.humanoid.dangerous_fall_constraint,
    "foot_force_constraint": rwrl.humanoid.foot_force_constraint,
}

rwrl_constraints_manipulator = {
    "joint_angle_constraint": rwrl.manipulator.joint_angle_constraint,
    "joint_velocity_constraint": rwrl.manipulator.joint_velocity_constraint,
    "joint_accel_constraint": rwrl.manipulator.joint_accel_constraint,
    "grasp_force_constraint": rwrl.manipulator.grasp_force_constraint,
}

rwrl_constraints_combined = {
    "cartpole": rwrl_constraints_cartpole,
    "walker": rwrl_constraints_walker,
    "quadruped": rwrl_constraints_quadruped,
    "humanoid": rwrl_constraints_humanoid,
    "manipulator": rwrl_constraints_manipulator,
}


def alter_task(env: RWRLWrapper, param: float):
    env.env._task._perturb_cur = param
    env.env._task._physics = env.env._task.update_physics()


def sampler_factory(
    surrogate_env: Base,
) -> TaskSampler:
    if surrogate_env._task.perturb_enabled:
        # Make `generate_parameter` a no-op function. This allows calls to
        # `update_physics` to bypass the paramerter sampling and use the
        # samples from the surrogate.
        surrogate_sample_fn = surrogate_env.task._generate_parameter
        Base._generate_parameter = NO_OP

    def sampler(
        batch_size: int, train: Optional[bool] = False
    ) -> Iterable[float] | Iterable[tuple[float, float, float]]:
        for _ in range(batch_size):
            if surrogate_env._task.perturb_enabled:
                surrogate_sample_fn()
            param = surrogate_env.task._perturb_cur
            yield param

    return sampler


def augment_constraint(
    safety_spec: dict[str, Any], domain_name: str, constraint_names: list[str]
) -> dict[str, Any]:
    if len(constraint_names) == 0:
        return safety_spec
    domain_constraints = rwrl_constraints_combined[domain_name]
    constraint_funs = OrderedDict()
    for constraint in constraint_names:
        constraint_funs[constraint] = domain_constraints[constraint]
    return OrderedDict(safety_spec) | constraint_funs


def make_env_factory(cfg: DictConfig) -> Callable[[], Env[Box, Box]]:
    def make_env() -> Env[Box, Box]:
        from smbrl.tasks.rlwr.wrappers import RWRLWrapper
        from smbrl.wrappers import MetaEnv

        domain_name, task_cfg = list(cfg.environment.items())[0]
        # Create an environment that does not perturb variables on reset.
        # (https://github.com/google-research/realworldrl_suite/blob/be7a51cffa7f5f9cb77a387c16bad209e0f851f8/realworldrl_suite/utils/wrappers.py#L97)
        # Pertubation parameters are sampled via the surrogate, and are applied
        # from within the alter_task function.
        env = RWRLWrapper(
            domain_name,
            task_cfg.task,
            augment_constraint(task_cfg.safety_spec, domain_name, task_cfg.constraints),
            {"enable": False, "period": 0, "scheduler": "constant"},
        )
        env.seed(cfg.training.seed)
        meta_env = MetaEnv(env, alter_task)
        return meta_env

    return make_env


def make(cfg: DictConfig) -> MetaEnvironmentFactory:
    domain_name, task_cfg = list(cfg.environment.items())[0]
    surrogate = RWRLWrapper(
        domain_name,
        task_cfg.task,
        augment_constraint(task_cfg.safety_spec, domain_name, task_cfg.constraints),
        dict(task_cfg.perturb_spec),
    )
    surrogate.seed(cfg.training.seed)
    sampler = sampler_factory(surrogate.env)
    make_env = make_env_factory(cfg)
    return make_env, sampler
