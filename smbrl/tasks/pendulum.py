from typing import Callable, Iterable, NamedTuple, Optional, no_type_check

import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.core import Wrapper
from gymnasium.envs.classic_control.pendulum import angle_normalize
from gymnasium.spaces import Box
from omegaconf import DictConfig

from smbrl.types import MetaEnvironmentFactory, TaskSampler


def alter_gravity_direction(env: "GravityPendulum", theta: float) -> None:
    env.theta_0 = theta


@no_type_check
def alter_gravity_magnitude(env: "GravityPendulum", g: float) -> None:
    env.unwrapped.g = g


class PendulumParams(NamedTuple):
    g: float
    theta_0: float
    length: float


@no_type_check
def alter_all(env: "GravityPendulum", params: PendulumParams) -> None:
    env.unwrapped.g = params.g
    env.unwrapped.theta_0 = params.theta_0
    env.unwrapped.l = params.length  # noqa E741


def make_sampler(
    seed: int,
    limits: tuple[float, float] | tuple[npt.ArrayLike, npt.ArrayLike],
    _all: bool,
) -> TaskSampler:
    rs = np.random.RandomState(seed)

    def gravity_sampler(
        batch_size: int, train: Optional[bool] = False
    ) -> Iterable[float] | Iterable[tuple[float, float, float]]:
        if _all:
            limits_array = np.split(np.asarray(limits).T, 2)
        else:
            limits_array = np.split(np.asarray(limits), 2)
        for _ in range(batch_size):
            sample = rs.uniform(*limits_array)
            if not _all:
                yield float(sample)
            else:
                yield PendulumParams(*map(float, sample[0]))

    return gravity_sampler


class GravityPendulum(Wrapper[Box, Box]):
    def __init__(self, env):
        super().__init__(env)
        self.theta_0 = 0.0

    @no_type_check
    def step(self, u):
        th, thdot = self.unwrapped.state  # th := theta

        g = self.unwrapped.g
        m = self.unwrapped.m
        length = self.unwrapped.l
        dt = self.unwrapped.dt

        u = np.clip(u, -self.unwrapped.max_torque, self.unwrapped.max_torque)[0]
        self.unwrapped.last_u = u  # for rendering
        costs = (
            angle_normalize(th + self.theta_0) ** 2
            + 0.1 * thdot**2
            + 0.001 * (u**2)
        )

        newthdot = (
            thdot
            + (
                3 * g / (2 * length) * np.sin(th + self.theta_0)
                + 3.0 / (m * length**2) * u
            )
            * dt
        )
        newthdot = np.clip(
            newthdot, -self.unwrapped.max_speed, self.unwrapped.max_speed
        )
        newth = th + newthdot * dt

        self.unwrapped.state = np.array([newth, newthdot])

        if self.unwrapped.render_mode == "human":
            self.unwrapped.render()
        return self.unwrapped._get_obs(), -costs, False, False, {}


def make_env_factory(cfg: DictConfig) -> Callable[[], Env[Box, Box]]:
    def env() -> Env[Box, Box]:
        import gymnasium

        from smbrl.wrappers import MetaEnv

        env = gymnasium.make("Pendulum-v1", render_mode="rgb_array")
        env._max_episode_steps = cfg.training.time_limit  # type: ignore
        env = GravityPendulum(env)
        match cfg.environment.pendulum.vary:
            case "gravity_direction":
                alter_fn = alter_gravity_direction
            case "gravity_magnitude":
                alter_fn = alter_gravity_magnitude
            case "all":
                alter_fn = alter_all
            case _:
                raise NotImplementedError
        env = MetaEnv(env, alter_fn)
        return env

    return env


def make(cfg: DictConfig) -> MetaEnvironmentFactory:
    make_env = make_env_factory(cfg)
    _all = cfg.environment.pendulum.vary == "all"
    sampler = make_sampler(cfg.training.seed, cfg.environment.pendulum.limits, _all)
    return make_env, sampler
