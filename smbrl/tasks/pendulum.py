import inspect
from os import path
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
            assert len(limits) == 2
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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        initial_angle = angle_normalize(np.pi - self.theta_0)
        if options is not None:
            options["init_x"] = initial_angle
        else:
            options = {"init_x": initial_angle}
        return self.env.reset(options=options)

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
            angle_normalize(th - self.theta_0) ** 2
            + 0.1 * thdot**2
            + 0.001 * (u**2)
        )
        newthdot = (
            thdot
            + (
                3 * g / (2 * length) * np.sin(th - self.theta_0)
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
            self.render()
        return self.unwrapped._get_obs(), -costs, False, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            return

        import pygame
        from pygame import gfxdraw

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = self.unwrapped.l * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )
        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gravity_direction = (0, -3.0)
        gravity_direction = pygame.math.Vector2(gravity_direction).rotate_rad(
            self.theta_0
        )
        gravity_direction = (
            int(gravity_direction[0] * scale) + offset,
            int(gravity_direction[1] * scale) + offset,
        )
        gfxdraw.line(
            self.surf,
            offset,
            offset,
            gravity_direction[0],
            gravity_direction[1],
            (51, 51, 255),
        )
        fname = path.join(
            path.dirname(inspect.getfile(self.unwrapped.__class__)),
            "assets/clockwise.png",
        )
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


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
