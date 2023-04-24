from typing import no_type_check

import numpy as np
from gymnasium.core import Wrapper
from gymnasium.envs.classic_control.pendulum import angle_normalize
from gymnasium.spaces import Box


def alter_gravity(env: "GravityPendulum", theta: float) -> None:
    env.theta_0 = angle_normalize(theta)


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
