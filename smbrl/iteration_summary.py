from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy import typing as npt

from sadam.trajectory import Trajectory


@dataclass
class IterationSummary:
    _data: list[list[Trajectory]] = field(default_factory=list)
    cost_boundary: float = 25.0

    @property
    def empty(self):
        return len(self._data) == 0

    @property
    def metrics(self) -> Tuple[float, float, float]:
        rewards, costs = [], []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                *_, r, c = trajectory.as_numpy()
                rewards.append(r)
                costs.append(c)
        # Stack data from all tasks on the first axis,
        # giving a [#tasks, #episodes, #time, ...] shape.
        rewards = np.stack(rewards)
        costs = np.stack(costs)
        return (
            _objective(rewards),
            _cost_rate(costs),
            _feasibility(costs, self.cost_boundary),
        )

    @property
    def videos(self):
        all_vids = []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                all_vids.append(trajectory.frames)
        return all_vids

    def extend(self, samples: List[Trajectory]):
        self._data.append(samples)


def _objective(rewards: npt.NDArray[np.float32]) -> float:
    return rewards.sum(2).mean()


def _cost_rate(costs: npt.NDArray[np.float32]) -> float:
    return costs.mean()


def _feasibility(costs: npt.NDArray[np.float32], boundary) -> float:
    return (costs.sum(2).mean(1) <= boundary).mean()
