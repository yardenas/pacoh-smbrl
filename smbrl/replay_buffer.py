from typing import Iterator, Optional

import numpy as np
from tensorflow import data as tfd

from smbrl.trajectory import TrajectoryData


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        max_length: int,
        seed: int,
        capacity: int,
        num_episodes: int,
        batch_size: int,
        num_shots: int,
        sequence_length: int,
        precision: int,
    ):
        self.task_id = 0
        self.dtype = {16: np.float16, 32: np.float32}[precision]
        self.observation = np.zeros(
            (
                capacity,
                num_episodes,
                max_length + 1,
            )
            + observation_shape,
            dtype=self.dtype,
        )
        self.action = np.zeros(
            (
                capacity,
                num_episodes,
                max_length,
            )
            + action_shape,
            dtype=self.dtype,
        )
        self.reward = np.zeros(
            (
                capacity,
                num_episodes,
                max_length,
            ),
            dtype=self.dtype,
        )
        self.cost = np.zeros(
            (
                capacity,
                num_episodes,
                max_length,
            ),
            dtype=self.dtype,
        )
        self.valid_tasks = 0
        self.episode_id = 0
        self.rs = np.random.RandomState(seed)
        example = next(
            iter(self._sample_batch(batch_size, sequence_length, num_shots, capacity))
        )
        self._generator = lambda: self._sample_batch(
            batch_size, sequence_length, num_shots
        )
        self._dataset = _make_dataset(self._generator, example)

    @property
    def num_episodes(self):
        return self.cost.shape[1]

    def add(self, trajectory: TrajectoryData) -> None:
        capacity, *_ = self.reward.shape
        batch_size = min(trajectory.observation.shape[0], capacity)
        # Discard data if batch size overflows capacity.
        end = min(self.task_id + batch_size, capacity)
        task_slice = slice(self.task_id, end)
        for data, val in zip(
            (self.action, self.reward, self.cost),
            (trajectory.action, trajectory.reward, trajectory.cost),
        ):
            data[task_slice, self.episode_id] = val[:batch_size].astype(self.dtype)
        observation = np.concatenate(
            [
                trajectory.observation[:batch_size],
                trajectory.next_observation[:batch_size, -1:],
            ],
            axis=1,
        ).astype(self.dtype)
        self.observation[task_slice, self.episode_id] = observation
        self.episode_id += 1
        if self.episode_id == self.num_episodes:
            self.task_id = (self.task_id + batch_size) % capacity
            self.valid_tasks = min(self.valid_tasks + batch_size, capacity)
            self.episode_id = 0

    def _sample_batch(
        self,
        batch_size: int,
        sequence_length: int,
        num_shots: int,
        valid_tasks: Optional[int] = None,
    ):
        if valid_tasks is not None:
            valid_tasks = valid_tasks
        else:
            valid_tasks = self.valid_tasks
        num_episodes, time_limit = self.observation.shape[1:3]
        assert time_limit > sequence_length and num_episodes >= num_shots
        while True:
            timestep_ids = _make_ids(
                self.rs,
                time_limit - sequence_length - 1,
                sequence_length + 1,
                batch_size,
                (0, 1),
            )
            episode_ids = _make_ids(
                self.rs, self.episode_id + 2 - num_shots, num_shots, batch_size, (0, 2)
            )
            task_ids = self.rs.choice(valid_tasks, size=batch_size)
            # Sample a sequence of length H for the actions, rewards and costs,
            # and a length of H + 1 for the observations (which is needed for
            # value-function bootstrapping)
            a, r, c = [
                x[task_ids[:, None, None], episode_ids, timestep_ids[..., :-1]]
                for x in (
                    self.action,
                    self.reward,
                    self.cost,
                )
            ]
            obs_sequence = self.observation[
                task_ids[:, None, None], episode_ids, timestep_ids
            ]
            o = obs_sequence[:, :, :-1]
            next_o = obs_sequence[:, :, 1:]
            yield o, next_o, a, r, c

    def sample(self, n_batches: int) -> Iterator[TrajectoryData]:
        if self.empty:
            return
        for batch in self._dataset.take(n_batches):
            yield TrajectoryData(*map(lambda x: x.numpy(), batch))

    def sample_batch(self) -> TrajectoryData:
        batch = next(self._dataset.take(1))
        return TrajectoryData(*map(lambda x: x.numpy(), batch))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_dataset"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        example = next(iter(self._generator()))
        self._dataset = _make_dataset(self._generator, example)

    @property
    def empty(self):
        return self.valid_tasks == 0


def _make_ids(rs, low, n_samples, batch_size, dim):
    low = rs.choice(low, batch_size)
    ids_axis = np.arange(n_samples)
    ids_axis = np.expand_dims(ids_axis, axis=dim)
    ids = low[:, None, None] + np.repeat(ids_axis, batch_size, axis=0)
    return ids


def _make_dataset(generator, example):
    dataset = tfd.Dataset.from_generator(
        generator,
        *zip(*tuple((v.dtype, v.shape) for v in example)),
    )
    dataset = dataset.prefetch(10)
    return dataset
