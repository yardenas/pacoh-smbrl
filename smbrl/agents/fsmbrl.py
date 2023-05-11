from typing import Iterator

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from smbrl.agents import model_learning as ml
from smbrl.agents import smbrl
from smbrl.agents.base import AgentBase
from smbrl.agents.models import S4Model
from smbrl.logging import TrainingLogger
from smbrl.metrics import MetricsAccumulator
from smbrl.replay_buffer import ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import FloatArray
from smbrl.utils import Count, Learner, add_to_buffer, normalize

regression_step = eqx.filter_jit(ml.regression_step)


class fSMBRL(AgentBase):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        super().__init__(config, logger)
        self.obs_normalizer = MetricsAccumulator()
        self.replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            precision=config.training.precision,
            sequence_length=config.agent.replay_buffer.sequence_length
            // config.training.action_repeat,
            num_shots=config.agent.replay_buffer.num_shots,
            batch_size=config.agent.replay_buffer.batch_size,
            capacity=config.agent.replay_buffer.capacity,
            num_episodes=config.training.episodes_per_task,
        )
        self.model = S4Model(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            key=next(self.prng),
            sequence_length=config.agent.replay_buffer.sequence_length
            // config.training.action_repeat,
            **config.agent.model,
        )
        self.model_learner = Learner(
            self.model,
            config.agent.model_optimizer,
        )
        self.replan = Count(config.agent.replan_every)
        self.plan = np.zeros(
            (config.training.parallel_envs, config.agent.plan_horizon)
            + action_space.shape
        )
        self.s4_state = self.model.init_state
        self.ssm = self.model.ssm

    def __call__(
        self,
        observation: FloatArray,
    ) -> FloatArray:
        if self.replan():
            normalized_obs = normalize(
                observation,
                self.obs_normalizer.result.mean,
                self.obs_normalizer.result.std,
            )
            horizon = self.config.agent.plan_horizon
            init_guess = jnp.zeros((horizon, self.replay_buffer.action.shape[-1]))
            action = smbrl.policy(
                normalized_obs,
                bind_to_model(
                    self.model.sample, layers_ssm=self.ssm, layers_hidden=self.s4_state
                ),
                horizon,
                init_guess,
                next(self.prng),
                self.config.agent.cem,
            )
            self.plan = np.asarray(action)
        return self.plan[:, self.replan.count]

    def observe(self, trajectory: TrajectoryData) -> None:
        self.obs_normalizer.update_state(
            np.concatenate(
                [trajectory.observation, trajectory.next_observation[:, -1:]],
                axis=1,
            ),
            axis=(0, 1),
        )
        add_to_buffer(
            self.replay_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )
        self.update_model()

    def update_model(self):
        for batch in sample_data(self.replay_buffer, self.config.agent.update_steps):
            (self.model, self.model_learner.state), loss = regression_step(
                batch,
                self.model,
                self.model_learner,
                self.model_learner.state,
            )
        self.logger["agent/model/loss"] = float(loss.mean())


def bind_to_model(fn, **kwargs):
    return eqx.Partial(fn, **kwargs)


def sample_data(
    replay_buffer: ReplayBuffer, n_batches: int
) -> Iterator[TrajectoryData]:
    for batch in replay_buffer.sample(n_batches):
        yield ml.prepare_data(batch)
