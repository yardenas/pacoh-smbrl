from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from smbrl import metrics as m
from smbrl import pacoh_nn as pch
from smbrl.agents.base import AgentBase
from smbrl.logging import TrainingLogger
from smbrl.models import Model
from smbrl.replay_buffer import ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import FloatArray
from smbrl.utils import Learner, add_to_buffer


class ASMBRL(AgentBase):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        super().__init__(config, logger)
        self.obs_normalizer = m.MetricsAccumulator()
        buffer_factory = lambda c, b: ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            precision=config.training.precision,
            sequence_length=config.agents.asmbrl.replay_buffer.sequence_length
            // config.training.action_repeat,
            num_shots=config.agents.asmbrl.replay_buffer.num_shots,
            batch_size=config.agents.asmbrl.replay_buffer.batch_size,
            capacity=c,
            num_episodes=b,
        )
        self.slow_buffer = buffer_factory(
            config.agents.asmbrl.replay_buffer.capacity,
            config.training.episodes_per_task,
        )
        self.fast_buffer = buffer_factory(
            config.training.parallel_envs, config.training.adaptation_budget
        )
        model_factory = lambda key: Model(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            key=key,
            **config.agents.asmbrl.model,
        )
        self.adaptive_model = AdaptiveBayesianModel(
            model_factory, next(self.prng), config
        )

    def __call__(
        self,
        observation: FloatArray,
    ) -> FloatArray:
        # Algorithm sketch:
        # 1. Normalize observation
        # 2. If step mode N == 0:
        # 3.   update_policy (non-blocking/async, dispatch to a thread).
        #      Can use splines to make CEM search space better on longer horizons.
        # 4. get action from current policy.
        # normalized_obs = normalize(
        #     observation, self.obs_normalizer.result.mean,
        # self.obs_normalizer.result.std
        # )
        horizon = self.config.agents.asmbrl.plan_horizon
        init_guess = jnp.zeros((horizon, self.fast_buffer.action.shape[-1]))
        action = init_guess[0]
        return np.asarray(action)

    def observe(self, trajectory: TrajectoryData) -> None:
        self.obs_normalizer.update_state(
            np.concatenate(
                [trajectory.observation, trajectory.next_observation[:, -1:]],
                axis=1,
            ),
            axis=(0, 1),
        )
        add_to_buffer(
            self.slow_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )
        self.episodes += 1

    def adapt(self, trajectory: TrajectoryData) -> None:
        add_to_buffer(
            self.fast_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )


class AdaptiveBayesianModel:
    def __init__(
        self,
        model_factory: Callable[[jax.random.KeyArray], Model],
        key: jax.random.KeyArray,
        config: DictConfig,
    ):
        self.hyper_prior = pch.make_hyper_prior(model_factory)
        key, n_key = jax.random.split(key)
        self.hyper_posterior = pch.make_hyper_posterior(
            model_factory,
            n_key,
            config.agents.asmbrl.pacoh.n_particles,
            config.agents.asmbrl.pacoh.posterior_stddev,
        )
        self.model_learner = Learner(
            self.hyper_posterior, config.agents.asmbrl.model_optimizer
        )
