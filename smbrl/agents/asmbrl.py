from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

import smbrl.model_learning as ml
from smbrl import metrics as m
from smbrl import pacoh_nn as pch
from smbrl.agents.base import AgentBase
from smbrl.logging import TrainingLogger
from smbrl.models import Model
from smbrl.replay_buffer import OnPolicyReplayBuffer, ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import Data, FloatArray
from smbrl.utils import Learner, add_to_buffer


def buffer_factory(
    buffer_type,
    observation_shape,
    action_shape,
    max_length,
    seed,
    precision,
    sequence_length,
    num_shots,
):
    cls = dict(slow=ReplayBuffer, fast=OnPolicyReplayBuffer)[buffer_type]
    make = lambda capacity, budget, batch_size: cls(
        observation_shape=observation_shape,
        action_shape=action_shape,
        max_length=max_length,
        seed=seed,
        precision=precision,
        sequence_length=sequence_length,
        num_shots=num_shots,
        batch_size=batch_size,
        capacity=capacity,
        num_episodes=budget,
    )
    return make


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
        buffer_kwargs = dict(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            precision=config.training.precision,
            sequence_length=config.agents.asmbrl.replay_buffer.sequence_length
            // config.training.action_repeat,
            num_shots=config.agents.asmbrl.replay_buffer.num_shots,
        )
        self.slow_buffer = buffer_factory("slow", **buffer_kwargs)(
            config.agents.asmbrl.replay_buffer.capacity,
            config.training.episodes_per_task,
            config.agents.asmbrl.replay_buffer.batch_size,
        )
        self.fast_buffer = buffer_factory("fast", **buffer_kwargs)(
            config.training.parallel_envs,
            config.training.adaptation_budget,
            32,
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
        return np.repeat(np.asarray(action)[None], observation.shape[0], axis=0)

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
        data = ml.prepare_data(self.slow_buffer, self.config.agents.asmbrl.update_steps)
        pacoh_cfg = self.config.agents.asmbrl.pacoh
        self.adaptive_model.update_hyper_posterior(
            data,
            next(self.prng),
            pacoh_cfg.n_prior_samples,
            pacoh_cfg.prior_weight,
            pacoh_cfg.bandwidth,
        )
        self.episodes += 1

    def adapt(self, trajectory: TrajectoryData) -> None:
        add_to_buffer(
            self.fast_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )
        posterior_cfg = self.config.agents.asmbrl.posterior
        data = ml.prepare_data(self.fast_buffer, posterior_cfg.update_steps)
        self.adaptive_model.infer_posteriors(
            data,
            next(self.prng),
            posterior_cfg.update_steps,
            posterior_cfg.n_prior_samples,
            posterior_cfg.learning_rate,
            posterior_cfg.prior_weight,
            posterior_cfg.bandwidth,
        )

    def reset(self):
        self.adaptive_model.reset_model(next(self.prng))


class AdaptiveBayesianModel:
    def __init__(
        self,
        model_factory: Callable[[jax.random.KeyArray], Model],
        key: jax.random.KeyArray,
        config: DictConfig,
    ):
        key, n_key = jax.random.split(key)
        self.hyper_prior = pch.make_hyper_prior(model_factory(n_key))
        key, n_key = jax.random.split(key)
        self.hyper_posterior = pch.make_hyper_posterior(
            model_factory,
            n_key,
            config.agents.asmbrl.pacoh.n_particles,
            config.agents.asmbrl.pacoh.posterior_stddev,
        )
        self.learner = Learner(
            self.hyper_posterior, config.agents.asmbrl.model_optimizer
        )
        self.posteriors = self.hyper_posterior.sample(key)

    def update_hyper_posterior(
        self,
        data: Data,
        key: jax.random.KeyArray,
        n_prior_samples: int,
        prior_weight: float,
        bandwidth: float,
    ) -> None:
        pacoh = eqx.filter_jit(ml.pacoh_regression)
        (self.hyper_posterior, self.learner.state), logprobs = pacoh(
            data,
            self.hyper_prior,
            self.hyper_posterior,
            self.learner,
            self.learner.state,
            n_prior_samples,
            prior_weight,
            bandwidth,
            key,
        )

    def infer_posteriors(
        self,
        data: Data,
        key: jax.random.KeyArray,
        update_steps: int,
        n_prior_samples: int,
        learning_rate: float,
        prior_weight: float,
        bandwidth: float,
    ) -> None:
        infer_posterior_per_task = lambda data: pch.infer_posterior(
            data,
            self.hyper_posterior,
            update_steps,
            n_prior_samples,
            learning_rate,
            key,
            prior_weight,
            bandwidth,
        )
        infer_posterior_per_task = jax.jit(jax.vmap(infer_posterior_per_task, 1))
        self.posteriors, _ = infer_posterior_per_task(data)

    def reset_model(self, key: jax.random.KeyArray) -> None:
        self.posteriors = self.hyper_posterior.sample(key)
