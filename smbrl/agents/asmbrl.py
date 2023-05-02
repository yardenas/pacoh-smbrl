from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

import smbrl.agents.model_learning as ml
from smbrl import metrics as m
from smbrl.agents import cem
from smbrl.agents import pacoh_nn as pch
from smbrl.agents.base import AgentBase
from smbrl.agents.models import FeedForwardModel
from smbrl.logging import TrainingLogger
from smbrl.replay_buffer import OnPolicyReplayBuffer, ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import Data, FloatArray
from smbrl.utils import Count, Learner, add_to_buffer, ensemble_predict, normalize


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


@eqx.filter_jit
def policy(
    model: FeedForwardModel,
    observation: jax.Array,
    horizon: int,
    init_guess: jax.Array,
    key: jax.random.KeyArray,
    cem_config: cem.CEMConfig,
):
    def sample_per_task(model):
        # We get an ensemble of models which are specialized for a specific task.
        # 1. Use each member of the ensemble to make predictions.
        # 2. Average over these predictions to (approximately) marginalize
        #    over posterior parameters.
        ensemble_sample = lambda m, h, o, k, a: ensemble_predict(
            m.sample, (None, 0, None, 0)
        )(
            h,
            o[None],
            k,
            a[None],
        )
        return lambda h, o, k, a: jax.tree_map(
            lambda x: x.squeeze(1).mean(0), ensemble_sample(model, h, o, k, a)
        )

    cem_per_env = jax.vmap(
        lambda m, o: cem.policy(
            o, sample_per_task(m), horizon, init_guess, key, cem_config
        ),
    )
    action = cem_per_env(model, observation)
    return action


@eqx.filter_jit
def infer_posterior_per_task(
    data: Data,
    prior: FeedForwardModel,
    posterior: FeedForwardModel,
    update_steps: int,
    learning_rate: float,
    key: jax.random.KeyArray,
    prior_weight: float,
    bandwidth: float,
) -> tuple[FeedForwardModel, jax.Array]:
    def infer_posterior(data, posterior):
        posterior, logprobs = pch.infer_posterior(
            data,
            prior,
            posterior,
            update_steps,
            learning_rate,
            key,
            prior_weight,
            bandwidth,
        )
        return posterior, logprobs

    infer_posterior = jax.vmap(infer_posterior, (1, 0))
    posteriors, logprobs = infer_posterior(data, posterior)
    return posteriors, logprobs.mean()


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
        self.buffer_kwargs = dict(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            precision=config.training.precision,
            sequence_length=config.agent.replay_buffer.sequence_length
            // config.training.action_repeat,
            num_shots=config.agent.replay_buffer.num_shots,
        )
        self.slow_buffer = buffer_factory("slow", **self.buffer_kwargs)(
            config.agent.replay_buffer.capacity,
            config.training.episodes_per_task,
            config.agent.replay_buffer.batch_size,
        )
        self.fast_buffer = buffer_factory("fast", **self.buffer_kwargs)(
            config.training.parallel_envs,
            config.training.adaptation_budget,
            config.agent.replay_buffer.batch_size,
        )
        model_factory = lambda key: FeedForwardModel(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            key=key,
            **config.agent.model,
        )
        self.pacoh_learner = PACOHLearner(model_factory, next(self.prng), config)
        self.model = self.pacoh_learner.sample_prior(
            next(self.prng),
            config.agent.posterior.n_prior_samples,
            config.training.task_batch_size,
        )
        self.replan = Count(config.agent.replan_every)
        self.plan = np.zeros(
            (config.training.parallel_envs, config.agent.plan_horizon)
            + action_space.shape
        )

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
            init_guess = jnp.zeros((horizon, self.fast_buffer.action.shape[-1]))
            action = policy(
                self.model,
                normalized_obs,
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
            self.slow_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )
        data = ml.prepare_data(self.slow_buffer, self.config.agent.pacoh.num_examples)
        pacoh_cfg = self.config.agent.pacoh
        logprobs = self.pacoh_learner.update_hyper_posterior(
            data,
            next(self.prng),
            pacoh_cfg.n_prior_samples,
            self.config.agent.update_steps,
            pacoh_cfg.prior_weight,
            pacoh_cfg.bandwidth,
        )
        self.logger["agent/model/pacoh_logprobs"] = float(logprobs)

    def adapt(self, trajectory: TrajectoryData) -> None:
        add_to_buffer(
            self.fast_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )
        posterior_cfg = self.config.agent.posterior
        data = ml.prepare_data(self.fast_buffer, posterior_cfg.num_examples)
        self.model, logprobs = self.pacoh_learner.infer_posteriors(
            data,
            self.model,
            posterior_cfg.update_steps,
            posterior_cfg.learning_rate,
            next(self.prng),
            posterior_cfg.prior_weight,
            posterior_cfg.bandwidth,
        )
        self.logger["agent/model/posterior_logprobs"] = float(logprobs)

    def reset(self):
        self.model = self.pacoh_learner.sample_prior(
            next(self.prng),
            self.config.agent.posterior.n_prior_samples,
            self.config.training.task_batch_size,
        )
        self.fast_buffer = buffer_factory("fast", **self.buffer_kwargs)(
            self.config.training.parallel_envs,
            self.config.training.adaptation_budget,
            self.config.agent.replay_buffer.batch_size,
        )


class PACOHLearner:
    def __init__(
        self,
        model_factory: Callable[[jax.random.KeyArray], FeedForwardModel],
        key: jax.random.KeyArray,
        config: DictConfig,
    ):
        key, n_key = jax.random.split(key)
        self.hyper_prior = pch.make_hyper_prior(model_factory(n_key))
        key, n_key = jax.random.split(key)
        self.hyper_posterior = pch.make_hyper_posterior(
            model_factory,
            n_key,
            config.agent.pacoh.n_particles,
            config.agent.pacoh.posterior_stddev,
        )
        self.learner = Learner(self.hyper_posterior, config.agent.model_optimizer)
        self.prior = pch.compute_prior(self.hyper_posterior)

    def update_hyper_posterior(
        self,
        data: Data,
        key: jax.random.KeyArray,
        n_prior_samples: int,
        update_steps: int,
        prior_weight: float,
        bandwidth: float,
    ) -> jax.Array:
        (self.hyper_posterior, self.learner.state), logprobs = ml.pacoh_regression(
            data,
            self.hyper_prior,
            self.hyper_posterior,
            self.learner,
            self.learner.state,
            update_steps,
            n_prior_samples,
            prior_weight,
            bandwidth,
            key,
        )
        return logprobs.mean()

    def infer_posteriors(
        self,
        data: Data,
        posterior: FeedForwardModel,
        update_steps: int,
        learning_rate: float,
        key: jax.random.KeyArray,
        prior_weight: float,
        bandwidth: float,
    ) -> FeedForwardModel:
        return infer_posterior_per_task(
            data,
            self.prior,
            posterior,
            update_steps,
            learning_rate,
            key,
            prior_weight,
            bandwidth,
        )

    def sample_prior(
        self, key: jax.random.KeyArray, n_prior_samples: int, task_batch_size: int
    ) -> FeedForwardModel:
        model = jax.vmap(pch.sample_prior_models, (None, 0, None))(
            self.hyper_posterior,
            jax.random.split(key, task_batch_size),
            n_prior_samples,
        )
        self.prior = pch.compute_prior(self.hyper_posterior)
        return model
