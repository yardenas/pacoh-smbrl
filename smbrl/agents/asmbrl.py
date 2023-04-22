from typing import Callable

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
        self.pacoh_learner = PACOHLearner(model_factory, next(self.prng), config)
        self.model = self.pacoh_learner.sample_prior(next(self.prng))

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
        self.pacoh_learner.update_hyper_posterior(
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
        self.model = self.pacoh_learner.infer_posteriors(
            data,
            next(self.prng),
            posterior_cfg.update_steps,
            posterior_cfg.n_prior_samples,
            posterior_cfg.learning_rate,
            posterior_cfg.prior_weight,
            posterior_cfg.bandwidth,
        )

    def reset(self):
        pass


# class Posterior(eqx.Module):
#     model: Model

#     # Two challenges:
#     # 1. How to use the ensemble? (Look below for a hint)
#     # 2. Overcome the multi-task/batched setting.
#     def sample(
#         self,
#         horizon: int,
#         initial_state: jax.Array,
#         key: jax.random.KeyArray,
#         action_sequence: Optional[jax.Array] = None,
#     ) -> Prediction:
#         pass
#         # TODO (yarden): think of how to use the posterior for sampling.
#         # Hint: we model/learn trajectories, so the posterior
#         # predictive distribution is over _trajectories_
#         ensemble_sample = lambda model: eqx.filter_vmap(model.sample)(
#             horizon, initial_state, action_sequence
#         )
#         ensemble_sample = eqx.filter_vmap(ensemble_sample)
#         return ensemble_sample(self.model)

#     def step(self, state: jax.Array, action: jax.Array) -> Prediction:
#         # FIXME: self.model here is wrong. It's fine to have a function that computes
#         # outputs based on state-action pairs for an ensemble,
#         # but self.model is an ensemle of ensembles, one ensemble for each task
#         mus, stddevs = pch.predict(self.model, to_ins(state, action))
#         state_dim = self.state_decoder.out_features // 2
#         split = lambda x: jnp.split(x, [state_dim], axis=-1)  # type: ignore
#         state_mu, reward_mu = split(mus)
#         state_stddev, reward_stddev = split(stddevs)
#         return Prediction(state_mu, reward_mu, state_stddev, reward_stddev)


class PACOHLearner:
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

    def update_hyper_posterior(
        self,
        data: Data,
        key: jax.random.KeyArray,
        n_prior_samples: int,
        prior_weight: float,
        bandwidth: float,
    ) -> None:
        (self.hyper_posterior, self.learner.state), logprobs = ml.pacoh_regression(
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
    ) -> Model:
        def infer_posterior_per_task(data):
            posterior, _ = pch.infer_posterior(
                data,
                self.hyper_posterior,
                update_steps,
                n_prior_samples,
                learning_rate,
                key,
                prior_weight,
                bandwidth,
            )
            return posterior

        # TODO (yarden): can be computed once on an outside scope.
        infer_posterior_per_task = jax.jit(jax.vmap(infer_posterior_per_task, 1))
        posteriors: Model = infer_posterior_per_task(data)
        return posteriors

    def sample_prior(self, key: jax.random.KeyArray) -> Model:
        model: Model = self.hyper_posterior.sample(key)
        return model
