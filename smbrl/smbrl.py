from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from numpy import typing as npt
from omegaconf import DictConfig
from optax import OptState

from smbrl import cem
from smbrl import metrics as m
from smbrl.logging import TrainingLogger
from smbrl.models import Model
from smbrl.replay_buffer import ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.utils import Learner

FloatArray = npt.NDArray[Union[np.float32, np.float64]]


class smbrl:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        self.prng = jax.random.PRNGKey(config.training.seed)
        self.config = config
        self.logger = logger
        self.obs_normalizer = m.MetricsAccumulator()
        self.replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            precision=config.training.precision,
            sequence_length=config.smbrl.replay_buffer.sequence_length
            // config.training.action_repeat,
            batch_size=config.smbrl.replay_buffer.batch_size,
            capacity=config.smbrl.replay_buffer.capacity,
        )
        self.model = Model(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            sequence_length=config.smbrl.replay_buffer.sequence_length,
            key=self.prng,
            **config.smbrl.model,
        )
        self.model_learner = Learner(self.model, config.smbrl.model_optimizer)
        self.episodes = 0

    def __call__(
        self,
        observation: FloatArray,
    ) -> FloatArray:
        normalized_obs = _normalize(
            observation, self.obs_normalizer.result.mean, self.obs_normalizer.result.std
        )
        action, self.hidden = self.policy(
            normalized_obs,
            self.model,
        )
        return np.asarray(action)

    @eqx.filter_jit
    def policy(
        self,
        observation: jax.Array,
        model: Model,
    ) -> jax.Array:
        def solve(observation):
            horizon = self.config.smbrl.plan_horizon
            objective = cem.make_objective(model, horizon, observation)
            action = cem.solve(
                objective,
                jnp.zeros(
                    (
                        horizon,
                        self.replay_buffer.action.shape[-1],
                    )
                ),
                jax.random.PRNGKey(self.config.training.seed),
                **self.config.smbrl.cem,
            )[0]
            return action

        return jax.vmap(solve)(observation)

    def observe(self, trajectory: TrajectoryData):
        self.obs_normalizer.update_state(
            np.concatenate(
                [trajectory.observation, trajectory.next_observation[:, -1:]], axis=1
            ),
            axis=(0, 1),
        )
        results = self.obs_normalizer.result
        normalize = lambda x: _normalize(x, results.mean, results.std)
        self.replay_buffer.add(
            TrajectoryData(
                normalize(trajectory.observation),
                normalize(trajectory.next_observation),
                trajectory.action,
                trajectory.reward * self.config.training.scale_reward,
                trajectory.cost,
            )
        )
        self.train()
        self.episodes += 1

    def reset(self):
        pass

    def train(self):
        for batch in self.replay_buffer.sample(self.config.smbrl.update_steps):
            loss, self.model, self.model_learner.state = self.update_model(
                self.model, self.model_learner.state, batch
            )
            self.logger["agent/model/loss"] = loss
        self.logger.log_metrics(self.episodes)

    @eqx.filter_jit
    def update_model(self, model: Model, opt_state: OptState, batch: TrajectoryData):
        def loss(
            model,
            state_sequence,
            action_sequence,
            next_state_sequence,
            reward_sequence,
        ):
            preds = jax.vmap(model)(state_sequence, action_sequence)[1]
            state_loss = (preds.next_state - next_state_sequence) ** 2
            reward_loss = (preds.reward.squeeze(-1) - reward_sequence) ** 2
            return 0.5 * (state_loss.mean() + reward_loss.mean())

        loss_fn = eqx.filter_value_and_grad(loss)
        loss, grads = loss_fn(
            model, batch.observation, batch.action, batch.next_observation, batch.reward
        )
        new_model, new_opt_state = self.model_learner.grad_step(model, grads, opt_state)
        return loss, new_model, new_opt_state

    def __getstate__(self):
        """
        Define how the agent should be pickled.
        """
        state = self.__dict__.copy()
        del state["logger"]
        return state

    def __setstate__(self, state):
        """
        Define how the agent should be loaded.
        """
        self.__dict__.update(state)
        self.logger = TrainingLogger(self.config.log_dir)


def _normalize(
    observation: FloatArray,
    mean: FloatArray,
    std: FloatArray,
) -> FloatArray:
    diff = observation - mean
    return diff / (std + 1e-8)
