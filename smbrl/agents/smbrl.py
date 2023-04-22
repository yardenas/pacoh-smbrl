import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from smbrl import metrics as m
from smbrl.agents import cem
from smbrl.agents import model_learning as ml
from smbrl.agents.base import AgentBase
from smbrl.agents.models import Model
from smbrl.logging import TrainingLogger
from smbrl.replay_buffer import ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import FloatArray
from smbrl.utils import Learner, add_to_buffer, normalize


class SMBRL(AgentBase):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        super().__init__(config, logger)
        self.obs_normalizer = m.MetricsAccumulator()
        self.replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            precision=config.training.precision,
            sequence_length=config.agents.smbrl.replay_buffer.sequence_length
            // config.training.action_repeat,
            num_shots=config.agents.smbrl.replay_buffer.num_shots,
            batch_size=config.agents.smbrl.replay_buffer.batch_size,
            capacity=config.agents.smbrl.replay_buffer.capacity,
            num_episodes=config.training.episodes_per_task,
        )
        self.model = Model(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            key=next(self.prng),
            **config.agents.smbrl.model,
        )
        self.model_learner = Learner(self.model, config.agents.smbrl.model_optimizer)

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
        normalized_obs = normalize(
            observation, self.obs_normalizer.result.mean, self.obs_normalizer.result.std
        )
        horizon = self.config.agents.smbrl.plan_horizon
        init_guess = jnp.zeros((horizon, self.replay_buffer.action.shape[-1]))
        action = cem.policy(
            normalized_obs,
            self.model,
            horizon,
            init_guess,
            next(self.prng),
            self.config.agents.smbrl.cem,
        )
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
            self.replay_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )
        self.update_model()
        self.episodes += 1

    def update_model(self):
        x, y = ml.prepare_data(
            self.replay_buffer, self.config.agents.smbrl.update_steps
        )
        (self.model, self.model_learner.state), loss = ml.simple_regression(
            (x, y),
            self.model,
            self.model_learner,
            self.model_learner.state,
            next(self.prng),
        )
        self.logger["agent/model/loss"] = float(loss.mean())
        self.logger.log_metrics(self.episodes)
