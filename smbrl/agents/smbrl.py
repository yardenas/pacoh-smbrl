import equinox as eqx
import jax
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from smbrl import metrics as m
from smbrl.agents import model_learning as ml
from smbrl.agents.actor_critic import ModelBasedActorCritic
from smbrl.agents.base import AgentBase
from smbrl.agents.models import FeedForwardModel
from smbrl.logging import TrainingLogger
from smbrl.replay_buffer import ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import FloatArray
from smbrl.utils import Count, Learner, add_to_buffer, normalize


@eqx.filter_jit
def policy(actor, observation, key):
    return jax.vmap(lambda o, k: actor.act(o, k))(
        observation, jax.random.split(key, observation.shape[0])
    )


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
            sequence_length=config.agent.replay_buffer.sequence_length
            // config.training.action_repeat,
            num_shots=config.agent.replay_buffer.num_shots,
            batch_size=config.agent.replay_buffer.batch_size,
            capacity=config.agent.replay_buffer.capacity,
            num_episodes=config.training.episodes_per_task,
        )
        self.model = FeedForwardModel(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            key=next(self.prng),
            **config.agent.model,
        )
        self.model_learner = Learner(self.model, config.agent.model_optimizer)
        self.actor_critic = ModelBasedActorCritic(
            np.prod(observation_space.shape),
            np.prod(action_space.shape),
            config.agent.actor,
            config.agent.critic,
            config.agent.actor_optimizer,
            config.agent.critic_optimizer,
            config.agent.plan_horizon,
            config.agent.discount,
            config.agent.lambda_,
            next(self.prng),
        )
        self.should_train = Count(config.agent.train_every)

    def __call__(
        self,
        observation: FloatArray,
    ) -> FloatArray:
        if not self.replay_buffer.empty and self.should_train():
            self.update()
        normalized_obs = normalize(
            observation,
            self.obs_normalizer.result.mean,
            self.obs_normalizer.result.std,
        )
        action = policy(self.actor_critic.actor, normalized_obs, next(self.prng))
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

    def update(self) -> None:
        for batch in self.replay_buffer.sample(self.config.agent.update_steps):
            self.update_model(batch)
            initial_states = batch.observation.reshape(-1, batch.observation.shape[-1])
            outs = self.actor_critic.update(self.model, initial_states, next(self.prng))
            for k, v in outs.items():
                self.logger[k] = v

    def update_model(self, batch: TrajectoryData) -> None:
        regression_batch = ml.prepare_data(batch)
        (self.model, self.model_learner.state), loss = ml.regression_step(
            regression_batch,
            self.model,
            self.model_learner,
            self.model_learner.state,
        )
        self.logger["agent/model/loss"] = float(loss.mean())
