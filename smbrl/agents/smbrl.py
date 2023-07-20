from math import ceil
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from smbrl import metrics as m
from smbrl.agents import rssm
from smbrl.agents.actor_critic import ModelBasedActorCritic
from smbrl.agents.base import AgentBase
from smbrl.logging import TrainingLogger
from smbrl.replay_buffer import ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import FloatArray
from smbrl.utils import Count, Learner, add_to_buffer, normalize


class AgentState(NamedTuple):
    rssm_state: rssm.State
    prev_action: jax.Array

    @classmethod
    def init(cls, batch_size: int, cell: rssm.RSSM, action_dim: int) -> "AgentState":
        rssm_state = cell.init
        rssm_state = jax.tree_map(
            lambda x: jnp.repeat(x[None], batch_size, 0), rssm_state
        )
        prev_action = jnp.zeros((batch_size, action_dim))
        self = cls(rssm_state, prev_action)
        return self


@eqx.filter_jit
def policy(actor, model, prev_state, observation, key):
    def per_env_policy(prev_state, observation, key):
        model_key, policy_key = jax.random.split(key)
        current_rssm_state = model.step(
            prev_state.rssm_state, observation, prev_state.prev_action, model_key
        )
        action = actor.act(current_rssm_state.flatten(), policy_key)
        return action, AgentState(current_rssm_state, action)

    return jax.vmap(per_env_policy)(
        prev_state, observation, jax.random.split(key, observation.shape[0])
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
        self.model = rssm.WorldModel(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            key=next(self.prng),
            **config.agent.model,
        )
        self.model_learner = Learner(self.model, config.agent.model_optimizer)
        self.actor_critic = ModelBasedActorCritic(
            config.agent.model.stochastic_size + config.agent.model.deterministic_size,
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
        self.state = AgentState.init(
            config.training.parallel_envs, self.model.cell, np.prod(action_space.shape)
        )
        steps_per_iteration = (
            config.training.time_limit // config.training.action_repeat
        )
        retrain_every = ceil(steps_per_iteration / config.agent.update_steps)
        self.should_train = Count(retrain_every)

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
        actions, self.state = policy(
            self.actor_critic.actor,
            self.model,
            self.state,
            normalized_obs,
            next(self.prng),
        )
        return np.asarray(actions)

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
        for batch in self.replay_buffer.sample(1):
            inferrered_rssm_states = self.update_model(batch)
            initial_states = inferrered_rssm_states.reshape(
                -1, inferrered_rssm_states.shape[-1]
            )
            outs = self.actor_critic.update(self.model, initial_states, next(self.prng))
            for k, v in outs.items():
                self.logger[k] = v

    def update_model(self, batch: TrajectoryData) -> jax.Array:
        features, actions = prepare_features(batch)
        (self.model, self.model_learner.state), (loss, rest) = rssm.variational_step(
            features,
            actions,
            self.model,
            self.model_learner,
            self.model_learner.state,
            next(self.prng),
            self.config.agent.beta,
            self.config.agent.free_nats,
        )
        self.logger["agent/model/loss"] = float(loss.mean())
        self.logger["agent/model/reconstruction"] = float(
            rest["reconstruction_loss"].mean()
        )
        self.logger["agent/model/kl"] = float(rest["kl_loss"].mean())
        return rest["states"].flatten()


def prepare_features(batch: TrajectoryData) -> tuple[rssm.Features, FloatArray]:
    reward = batch.reward[..., None]
    terminals = jnp.zeros_like(reward)
    dones = jnp.zeros_like(reward)
    dones = dones.at[:, -1::].set(1.0)
    features = rssm.Features(
        jnp.asarray(batch.next_observation),
        jnp.asarray(reward),
        jnp.asarray(batch.cost[..., None]),
        jnp.asarray(terminals),
        jnp.asarray(dones),
    )
    flat = lambda x: x.reshape(-1, *x.shape[2:])
    features = jax.tree_map(flat, features)
    return features, flat(batch.action)
