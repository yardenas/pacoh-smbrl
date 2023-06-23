import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from smbrl import metrics as m
from smbrl.agents import maki
from smbrl.agents.actor_critic import ModelBasedActorCritic
from smbrl.agents.base import AgentBase
from smbrl.logging import TrainingLogger
from smbrl.replay_buffer import OnPolicyReplayBuffer, ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import FloatArray
from smbrl.utils import Learner, add_to_buffer, contextualize, normalize


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
def policy(actor, observation, key):
    return jax.vmap(lambda o: actor.act(o, key))(observation)


class MMBRL(AgentBase):
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
        self.model = maki.WorldModel(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            context_size=config.agent.model.context_size,
            key=next(self.prng),
        )
        self.context = np.zeros(
            [config.training.parallel_envs, config.agent.model.context_size]
        )
        self.model_learner = Learner(self.model, config.agent.model_optimizer)
        self.actor_critic = ModelBasedActorCritic(
            np.prod(observation_space.shape) + config.agent.model.context_size,
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

    def __call__(
        self,
        observation: FloatArray,
    ) -> FloatArray:
        normalized_obs = normalize(
            observation,
            self.obs_normalizer.result.mean,
            self.obs_normalizer.result.std,
        )
        normalized_obs = contextualize(normalized_obs, self.context)
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
            self.slow_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )
        self.update()

    def adapt(self, trajectory: TrajectoryData) -> None:
        if self.config.agent.model.context_size == 0:
            return
        add_to_buffer(
            self.fast_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )
        features = prepare_features(trajectory)
        context_update = self.model.infer_context(
            features, jnp.asarray(trajectory.action)
        )
        self.context = (np.asarray(context_update.shift) + self.context) / 2

    def update(self) -> None:
        for batch in self.slow_buffer.sample(self.config.agent.update_steps):
            context_posterior = self.update_model(batch)
            context_posterior, initial_states = prepare_actor_critic_batch(
                context_posterior, batch.observation
            )
            actor_loss, critic_loss = self.actor_critic.update(
                contextualize_sample(self.model.sample, context_posterior.shift),
                initial_states,
                next(self.prng),
            )
            self.logger["agent/actor/loss"] = float(actor_loss.mean())
            self.logger["agent/critic/loss"] = float(critic_loss.mean())

    def update_model(self, batch: TrajectoryData) -> maki.ShiftScale:
        features = prepare_features(batch)
        (self.model, self.model_learner.state), (loss, rest) = maki.variational_step(
            features,
            batch.action,
            batch.next_observation,
            self.model,
            self.model_learner,
            self.model_learner.state,
            next(self.prng),
            0.0,
            0.01,
        )
        self.logger["agent/model/loss"] = float(loss.mean())
        self.logger["agent/model/reconstruction"] = float(
            rest["reconstruction_loss"].mean()
        )
        self.logger["agent/model/kl"] = float(rest["kl_loss"].mean())
        return rest["posterior"]

    def reset(self):
        self.fast_buffer = buffer_factory("fast", **self.buffer_kwargs)(
            self.config.training.parallel_envs,
            self.config.training.adaptation_budget,
            self.config.agent.replay_buffer.batch_size,
        )
        self.context = np.ones_like(self.context)


def contextualize_sample(sample, context):
    fn = jax.vmap(sample, (None, 0, None, None, 0))
    return lambda h, i, k, p: fn(h, i, k, p, context)


def prepare_features(batch: TrajectoryData) -> maki.Features:
    reward = batch.reward[..., None]
    terminals = np.zeros_like(reward)
    dones = np.zeros_like(reward)
    dones[:, -1::] = 1.0
    features = maki.Features(
        batch.observation, reward, batch.cost[..., None], terminals, dones
    )
    return features


def prepare_actor_critic_batch(context, observation):
    shape = observation.shape[1:3]
    tile = lambda c: jnp.tile(c[:, None, None], (1, shape[0], shape[1], 1))
    context_posterior = jax.tree_map(tile, context)
    flatten = lambda x: x.reshape(-1, x.shape[-1])
    initial_states = flatten(observation)
    contexts = jax.tree_map(flatten, context_posterior)
    return contexts, initial_states
