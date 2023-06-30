from dataclasses import dataclass, field

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
from smbrl.agents.contextual_actor_critic import ContextualModelBasedActorCritic
from smbrl.logging import TrainingLogger
from smbrl.replay_buffer import ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import FloatArray, Prediction
from smbrl.utils import Learner, add_to_buffer, normalize


def actor_critic_factory(observation_space, action_space, config, key, belief):
    if config.agent.model.context_size == 0:
        return ModelBasedActorCritic(
            np.prod(observation_space.shape),
            np.prod(action_space.shape),
            config.agent.actor,
            config.agent.critic,
            config.agent.actor_optimizer,
            config.agent.critic_optimizer,
            config.agent.plan_horizon,
            config.agent.discount,
            config.agent.lambda_,
            key,
        )
    else:
        return ContextualModelBasedActorCritic(
            np.prod(observation_space.shape),
            np.prod(action_space.shape),
            config.agent.actor,
            config.agent.critic,
            config.agent.actor_optimizer,
            config.agent.critic_optimizer,
            config.agent.plan_horizon,
            config.agent.discount,
            config.agent.lambda_,
            key,
            belief,
        )


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
        self.adaptation_buffer = TrajectoryBuffer()
        self.model = maki.WorldModel(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            num_dynamics_layers=config.agent.model.num_dynamics_layers,
            dynamics_size=config.agent.model.dynamics_size,
            num_context_layers=config.agent.model.num_context_layers,
            hidden_size=config.agent.model.hidden_size,
            intermediate_size=config.agent.model.intermediate_size,
            num_heads=config.agent.model.num_heads,
            context_size=config.agent.model.context_size,
            key=next(self.prng),
        )
        belief = jnp.zeros(
            [config.training.parallel_envs, config.agent.model.context_size]
        )
        self.context_belief = maki.ShiftScale(belief, jnp.ones_like(belief))
        self.model_learner = Learner(self.model, config.agent.model_optimizer)
        self.actor_critic = actor_critic_factory(
            observation_space,
            action_space,
            config,
            next(self.prng),
            self.context_belief,
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
        if self.contextual:
            state = maki.BeliefAndState(self.context_belief, normalized_obs)
        else:
            state = normalized_obs
        action = policy(self.actor_critic.actor, state, next(self.prng))
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
        self.update()

    def adapt(self, trajectory: TrajectoryData) -> None:
        if not self.contextual:
            return
        add_to_buffer(
            self.adaptation_buffer,
            trajectory,
            self.obs_normalizer,
            self.config.training.scale_reward,
        )
        trajectories = self.adaptation_buffer.get()
        self.context_belief = infer_context(trajectories, self.model)
        evaluate_model(self.model, trajectories, self.context_belief)

    def update(self) -> None:
        for batch in self.replay_buffer.sample(self.config.agent.update_steps):
            context_posterior = self.update_model(batch)
            context_posterior, initial_states = prepare_actor_critic_batch(
                context_posterior, batch.observation
            )
            if self.contextual:
                self.actor_critic.contextualize(context_posterior)
            actor_loss, critic_loss = self.actor_critic.update(
                self.model,
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
            1e-4,
            0.01,
        )
        self.logger["agent/model/loss"] = float(loss.mean())
        self.logger["agent/model/reconstruction"] = float(
            rest["reconstruction_loss"].mean()
        )
        self.logger["agent/model/kl"] = float(rest["kl_loss"].mean())
        return rest["posterior"]

    def reset(self):
        self.context_belief = maki.ShiftScale(
            jnp.zeros_like(self.context_belief.shift),
            jnp.ones_like(self.context_belief.scale),
        )
        self.adaptation_buffer.reset()

    @property
    def contextual(self):
        return self.config.agent.model.context_size > 0


@eqx.filter_jit
def infer_context(batch: TrajectoryData, model: maki.WorldModel) -> maki.ShiftScale:
    features = prepare_features(batch)
    return jax.vmap(model.infer_context)(features, jnp.asarray(batch.action))


def prepare_features(batch: TrajectoryData) -> maki.Features:
    reward = batch.reward[..., None]
    terminals = jnp.zeros_like(reward)
    dones = jnp.zeros_like(reward)
    dones = dones.at[:, -1::].set(1.0)
    features = maki.Features(
        jnp.asarray(batch.observation),
        jnp.asarray(reward),
        jnp.asarray(batch.cost[..., None]),
        jnp.asarray(terminals),
        jnp.asarray(dones),
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


@dataclass
class TrajectoryBuffer:
    data: list[TrajectoryData] = field(default_factory=list)

    def add(self, more_data: TrajectoryData) -> None:
        self.data.append(more_data)

    def get(self) -> TrajectoryData:
        # list of tuples -> tuple of lists
        data = TrajectoryData(*map(lambda x: np.stack(x, 1)[:, :3], zip(*self.data)))
        return data

    def reset(self) -> None:
        self.data = []


def evaluate_model(model, batch, context):
    horizon = 15
    key = jax.random.PRNGKey(10)
    pred = eqx.filter_vmap(lambda o, a, c: model.sample(horizon, o[0], key, a, c))(
        batch.observation[:, -1], batch.action[:, -1, :horizon], context
    )
    pred = Prediction(pred.next_state.state, pred.reward)
    pred = jnp.concatenate([pred.next_state, pred.reward[..., None]], axis=-1)
    plot(
        batch.observation[:, -1:, :1],
        batch.next_observation[:, -1:, :horizon],
        pred[:, None],
        1,
        "model.png",
    )
    return pred


def plot(context, y, y_hat, context_t, savename):
    import matplotlib.pyplot as plt

    t_test = np.arange(y.shape[2])
    t_context = np.arange(context.shape[2])

    plt.figure(figsize=(10, 5), dpi=600)
    for i in range(min(6, context.shape[0])):
        plt.subplot(3, 4, i + 1)
        plt.plot(t_context, context[i, 0, :, 2], "b.", label="context")
        plt.plot(
            t_test,
            y_hat[i, 0, :, 2],
            "r",
            label="prediction",
            linewidth=1.0,
        )
        plt.plot(
            t_test,
            y[i, 0, :, 2],
            "c",
            label="ground truth",
            linewidth=1.0,
        )
        ax = plt.gca()
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(context_t, color="k", linestyle="--", linewidth=1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savename, bbox_inches="tight")
    plt.show(block=False)
    plt.close()
