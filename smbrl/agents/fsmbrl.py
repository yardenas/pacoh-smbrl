from typing import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from smbrl.agents import cem
from smbrl.agents import model_learning as ml
from smbrl.agents.base import AgentBase
from smbrl.agents.models import SSM, S4Model
from smbrl.logging import TrainingLogger
from smbrl.metrics import MetricsAccumulator
from smbrl.replay_buffer import ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import FloatArray
from smbrl.utils import Count, Learner, add_to_buffer, normalize

regression_step = eqx.filter_jit(ml.regression_step)


@eqx.filter_jit
def update_state(
    model: S4Model,
    ssm: list[SSM],
    prev_hidden: list[jax.Array],
    observation: jax.Array,
    action: jax.Array,
) -> list[jax.Array]:
    step = lambda o, a, h: model.step(o, a, False, ssm, h)
    hidden = jax.vmap(step)(observation, action, prev_hidden)[0]
    return hidden


@eqx.filter_jit
def policy(
    model: S4Model,
    ssm: list[SSM],
    prev_hidden: list[jax.Array],
    observation: jax.Array,
    horizon: int,
    init_guess: jax.Array,
    key: jax.random.KeyArray,
    cem_config: cem.CEMConfig,
) -> jax.Array:
    # vmap over observations and hiddens states for each task in the batch.
    cem_per_env = jax.vmap(
        lambda o, h: cem.policy(
            o,
            bind_to_model(model.sample, layers_hidden=h, layers_ssm=ssm),
            horizon,
            init_guess,
            key,
            cem_config,
        )
    )
    action = cem_per_env(observation, prev_hidden)
    return action


class fSMBRL(AgentBase):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        super().__init__(config, logger)
        self.obs_normalizer = MetricsAccumulator()
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
        self.model = S4Model(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            key=next(self.prng),
            sequence_length=config.agent.replay_buffer.sequence_length
            // config.training.action_repeat,
            **config.agent.model,
        )
        self.model_learner = Learner(
            self.model,
            config.agent.model_optimizer,
        )
        self.replan = Count(config.agent.replan_every)
        self.plan = np.zeros(
            (config.training.parallel_envs, config.agent.plan_horizon)
            + action_space.shape
        )
        self.s4_state = init_state(self.model, self.config.training.task_batch_size)
        self.ssm = self.model.ssm

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
            init_guess = jnp.zeros((horizon, self.replay_buffer.action.shape[-1]))
            action = policy(
                self.model,
                self.ssm,
                self.s4_state,
                normalized_obs,
                horizon,
                init_guess,
                next(self.prng),
                self.config.agent.cem,
            )
            self.plan = np.asarray(action)
        action = self.plan[:, self.replan.count]
        self.s4_state = update_state(
            self.model, self.ssm, self.s4_state, observation, action
        )
        return action

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
        self.s4_state = init_state(self.model, self.config.training.task_batch_size)

    def update_model(self):
        for batch in sample_data(self.replay_buffer, self.config.agent.update_steps):
            (self.model, self.model_learner.state), loss = regression_step(
                batch,
                self.model,
                self.model_learner,
                self.model_learner.state,
            )
        self.logger["agent/model/loss"] = float(loss.mean())
        self.ssm = self.model.ssm

    def reset(self):
        pass


def bind_to_model(fn, **kwargs):
    return eqx.Partial(fn, **kwargs)


def sample_data(
    replay_buffer: ReplayBuffer, n_batches: int
) -> Iterator[TrajectoryData]:
    for batch in replay_buffer.sample(n_batches):
        yield ml.prepare_data(batch)


def init_state(model, batch_size):
    return list(map(lambda x: np.tile(x, (batch_size, 1, 1)), model.init_state))
