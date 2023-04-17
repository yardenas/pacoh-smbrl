import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig

from smbrl import cem
from smbrl import metrics as m
from smbrl import model_learning as ml
from smbrl.agents.base import AgentBase
from smbrl.logging import TrainingLogger
from smbrl.models import Model
from smbrl.replay_buffer import ReplayBuffer
from smbrl.trajectory import TrajectoryData
from smbrl.types import FloatArray, ModelUpdateFn
from smbrl.utils import Learner, add_to_buffer, normalize


class SMBRL(AgentBase):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: DictConfig,
        logger: TrainingLogger,
        model_update_fn: ModelUpdateFn = ml.simple_regression,
    ):
        super().__init__(config, logger)
        self.obs_normalizer = m.MetricsAccumulator()
        self.replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            precision=config.training.precision,
            sequence_length=config.smbrl.replay_buffer.sequence_length
            // config.training.action_repeat,
            num_shots=config.smbrl.replay_buffer.num_shots,
            batch_size=config.smbrl.replay_buffer.batch_size,
            capacity=config.smbrl.replay_buffer.capacity,
            num_episodes=config.training.episodes_per_task,
        )
        self.model = Model(
            state_dim=np.prod(observation_space.shape),
            action_dim=np.prod(action_space.shape),
            key=next(self.prng),
            **config.smbrl.model,
        )
        self.model_learner = Learner(self.model, config.smbrl.model_optimizer)
        self.model_update_fn = model_update_fn
        self.episodes = 0

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
        action = self.policy(
            normalized_obs,
            self.model,
        )
        return np.asarray(action)

    # TODO (yarden): this can be refactored out into a planner
    # (which can be used by other agents)
    @eqx.filter_jit
    def policy(
        self,
        observation: jax.Array,
        model: Model,
    ) -> jax.Array:
        def solve(observation):
            horizon = self.config.smbrl.plan_horizon
            objective = cem.make_objective(model, horizon, observation)
            init_quess = jnp.zeros((horizon, self.replay_buffer.action.shape[-1]))
            action = cem.solve(
                objective,
                init_quess,
                jax.random.PRNGKey(self.config.training.seed),
                **self.config.smbrl.cem,
            )[0]
            return action

        actions: jax.Array = jax.vmap(solve)(observation)
        return actions

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
        batches = [
            batch for batch in self.replay_buffer.sample(self.config.smbrl.update_steps)
        ]
        # What happens below:
        # 1. Transpose list of (named-)tuples into a named tuple of lists
        # 2. Stack the lists for each data type inside
        # the named tuple (e.g., observations, actions, etc.)
        transposed_batches = TrajectoryData(*map(np.stack, zip(*batches)))
        x = ml.to_ins(transposed_batches.observation, transposed_batches.action)
        y = ml.to_outs(transposed_batches.next_observation, transposed_batches.reward)
        (self.model, self.model_learner.state), loss = self.model_update_fn(
            (x, y),
            self.model,
            self.model_learner,
            self.model_learner.state,
            next(self.prng),
        )
        self.logger["agent/model/loss"] = float(loss.mean())
        self.logger.log_metrics(self.episodes)
