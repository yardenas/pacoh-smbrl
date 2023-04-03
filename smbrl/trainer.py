import os
from typing import Any, Callable, Iterable, List, Optional

import cloudpickle
import numpy as np
from gymnasium import Env
from omegaconf import DictConfig

from smbrl import acting, episodic_async_env, logging, smbrl, utils

TaskSamplerFactory = Callable[[int, Optional[bool]], Iterable[Any]]


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        make_env: Callable[[], Env],
        task_sampler: TaskSamplerFactory,
        agent: Optional[smbrl.smbrl] = None,
        start_epoch: int = 0,
        seeds: Optional[List[int]] = None,
        namespace: Optional[str] = None,
    ):
        self.config = config
        self.agent = agent
        self.make_env = make_env
        self.tasks_sampler = task_sampler
        self.epoch = start_epoch
        self.seeds = seeds
        self.logger = None
        self.state_writer = None
        self.env = None
        self.namespace = namespace

    def __enter__(self):
        if self.namespace is not None:
            log_path = f"{self.config.training.log_dir}/{self.namespace}"
        else:
            log_path = self.config.training.log_dir
        self.logger = logging.TrainingLogger(self.config.training.log_dir)
        self.state_writer = logging.StateWriter(log_path)
        self.env = episodic_async_env.EpisodicAsync(
            self.make_env,
            self.config.training.parallel_envs,
            self.config.training.time_limit,
            self.config.training.action_repeat,
        )
        # Get next batch of tasks.
        tasks = next(utils.grouper(self.tasks(train=True), self.env.num_envs))
        if self.seeds is not None:
            self.env.reset(seed=self.seeds, options={"task": tasks})
        else:
            self.env.reset(seed=self.config.training.seed, options={"task": tasks})
        if self.agent is None:
            self.agent = smbrl.smbrl(
                self.env.observation_space,
                self.env.action_space,
                self.config,
                self.logger,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.logger is not None and self.state_writer is not None
        self.state_writer.close()
        self.logger.close()

    def train(self, epochs: Optional[int] = None):
        epoch, logger, state_writer = self.epoch, self.logger, self.state_writer
        assert logger is not None and state_writer is not None
        for epoch in range(epoch, epochs or self.config.training.epochs):
            print(f"Training epoch #{epoch}")
            self._step(
                train=True,
                episodes_per_task=self.config.training.episodes_per_task,
                prefix="train",
                epoch=epoch,
            )
            if (epoch + 1) % self.config.training.eval_every == 0:
                print("Evaluating...")
                self._step(
                    train=False,
                    episodes_per_task=self.config.training.eval_episodes_per_task,
                    prefix="eval",
                    epoch=epoch,
                )
            self.epoch = epoch + 1
            state_writer.write(self.state)
        logger.flush()

    def _step(self, train: bool, episodes_per_task: int, prefix: str, epoch: int):
        config, agent, env, logger = self.config, self.agent, self.env, self.logger
        assert env is not None and agent is not None and logger is not None
        render_episodes = int(not train) * self.config.training.render_episodes
        summary = acting.epoch(
            agent,
            env,
            self.tasks(train=train),
            episodes_per_task,
            train=train,
            render_episodes=render_episodes,
        )
        step = (
            epoch
            * config.training.episodes_per_task
            * config.training.action_repeat
            * config.training.time_limit
            * config.training.parallel_envs
        )
        objective, cost_rate, feasibilty = summary.metrics
        logger.log_summary(
            {
                f"{prefix}/objective": objective,
                f"{prefix}/cost_rate": cost_rate,
                f"{prefix}/feasibility": feasibilty,
            },
            step,
        )
        if render_episodes > 0:
            logger.log_video(
                np.asarray(summary.videos)[0].swapaxes(0, 1)[:5],
                step,
                "video",
                30 / config.training.action_repeat,
            )

    def get_env_random_state(self):
        assert self.env is not None
        rs = [
            state.get_state()[1]
            for state in self.env.get_attr("rs")
            if state is not None
        ]
        if not rs:
            rs = [
                state.bit_generator.state["state"]["state"]
                for state in self.env.get_attr("np_random")
            ]
        return rs

    def tasks(self, train: bool) -> Iterable[Any]:
        return self.tasks_sampler(self.config.training.task_batch_size, train)

    @classmethod
    def from_pickle(cls, config: DictConfig, namespace: Optional[str] = None):
        if namespace is not None:
            log_path = f"{config.training.log_dir}/{namespace}"
        else:
            log_path = config.training.log_dir
        with open(os.path.join(log_path, "state.pkl"), "rb") as f:
            make_env, env_rs, agent, epoch, task_sampler = cloudpickle.load(f).values()
        print(f"Resuming experiment from: {log_path}...")
        assert agent.config == config, "Loaded different hyperparameters."
        return cls(
            config=agent.config,
            make_env=make_env,
            task_sampler=task_sampler,
            start_epoch=epoch,
            seeds=env_rs,
            agent=agent,
            namespace=namespace,
        )

    @property
    def state(self):
        return {
            "make_env": self.make_env,
            "env_rs": self.get_env_random_state(),
            "agent": self.agent,
            "epoch": self.epoch,
            "task_sampler": self.tasks_sampler,
        }