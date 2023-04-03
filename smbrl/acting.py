from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from smbrl import utils
from smbrl.episodic_async_env import EpisodicAsync
from smbrl.iteration_summary import IterationSummary
from smbrl.logging import TrainingLogger
from smbrl.smbrl import SMBRL
from smbrl.trajectory import Trajectory, TrajectoryData, Transition


def log_results(trajectory: TrajectoryData, logger: TrainingLogger, step: int):
    logger.log_summary(
        {
            "train/episode_reward_mean": float(trajectory.reward.sum(1).mean()),
            "train/episode_cost_mean": float(trajectory.cost.sum(1).mean()),
        },
        step,
    )


def interact(
    agent: SMBRL,
    environment: EpisodicAsync,
    num_episodes: int,
    train: bool,
    render_episodes: int = 0,
):
    observations = environment.reset()
    episode_count = 0
    episodes: list[Trajectory] = []
    trajectory = Trajectory()
    with tqdm(total=num_episodes) as pbar:
        while episode_count < num_episodes:
            if render_episodes:
                trajectory.frames.append(environment.render())
            actions = agent(observations)
            next_observations, rewards, done, infos = environment.step(actions)
            costs = np.array([info.get("cost", 0) for info in infos])
            transition = Transition(
                observations,
                next_observations,
                actions,
                rewards,
                costs,
            )
            trajectory.transitions.append(transition)
            observations = next_observations
            if done.all():
                np_trajectory = trajectory.as_numpy()
                if train:
                    agent.observe(np_trajectory)
                agent.reset()
                log_results(np_trajectory, agent.logger, agent.episodes)
                render_episodes = max(render_episodes - 1, 0)
                observations = environment.reset()
                episodes.append(trajectory)
                trajectory = Trajectory()
                pbar.update(1)
                episode_count += 1
    return episodes


def epoch(
    agent: SMBRL,
    env: EpisodicAsync,
    tasks: Iterable[Any],
    episodes_per_task: int,
    train: bool,
    render_episodes: int = 0,
) -> IterationSummary:
    summary = IterationSummary()
    batches = list(utils.grouper(tasks, env.num_envs))
    for batch in batches:
        assert len(batch) == env.num_envs
        env.reset(options={"task": batch})
        samples = interact(
            agent,
            env,
            episodes_per_task,
            train=train,
            render_episodes=render_episodes,
        )
        summary.extend(samples)
    return summary
