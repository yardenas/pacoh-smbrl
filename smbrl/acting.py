from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from smbrl.episodic_async_env import EpisodicAsync  # type: ignore
from smbrl.iteration_summary import IterationSummary
from smbrl.logging import TrainingLogger
from smbrl.trajectory import Trajectory, TrajectoryData, Transition
from smbrl.types import Agent
from smbrl.utils import grouper


def log_results(
    trajectory: TrajectoryData, logger: TrainingLogger, step: int, prefix: str
) -> tuple[float, float]:
    reward = float(trajectory.reward.sum(1).mean())
    cost = float(trajectory.cost.sum(1).mean())
    logger.log_summary(
        {
            f"{prefix}/episode_reward_mean": reward,
            f"{prefix}/episode_cost_mean": cost,
        },
        step,
    )
    return reward, cost


def interact(
    agent: Agent,
    environment: EpisodicAsync,
    num_episodes: int,
    adaptation_episodes: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[list[Trajectory], int]:
    assert 0 <= adaptation_episodes <= num_episodes
    observations = environment.reset()
    episode_count = 0
    episodes: list[Trajectory] = []
    trajectory = Trajectory()
    with tqdm(
        total=num_episodes,
        unit=f"Episode (✕ {environment.num_envs} parallel)",
    ) as pbar:
        while episode_count < num_episodes:
            render = episode_count >= adaptation_episodes - 1 and render_episodes
            if render:
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
            if done.any():
                assert done.all()
                np_trajectory = trajectory.as_numpy()
                step += int(np.prod(np_trajectory.reward.shape))
                if train:
                    agent.observe(np_trajectory)
                if episode_count < adaptation_episodes:
                    agent.adapt(np_trajectory)
                reward, cost = log_results(
                    np_trajectory, agent.logger, step, "train" if train else "evaluate"
                )
                pbar.set_postfix({"reward": reward, "cost": cost})
                render_episodes = max(render_episodes - 1, 0)
                episodes.append(trajectory)
                trajectory = Trajectory()
                pbar.update(1)
                episode_count += 1
                observations = environment.reset()
    return episodes, step


def epoch(
    agent: Agent,
    env: EpisodicAsync,
    tasks: Iterable[Any],
    episodes_per_task: int,
    adaptation_episodes: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[IterationSummary, int]:
    summary = IterationSummary()
    batches = list(grouper(tasks, env.num_envs))
    for batch in batches:
        assert len(batch) == env.num_envs
        env.reset(options={"task": batch})
        samples, step = interact(
            agent,
            env,
            episodes_per_task,
            adaptation_episodes,
            train,
            step,
            render_episodes,
        )
        agent.reset()
        summary.extend(samples)
    return summary, step
