from gymnasium.spaces import Box
from omegaconf import DictConfig

from smbrl.agents.mmbrl import MMBRL
from smbrl.agents.smbrl import SMBRL
from smbrl.logging import TrainingLogger
from smbrl.types import Agent


def make(
    observation_space: Box,
    action_space: Box,
    config: DictConfig,
    logger: TrainingLogger,
) -> Agent:
    match config.agent.name:
        case "smbrl":
            return SMBRL(observation_space, action_space, config, logger)
        case "mmbrl":
            return MMBRL(observation_space, action_space, config, logger)
        case _:
            raise NotImplementedError


__all__ = ["make"]
