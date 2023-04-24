from gymnasium.spaces import Box
from omegaconf import DictConfig

from smbrl.agents.asmbrl import ASMBRL
from smbrl.agents.smbrl import SMBRL
from smbrl.logging import TrainingLogger
from smbrl.types import Agent


def make(
    observation_space: Box,
    action_space: Box,
    config: DictConfig,
    logger: TrainingLogger,
) -> Agent:
    if "smbrl" in config.agents:
        return SMBRL(
            observation_space,
            action_space,
            config,
            logger,
        )
    elif "asmbrl" in config.agents:
        return ASMBRL(
            observation_space,
            action_space,
            config,
            logger,
        )
    else:
        raise NotImplementedError


__all__ = ["make"]
