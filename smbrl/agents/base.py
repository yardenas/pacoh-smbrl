from omegaconf import DictConfig

from smbrl import types
from smbrl.logging import TrainingLogger
from smbrl.trajectory import TrajectoryData
from smbrl.utils import PRNGSequence


class AgentBase(types.Agent):
    def __init__(
        self,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        self.prng = PRNGSequence(config.training.seed)
        self.config = config
        self.logger = logger
        self.episodes = 0

    def __call__(
        self,
        observation: types.FloatArray,
        train: bool = False,
    ) -> types.FloatArray:
        raise NotImplementedError

    def observe(self, trajectory: TrajectoryData) -> None:
        raise NotImplementedError

    def adapt(self, trajectory: TrajectoryData) -> None:
        pass

    def reset(self):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = TrainingLogger(self.config.log_dir)
