import logging

import hydra
from omegaconf import OmegaConf

from smbrl import tasks
from smbrl.trainer import Trainer

log = logging.getLogger("experiment")


@hydra.main(version_base=None, config_path="smbrl/configs", config_name="config")
def experiment(cfg):
    log.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    make_env, task_sampler = tasks.make(cfg)
    with Trainer(cfg, make_env, task_sampler) as trainer:
        trainer.train()


if __name__ == "__main__":
    experiment()
