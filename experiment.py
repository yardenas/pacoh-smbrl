import logging
import os

import hydra
from omegaconf import OmegaConf

from smbrl import tasks
from smbrl.trainer import Trainer

log = logging.getLogger("experiment")


def should_resume():
    log_path = os.getcwd()
    state_path = os.path.join(log_path, "state.pkl")
    return os.path.exists(state_path)


def start_fresh(cfg):
    make_env, task_sampler = tasks.make(cfg)
    return Trainer(cfg, make_env, task_sampler)


def load_state(cfg, state_path):
    return Trainer.from_pickle(cfg, state_path)


@hydra.main(version_base=None, config_path="smbrl/configs", config_name="config")
def experiment(cfg):
    log.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    log_path = os.getcwd()
    state_path = os.path.join(log_path, "state.pkl")
    should_resume = os.path.exists(state_path)
    if should_resume:
        log.info(f"Resuming experiment from: {state_path}")
        trainer = load_state(cfg, state_path)
    else:
        log.info("Starting a new experiment.")
        trainer = start_fresh(cfg)
    with trainer as trainer:
        trainer.train()


if __name__ == "__main__":
    experiment()
