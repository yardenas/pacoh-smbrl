# Getting Started

`python experiment.py agent=asmbrl training.time_limit=100 training.scale_reward=0.1 agent.model_optimizer.lr=1e-3,5e-4,1e-4 hydra/launcher=slurm --multirun`

`python experiment.py environment=rwrl_cartpole +experiment=cartpole hydra/launcher=slurm hydra.launcher.timeout_min=480 log_dir=/cluster/scratch/yardas/tune_safety/crpo environment.rwrl_cartpole.perturb_spec.min=1. environment.rwrl_cartpole.perturb_spec.max=1. +agent/penalizer=lbsgd agent.safety_discount=0.99,0.992,0.993,0.994 --multirun`