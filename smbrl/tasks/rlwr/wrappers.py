"""https://github.com/jqueeney/robust-safe-rl/tree/main"""


import numpy as np

from smbrl.tasks.dm_control import DMCWrapper


def make_rwrl_env(domain_name, task_name, env_setup_kwargs):
    """Creates Real World RL Suite task."""
    env = RWRLWrapper(domain_name, task_name, env_setup_kwargs)
    return env


class RWRLWrapper(DMCWrapper):
    """Wrapper to convert Real World RL Suite tasks to OpenAI Gym format."""

    def __init__(self, domain_name, task_name, safety_spec, perturb_spec):
        """Initializes Real World RL Suite tasks.

        Args:
            domain_name (str): name of RWRL Suite domain
            task_name (str): name of RWRL Suite task
        """
        import realworldrl_suite.environments as rwrl

        env = rwrl.load(
            domain_name=domain_name,
            task_name=task_name,
            safety_spec=safety_spec,
            perturb_spec=perturb_spec,
        )
        exclude_keys = ["constraints"]
        self._setup(env, exclude_keys)

    def step(self, a):
        """Takes step in environment.

        Args:
            a (np.ndarray): action

        Returns:
            s (np.ndarray): flattened next state
            r (float): reward
            t (bool): terminal flag
            truncated (bool): truncated flag
            info (dict): dictionary with additional environment info
        """
        s, r, t, d, info = super(RWRLWrapper, self).step(a)
        constraints = info.get("constraints", np.array([True]))
        cost = 1.0 - np.all(constraints)
        info["cost"] = cost
        return s, r, t, d, info
