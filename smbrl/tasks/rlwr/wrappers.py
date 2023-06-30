"""https://github.com/jqueeney/robust-safe-rl/tree/main"""

import copy
from collections import OrderedDict

import numpy as np
import realworldrl_suite.environments as rwrl
from dm_control import suite
from dm_control.suite.wrappers import action_scale
from dm_env import specs
from gymnasium import spaces


def make_dmc_env(domain_name, task_name):
    """Creates DeepMind Control task with actions scaled to [-1,1]."""
    env = DMCWrapper(domain_name, task_name)
    return env


# From:
# https://github.com/rail-berkeley/softlearning/blob/master/softlearning/environments/adapters/dm_control_adapter.py
def convert_dm_control_to_gym_space(dm_control_space):
    """Recursively convert dm_control_space into gym space.
    Note: Need to check the following cases of the input type, in the following
    order:
       (1) BoundedArraySpec
       (2) ArraySpec
       (3) OrderedDict.
    - Generally, dm_control observation_specs are OrderedDict with other spaces
      (e.g. ArraySpec) nested in it.
    - Generally, dm_control action_specs are of type `BoundedArraySpec`.
    To handle dm_control observation_specs as inputs, we check the following
    input types in order to enable recursive calling on each nested item.
    """
    if isinstance(dm_control_space, specs.BoundedArray):
        shape = dm_control_space.shape
        low = np.broadcast_to(dm_control_space.minimum, shape)
        high = np.broadcast_to(dm_control_space.maximum, shape)
        gym_box = spaces.Box(
            low=low, high=high, shape=None, dtype=dm_control_space.dtype
        )
        # Note: `gym.Box` doesn't allow both shape and min/max to be defined
        # at the same time. Thus we omit shape in the constructor and verify
        # that it's been implicitly set correctly.
        assert gym_box.shape == dm_control_space.shape, (
            gym_box.shape,
            dm_control_space.shape,
        )
        return gym_box
    elif isinstance(dm_control_space, specs.Array):
        if isinstance(dm_control_space, specs.BoundedArray):
            raise ValueError("The order of the if-statements matters.")
        return spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(
                dm_control_space.shape
                if (
                    len(dm_control_space.shape) == 1
                    or (
                        len(dm_control_space.shape) == 3
                        and np.issubdtype(dm_control_space.dtype, np.integer)
                    )
                )
                else (int(np.prod(dm_control_space.shape)),)
            ),
            dtype=dm_control_space.dtype,
        )
    elif isinstance(dm_control_space, OrderedDict):
        return spaces.Dict(
            OrderedDict(
                [
                    (key, convert_dm_control_to_gym_space(value))
                    for key, value in dm_control_space.items()
                ]
            )
        )
    else:
        raise ValueError(dm_control_space)


# Modified from:
# https://github.com/rail-berkeley/softlearning/blob/master/softlearning/environments/adapters/dm_control_adapter.py
class DMCWrapper:
    """Wrapper to convert DeepMind Control tasks to OpenAI Gym format."""

    def __init__(self, domain_name, task_name):
        """Initializes DeepMind Control tasks.

        Args:
            domain_name (str): name of DeepMind Control domain
            task_name (str): name of DeepMind Control task
        """

        # Supports tasks in DeepMind Control Suite
        env = suite.load(domain_name=domain_name, task_name=task_name)

        self._setup(env)

    def _setup(self, env, exclude_keys=[]):
        """Sets up environment and corresponding spaces.

        Args:
            env (object): DeepMind Control Suite environment
            exclude_keys (list): list of keys to exclude from observation
        """
        assert isinstance(env.observation_spec(), OrderedDict)
        assert isinstance(env.action_spec(), specs.BoundedArray)

        env = action_scale.Wrapper(
            env,
            minimum=np.ones_like(env.action_spec().minimum) * -1,
            maximum=np.ones_like(env.action_spec().maximum),
        )
        np.testing.assert_equal(env.action_spec().minimum, -1)
        np.testing.assert_equal(env.action_spec().maximum, 1)
        self.env = env

        # Can remove parts of observation by excluding keys here
        observation_keys = tuple(env.observation_spec().keys())
        self.observation_keys = tuple(
            key for key in observation_keys if key not in exclude_keys
        )

        observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())

        self.observation_space = type(observation_space)(
            [
                (name, copy.deepcopy(space))
                for name, space in observation_space.spaces.items()
                if name in self.observation_keys
            ]
        )

        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())

        if len(self.action_space.shape) > 1:
            raise NotImplementedError(
                "Shape of the action space ({}) is not flat, make sure to"
                " check the implemenation.".format(self.action_space)
            )

    def _filter_observation(self, observation):
        """Filters excluded keys from observation."""
        observation = type(observation)(
            (
                (name, np.reshape(value, self.observation_space.spaces[name].shape))
                for name, value in observation.items()
                if name in self.observation_keys
            )
        )
        return observation

    def step(self, a):
        """Takes step in environment.

        Args:
            a (np.ndarray): action

        Returns:
            s (np.ndarray): flattened next state
            r (float): reward
            d (bool): done flag
            info (dict): dictionary with additional environment info
        """
        time_step = self.env.step(a)
        r = time_step.reward or 0.0
        d = time_step.last()
        info = {
            key: value
            for key, value in time_step.observation.items()
            if key not in self.observation_keys
        }
        observation = self._filter_observation(time_step.observation)
        s = spaces.utils.flatten(self.observation_space, observation)
        return s, r, d, info

    def reset(self):
        """Resets environment and returns flattened initial state."""
        time_step = self.env.reset()
        observation = self._filter_observation(time_step.observation)
        s = spaces.utils.flatten(self.observation_space, observation)
        return s

    def seed(self, seed):
        self.env.task._random = np.random.RandomState(seed)

    def render(self, camera_id=None, **kwargs):
        self.env.task.visualize_reward = True
        if camera_id is None:
            camera_id = -1
        return self.env.physics.render(camera_id=camera_id)


def make_rwrl_env(domain_name, task_name, env_setup_kwargs):
    """Creates Real World RL Suite task."""
    env = RWRLWrapper(domain_name, task_name, env_setup_kwargs)
    return env


class RWRLWrapper(DMCWrapper):
    """Wrapper to convert Real World RL Suite tasks to OpenAI Gym format."""

    def __init__(self, domain_name, task_name, env_setup_kwargs):
        """Initializes Real World RL Suite tasks.

        Args:
            domain_name (str): name of RWRL Suite domain
            task_name (str): name of RWRL Suite task
            env_setup_kwargs (dict): setup parameters
        """
        safety_spec = create_safety_spec(domain_name, env_setup_kwargs)
        perturb_spec = create_perturb_spec(env_setup_kwargs)
        noise_spec = create_noise_spec(env_setup_kwargs)
        env = rwrl.load(
            domain_name=domain_name,
            task_name=task_name,
            safety_spec=safety_spec,
            perturb_spec=perturb_spec,
            noise_spec=noise_spec,
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
            d (bool): done flag
            info (dict): dictionary with additional environment info
        """
        s, r, d, info = super(RWRLWrapper, self).step(a)

        constraints = info.get("constraints", np.array([True]))
        cost = 1.0 - np.all(constraints)
        info["cost"] = cost

        return s, r, d, info


# Config helper functions
#########################################


def create_safety_spec(domain_name, env_setup_kwargs):
    """Creates safety_spec dictionary."""
    safety_spec = {
        "enable": True,
        "observations": True,
        "safety_coeff": env_setup_kwargs["safety_coeff"],
    }

    rwrl_constraints_list = env_setup_kwargs["rwrl_constraints"]
    rwrl_constraints_domain = rwrl_constraints_combined[domain_name]
    if env_setup_kwargs["rwrl_constraints_all"]:
        rwrl_constraints_list = list(rwrl_constraints_domain.keys())

    if rwrl_constraints_list:
        rwrl_constraints = OrderedDict()
        for constraint in rwrl_constraints_list:
            rwrl_constraints[constraint] = rwrl_constraints_domain[constraint]

        safety_spec["constraints"] = rwrl_constraints

    return safety_spec


def create_perturb_spec(env_setup_kwargs):
    """Creates perturb_spec dictionary."""
    perturb_spec = {
        "enable": False,
        "period": 1,
        "scheduler": "constant",
    }

    if env_setup_kwargs["perturb_param_name"]:
        perturb_spec["param"] = env_setup_kwargs["perturb_param_name"]

    perturb_min = env_setup_kwargs["perturb_param_min"]
    perturb_max = env_setup_kwargs["perturb_param_max"]

    if env_setup_kwargs["perturb_param_value"] is not None:
        perturb_spec["enable"] = True
        perturb_spec["start"] = env_setup_kwargs["perturb_param_value"]
        perturb_spec["min"] = env_setup_kwargs["perturb_param_value"]
        perturb_spec["max"] = env_setup_kwargs["perturb_param_value"]
    elif (perturb_min is not None) and (perturb_max is not None):
        perturb_spec["enable"] = True
        perturb_spec["start"] = (perturb_min + perturb_max) / 2
        perturb_spec["min"] = perturb_min
        perturb_spec["max"] = perturb_max
        perturb_spec["scheduler"] = "uniform"

    return perturb_spec


def create_noise_spec(env_setup_kwargs):
    """Creates noise_spec dictionary."""
    noise_spec = dict()

    action_noise_std = env_setup_kwargs["action_noise_std"]
    observation_noise_std = env_setup_kwargs["observation_noise_std"]
    if (action_noise_std > 0.0) or (observation_noise_std > 0.0):
        noise_spec["gaussian"] = {
            "enable": True,
            "actions": action_noise_std,
            "observations": observation_noise_std,
        }

    return noise_spec


# RWRL Constraints
#########################################

rwrl_constraints_cartpole = {
    "slider_pos_constraint": rwrl.cartpole.slider_pos_constraint,
    "balance_velocity_constraint": rwrl.cartpole.balance_velocity_constraint,
    "slider_accel_constraint": rwrl.cartpole.slider_accel_constraint,
}

rwrl_constraints_walker = {
    "joint_angle_constraint": rwrl.walker.joint_angle_constraint,
    "joint_velocity_constraint": rwrl.walker.joint_velocity_constraint,
    "dangerous_fall_constraint": rwrl.walker.dangerous_fall_constraint,
    "torso_upright_constraint": rwrl.walker.torso_upright_constraint,
}

rwrl_constraints_quadruped = {
    "joint_angle_constraint": rwrl.quadruped.joint_angle_constraint,
    "joint_velocity_constraint": rwrl.quadruped.joint_velocity_constraint,
    "upright_constraint": rwrl.quadruped.upright_constraint,
    "foot_force_constraint": rwrl.quadruped.foot_force_constraint,
}

rwrl_constraints_humanoid = {
    "joint_angle_constraint": rwrl.humanoid.joint_angle_constraint,
    "joint_velocity_constraint": rwrl.humanoid.joint_velocity_constraint,
    "upright_constraint": rwrl.humanoid.upright_constraint,
    "dangerous_fall_constraint": rwrl.humanoid.dangerous_fall_constraint,
    "foot_force_constraint": rwrl.humanoid.foot_force_constraint,
}

rwrl_constraints_manipulator = {
    "joint_angle_constraint": rwrl.manipulator.joint_angle_constraint,
    "joint_velocity_constraint": rwrl.manipulator.joint_velocity_constraint,
    "joint_accel_constraint": rwrl.manipulator.joint_accel_constraint,
    "grasp_force_constraint": rwrl.manipulator.grasp_force_constraint,
}

rwrl_constraints_combined = {
    "cartpole": rwrl_constraints_cartpole,
    "walker": rwrl_constraints_walker,
    "quadruped": rwrl_constraints_quadruped,
    "humanoid": rwrl_constraints_humanoid,
    "manipulator": rwrl_constraints_manipulator,
}
