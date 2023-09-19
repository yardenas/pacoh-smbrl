import numpy as np

from smbrl.agents.actor_critic import ModelBasedActorCritic
from smbrl.agents.augmented_lagrangian import AugmentedLagrangianPenalizer
from smbrl.agents.contextual_actor_critic import ContextualModelBasedActorCritic
from smbrl.agents.lbsgd import LBSGDPenalizer
from smbrl.agents.safe_actor_critic import SafeModelBasedActorCritic
from smbrl.agents.safe_contextual_actor_critic import (
    SafeContextualModelBasedActorCritic,
)


def make_penalizer(cfg):
    if cfg.agent.penalizer.name == "lbsgd":
        return LBSGDPenalizer(
            cfg.agent.penalizer.m_0,
            cfg.agent.penalizer.m_1,
            cfg.agent.penalizer.eta,
            cfg.agent.penalizer.eta_rate,
        )
    elif cfg.agent.penalizer.name == "lagrangian":
        return AugmentedLagrangianPenalizer(
            cfg.agent.penalizer.initial_lagrangian,
            cfg.agent.penalizer.initial_multiplier,
            cfg.agent.penalizer.multiplier_factor,
        )
    else:
        raise NotImplementedError


def make_actor_critic(safe, contextual, state_dim, action_dim, cfg, key, belief=None):
    if contextual:
        assert belief is not None
    if safe:
        # Account for the the discount factor in the budget.
        episode_safety_budget = (
            (
                (cfg.training.safety_budget / cfg.training.time_limit)
                / (1.0 - cfg.agent.safety_discount)
            )
            if cfg.agent.safety_discount < 1.0 - np.finfo(np.float32).eps
            else cfg.training.safety_budget
        ) + cfg.agent.safety_slack
        penalizer = make_penalizer(cfg)
        common_ins = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_config=cfg.agent.actor,
            critic_config=cfg.agent.critic,
            actor_optimizer_config=cfg.agent.actor_optimizer,
            critic_optimizer_config=cfg.agent.critic_optimizer,
            safety_critic_optimizer_config=cfg.agent.safety_critic_optimizer,
            horizon=cfg.agent.plan_horizon,
            discount=cfg.agent.discount,
            safety_discount=cfg.agent.safety_discount,
            lambda_=cfg.agent.lambda_,
            safety_budget=episode_safety_budget,
            penalizer=penalizer,
            key=key,
        )
    else:
        common_ins = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_config=cfg.agent.actor,
            critic_config=cfg.agent.critic,
            actor_optimizer_config=cfg.agent.actor_optimizer,
            critic_optimizer_config=cfg.agent.critic_optimizer,
            horizon=cfg.agent.plan_horizon,
            discount=cfg.agent.discount,
            lambda_=cfg.agent.lambda_,
            key=key,
        )
    if safe and contextual:
        return SafeContextualModelBasedActorCritic(
            **common_ins,
            belief=belief,
        )
    elif not safe and contextual:
        return ContextualModelBasedActorCritic(**common_ins, belief=belief)
    elif safe and not contextual:
        return SafeModelBasedActorCritic(**common_ins)
    else:
        return ModelBasedActorCritic(**common_ins)
