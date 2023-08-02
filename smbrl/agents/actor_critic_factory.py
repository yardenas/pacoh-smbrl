from smbrl.agents.actor_critic import ModelBasedActorCritic
from smbrl.agents.contextual_actor_critic import ContextualModelBasedActorCritic
from smbrl.agents.safe_actor_critic import SafeModelBasedActorCritic
from smbrl.agents.safe_contextual_actor_critic import (
    SafeContextualModelBasedActorCritic,
)


def make_actor_critic(safe, contextual, state_dim, action_dim, cfg, key, belief=None):
    if contextual:
        assert belief is not None
    if safe:
        common_ins = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_config=cfg.agent.actor,
            critic_config=cfg.agent.critic,
            actor_optimizer_config=cfg.agent.actor_optimizer,
            critic_optimizer_config=cfg.agent.critic_optimizer,
            horizon=cfg.agent.plan_horizon,
            discount=cfg.agent.discount,
            safety_discount=cfg.agent.safety_discount,
            lambda_=cfg.agent.lambda_,
            safety_budget=cfg.training.safety_budget,
            eta=cfg.agent.eta,
            m_0=cfg.agent.m_0,
            m_1=cfg.agent.m_1,
            eta_rate=cfg.agent.eta_rate,
            base_lr=cfg.agent.base_lr,
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
