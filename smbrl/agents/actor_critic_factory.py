from smbrl.agents.actor_critic import ModelBasedActorCritic
from smbrl.agents.contextual_actor_critic import ContextualModelBasedActorCritic
from smbrl.agents.safe_actor_critic import SafeModelBasedActorCritic
from smbrl.agents.safe_contextual_actor_critic import (
    SafeContextualModelBasedActorCritic,
)


def make_actor_critic(safe, contextual):
    if safe and contextual:
        return SafeContextualModelBasedActorCritic
    elif not safe and contextual:
        return ContextualModelBasedActorCritic
    elif safe and not contextual:
        return SafeModelBasedActorCritic
    else:
        return ModelBasedActorCritic
