from typing import NamedTuple

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


class Prediction(NamedTuple):
    next_state: jax.Array
    reward: jax.Array


class Model(eqx.Module):
    layers: list[eqx.nn.Linear]
    encoder: eqx.nn.Linear
    state_decoder: eqx.nn.Linear
    reward_decoder: eqx.nn.Linear

    def __init__(self, n_layers, state_dim, action_dim, hidden_size, *, key):
        keys = jax.random.split(key, 3 + n_layers)
        self.layers = [
            eqx.nn.Linear(hidden_size, hidden_size, key=k) for k in keys[:n_layers]
        ]
        self.encoder = eqx.nn.Linear(state_dim + action_dim, hidden_size, key=keys[1])
        self.state_decoder = eqx.nn.Linear(hidden_size, state_dim, key=keys[2])
        self.reward_decoder = eqx.nn.Linear(hidden_size, 1, key=keys[3])

    def __call__(self, state_sequence, action_sequence):
        x = jax.vmap(self.encoder)(
            jnp.concatenate([state_sequence, action_sequence], -1)
        )
        for layer in self.layers:
            x = jnn.relu(jax.vmap(layer)(x))
        next_state = jax.vmap(self.state_decoder)(x)
        reward = jax.vmap(self.reward_decoder)(x)
        return Prediction(next_state, reward)

    def sample(
        self,
        horizon,
        initial_state,
        key,
        action_sequence=None,
        policy=None,
    ):
        def f(carry, x):
            prev_state = carry
            action, key = x
            if action is None:
                assert policy is not None
                action = policy(prev_state).sample(key)
            out = self(
                prev_state[None],
                action[None],
            )
            return out.next_state[0], out

        if action_sequence is None:
            assert action_sequence is not None
            action_sequence = [None] * horizon
        else:
            assert len(action_sequence) == horizon
        init = initial_state
        inputs = (action_sequence, jax.random.split(key, horizon))
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        out = jax.tree_map(lambda x: x.squeeze(1), out)
        return out
