import jax
import jax.numpy as jnp
from optax import l2_loss

from smbrl.types import Data, Learner, Model, OptState, PRNGKey


def to_ins(observation, action):
    return jnp.concatenate([observation, action], -1)


def to_outs(next_state, reward):
    return jnp.concatenate([next_state, reward[..., None]], -1)


def bridge_model(model):
    def fn(x):
        state_dim = model.state_decoder.out_features // 2
        obs, acs = jnp.split(x, [state_dim], axis=-1)
        preds = model(obs, acs)
        y_hat = to_outs(preds.next_state, preds.reward)
        stddev = to_outs(preds.next_state_stddev, preds.reward_stddev)
        return y_hat, stddev

    return fn


def simple_regression(
    data: Data,
    model: Model,
    learner: Learner,
    opt_state: OptState,
    key: PRNGKey,
):
    def update(carry, inputs):
        model, opt_state = carry
        x, y = inputs
        if x.ndim == 4:
            x = x.reshape(-1, *x.shape[2:])
            y = y.reshape(-1, *y.shape[2:])
        # Bridge to make x -> [obs, acs], vmap over the batch dimension.
        loss = lambda model: l2_loss(bridge_model(jax.vmap(model))(x)[0], y).mean()
        loss, model_grads = jax.value_and_grad(loss)(model)
        new_model, new_opt_state = learner.grad_step(model, model_grads, opt_state)
        return (new_model, new_opt_state), loss

    return jax.lax.scan(
        update,
        (model, opt_state),
        data,
    )
