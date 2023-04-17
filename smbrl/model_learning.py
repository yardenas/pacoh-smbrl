import jax
import jax.numpy as jnp
import numpy as np
from optax import OptState, l2_loss

from smbrl import pacoh_nn as pch
from smbrl import types
from smbrl.models import Model, ParamsDistribution
from smbrl.trajectory import TrajectoryData
from smbrl.utils import Learner


def to_ins(observation, action):
    return jnp.concatenate([observation, action], -1)


def to_outs(next_state, reward):
    return jnp.concatenate([next_state, reward[..., None]], -1)


def prepare_data(replay_buffer, num_steps):
    batches = [batch for batch in replay_buffer.sample(num_steps)]
    # What happens below:
    # 1. Transpose list of (named-)tuples into a named tuple of lists
    # 2. Stack the lists for each data type inside
    # the named tuple (e.g., observations, actions, etc.)
    transposed_batches = TrajectoryData(*map(np.stack, zip(*batches)))
    x = to_ins(transposed_batches.observation, transposed_batches.action)
    y = to_outs(transposed_batches.next_observation, transposed_batches.reward)
    return x, y


def bridge_model(model):
    def fn(x):
        state_dim = model.state_decoder.out_features // 2
        obs, acs = jnp.split(x, [state_dim], axis=-1)  # type: ignore
        preds = model(obs, acs)
        y_hat = to_outs(preds.next_state, preds.reward)
        stddev = to_outs(preds.next_state_stddev, preds.reward_stddev)
        return y_hat, stddev

    return fn


def simple_regression(
    data: types.Data,
    model: Model,
    learner: Learner,
    opt_state: OptState,
    key: jax.random.KeyArray,
) -> types.ModelUpdate:
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


def pacoh_regression(
    data: types.Data,
    hyper_prior: ParamsDistribution,
    hyper_posterior: ParamsDistribution,
    learner: Learner,
    opt_state: OptState,
    n_prior_samples: int,
    prior_weight: float,
    bandwidth: float,
    key: jax.random.KeyArray,
) -> types.ModelUpdate:
    iters = data[0].shape[0]
    (hyper_posterior, opt_state), logprobs = pch.meta_train(
        data,
        hyper_prior,
        hyper_posterior,
        learner.optimizer,
        opt_state,
        iters,
        n_prior_samples,
        key,
        prior_weight,
        bandwidth,
    )
    return (hyper_posterior, opt_state), logprobs
