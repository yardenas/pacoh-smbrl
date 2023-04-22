import jax
import numpy as np
from optax import OptState, l2_loss

from smbrl import types
from smbrl.agents import models as m
from smbrl.agents import pacoh_nn as pch
from smbrl.trajectory import TrajectoryData
from smbrl.utils import Learner


def prepare_data(replay_buffer, num_steps):
    batches = [batch for batch in replay_buffer.sample(num_steps)]
    # What happens below:
    # 1. Transpose list of (named-)tuples into a named tuple of lists
    # 2. Stack the lists for each data type inside
    # the named tuple (e.g., observations, actions, etc.)
    transposed_batches = TrajectoryData(*map(np.stack, zip(*batches)))
    x = m.to_ins(transposed_batches.observation, transposed_batches.action)
    y = m.to_outs(transposed_batches.next_observation, transposed_batches.reward)
    return x, y


def simple_regression(
    data: types.Data,
    model: m.Model,
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
        loss = lambda model: l2_loss(jax.vmap(model)(x)[0], y).mean()
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
    hyper_prior: m.ParamsDistribution,
    hyper_posterior: m.ParamsDistribution,
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
