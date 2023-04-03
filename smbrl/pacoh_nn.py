import copy
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax

from smbrl.models import ParamsDistribution


def meta_train(
    data,
    prediction_fn,
    hyper_prior,
    hyper_posterior,
    optimizer,
    opt_state,
    iterations,
    n_prior_samples,
):
    hyper_posterior = copy.deepcopy(hyper_posterior)
    for i in range(iterations):
        meta_batch_x, meta_batch_y = next(data)
        hyper_posterior, opt_state, log_probs = train_step(
            meta_batch_x,
            meta_batch_y,
            prediction_fn,
            hyper_prior,
            hyper_posterior,
            jax.random.PRNGKey(0),
            n_prior_samples,
            optimizer,
            opt_state,
        )
        if i % 100 == 0:
            print(f"Iteration {i} log probs: {log_probs}")
    return hyper_posterior


@functools.partial(jax.jit, static_argnums=(2, 6, 7))
def train_step(
    meta_batch_x,
    meta_batch_y,
    prediction_fn,
    hyper_prior,
    hyper_posterior,
    key,
    n_prior_samples,
    optimizer,
    opt_state,
):
    grad_fn = jax.value_and_grad(
        lambda hyper_posterior: particle_loss(
            meta_batch_x,
            meta_batch_y,
            prediction_fn,
            hyper_posterior,
            hyper_prior,
            key,
            n_prior_samples,
        )
    )
    # vmap to compute the grads for each particle in the ensemble with respect
    # to its prediction's log probability.
    log_probs, log_prob_grads = jax.vmap(grad_fn)(hyper_posterior)
    # Compute the particles' kernel matrix and its per-particle gradients.
    num_particles = jax.tree_util.tree_flatten(log_prob_grads)[0][0].shape[0]
    particles_matrix, reconstruct_tree = _to_matrix(hyper_posterior, num_particles)
    kxx, kernel_vjp = jax.vjp(
        lambda x: rbf_kernel(x, particles_matrix), particles_matrix
    )
    # Summing along the 'particles axis' to compute the per-particle gradients, see
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    kernel_grads = kernel_vjp(jnp.ones(kxx.shape))[0]
    stein_grads = (
        jnp.matmul(kxx, _to_matrix(log_prob_grads, num_particles)[0]) + kernel_grads
    ) / num_particles
    stein_grads = reconstruct_tree(stein_grads.ravel())
    updates, new_opt_state = optimizer.update(stein_grads, opt_state)
    new_params = optax.apply_updates(hyper_posterior, updates)
    return (ParamsDistribution(*new_params), new_opt_state, log_probs.mean())


def _to_matrix(params, num_particles):
    flattened_params, reconstruct_tree = jax.tree_util.ravel_pytree(params)
    matrix = flattened_params.reshape((num_particles, -1))
    return matrix, reconstruct_tree


def particle_loss(
    meta_batch_x,
    meta_batch_y,
    prediction_fn,
    particle,
    hyper_prior,
    key,
    n_prior_samples,
):
    def estimate_mll(x, y):
        prior_samples = particle.sample(key, n_prior_samples)
        per_sample_pred = jax.vmap(prediction_fn, (0, None))
        y_hat, stddevs = per_sample_pred(prior_samples, x)
        log_likelihood = distrax.MultivariateNormalDiag(y_hat, stddevs).log_prob(y)
        batch_size = x.shape[0]
        mll = jax.scipy.special.logsumexp(
            log_likelihood, axis=0, b=jnp.sqrt(batch_size)
        ) - np.log(n_prior_samples)
        return mll

    # vmap estimate_mll over the task batch dimension, as specified
    # @ Algorithm 1 PACOH with SVGD approximation of Q∗ (MLL_Estimator)
    # @ https://arxiv.org/pdf/2002.05551.pdf.
    mll = jax.vmap(estimate_mll)(meta_batch_x, meta_batch_y)
    log_prob_prior = hyper_prior.log_prob(particle)
    return -(mll + log_prob_prior).mean()


# Based on tf-probability implementation of batched pairwise matrices:
# https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/math/psd_kernels/internal/util.py#L190
def rbf_kernel(x, y, bandwidth=None):
    row_norm_x = (x**2).sum(-1)[..., None]
    row_norm_y = (y**2).sum(-1)[..., None, :]
    pairwise = jnp.clip(row_norm_x + row_norm_y - 2.0 * jnp.matmul(x, y.T), 0.0)
    n_x = pairwise.shape[-2]
    bandwidth = bandwidth if bandwidth is not None else jnp.median(pairwise)
    bandwidth = 0.5 * bandwidth / jnp.log(n_x + 1)
    bandwidth = jnp.maximum(jax.lax.stop_gradient(bandwidth), 1e-5)
    k_xy = jnp.exp(-pairwise / bandwidth / 2)
    return k_xy


@functools.partial(jax.jit, static_argnums=(3, 5))
def infer_posterior(
    x,
    y,
    hyper_posterior,
    prediction_fn,
    key,
    update_steps,
    learning_rate,
):
    # Sample prior parameters from the hyper-posterior to form an ensemble
    # of neural networks.
    posterior_params = hyper_posterior.sample(key, 1)
    posterior_params = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), posterior_params
    )
    optimizer = optax.flatten(optax.adam(learning_rate))
    opt_state = optimizer.init(posterior_params)

    def loss(params):
        y_hat, stddevs = prediction_fn(params, x)
        log_likelihood = distrax.MultivariateNormalDiag(y_hat, stddevs).log_prob(y)
        return -log_likelihood.mean()

    def update(carry, _):
        posterior_params, opt_state = carry
        values, grads = jax.vmap(jax.value_and_grad(loss))(posterior_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        posterior_params = optax.apply_updates(posterior_params, updates)
        return (posterior_params, opt_state), values.mean()

    (posterior_params, _), losses = jax.lax.scan(
        update, (posterior_params, opt_state), None, update_steps
    )
    return posterior_params, losses


@functools.partial(jax.jit, static_argnums=(2))
def predict(
    posterior,
    x,
    prediction_fn,
):
    prediction_fn = jax.vmap(prediction_fn, in_axes=(0, None))
    y_hat, stddev = prediction_fn(posterior, x)
    return y_hat, stddev
