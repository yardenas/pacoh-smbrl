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


@functools.partial(jax.jit, static_argnums=(2, 5, 6))
def train_step(
    meta_batch_x,
    meta_batch_y,
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
    particle,
    hyper_prior,
    key,
    n_prior_samples,
):
    def estimate_mll(x, y):
        prior_samples = particle.sample(key, n_prior_samples)
        per_sample_pred = jax.vmap(prior_samples, (0, None))
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
    key,
    update_steps,
    learning_rate,
):
    # Sample prior parameters from the hyper-posterior to form an ensemble
    # of neural networks.
    prior = hyper_posterior.sample(key, 1)
    prior = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), prior)
    optimizer = optax.flatten(optax.adam(learning_rate))
    opt_state = optimizer.init(prior)

    def loss(model):
        y_hat, stddevs = model(x)
        log_likelihood = distrax.MultivariateNormalDiag(y_hat, stddevs).log_prob(y)
        return -log_likelihood.mean()

    def update(carry, _):
        model, opt_state = carry
        values, grads = jax.vmap(jax.value_and_grad(loss))(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = optax.apply_updates(model, updates)
        return (model, opt_state), values.mean()

    (prior, _), losses = jax.lax.scan(update, (prior, opt_state), None, update_steps)
    return prior, losses


@functools.partial(jax.jit, static_argnums=(2))
def predict(posterior, x):
    prediction_fn = jax.vmap(posterior, in_axes=(0, None))
    y_hat, stddev = prediction_fn(x)
    return y_hat, stddev


def make_hyper_prior(model):
    mean_prior_over_mus = jax.tree_map(jnp.zeros_like, model)
    mean_prior_over_stddevs = jax.tree_map(jnp.zeros_like, mean_prior_over_mus)
    hyper_prior = ParamsDistribution(
        ParamsDistribution(mean_prior_over_mus, mean_prior_over_stddevs),
        1.0,
    )
    return hyper_prior


def make_hyper_posterior(make_model, key, n_particles):
    init_ensemble = jax.vmap(make_model)
    particles_mus = init_ensemble(jnp.asarray(jax.random.split(key, n_particles)))
    particle_stddevs = jax.tree_map(lambda x: jnp.ones_like(x) * 1e-4, particles_mus)
    hyper_posterior = ParamsDistribution(particles_mus, particle_stddevs)
    return hyper_posterior
