import functools

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from smbrl.models import ParamsDistribution
from smbrl.utils import inv_softplus


def meta_train(
    data,
    hyper_prior,
    hyper_posterior,
    optimizer,
    opt_state,
    iterations,
    n_prior_samples,
    key,
    prior_weight=1e-4,
    bandwidth=10.0,
):
    def _update(carry, inputs):
        hyper_posterior, opt_state = carry
        key = inputs
        key, next_key = jax.random.split(key)
        ids = jax.random.choice(next_key, num_examples - 1)
        meta_batch_x, meta_batch_y = jax.tree_map(lambda x: x[ids], data)
        hyper_posterior, opt_state, log_probs = train_step(
            meta_batch_x,
            meta_batch_y,
            hyper_prior,
            hyper_posterior,
            key,
            n_prior_samples,
            optimizer,
            opt_state,
            prior_weight,
            bandwidth,
        )
        return (hyper_posterior, opt_state), log_probs

    num_examples = data[0].shape[0]
    return jax.lax.scan(
        _update,
        (hyper_posterior, opt_state),
        jnp.asarray(jax.random.split(key, iterations)),
    )


@functools.partial(jax.jit, static_argnums=(5, 6))
def train_step(
    meta_batch_x,
    meta_batch_y,
    hyper_prior,
    hyper_posterior,
    key,
    n_prior_samples,
    optimizer,
    opt_state,
    prior_weight,
    bandwidth,
):
    grad_fn = jax.value_and_grad(
        lambda hyper_posterior: particle_likelihood(
            meta_batch_x,
            meta_batch_y,
            hyper_posterior,
            hyper_prior,
            key,
            n_prior_samples,
            prior_weight,
        )
    )
    # vmap to compute the grads for each particle in the ensemble with respect
    # to its prediction's log probability.
    log_probs, log_prob_grads = jax.vmap(grad_fn)(hyper_posterior)
    # Compute the particles' kernel matrix and its per-particle gradients.
    num_particles = jax.tree_util.tree_flatten(log_prob_grads)[0][0].shape[0]
    particles_matrix, reconstruct_tree = _to_matrix(hyper_posterior, num_particles)
    kxx, kernel_vjp = jax.vjp(
        lambda x: rbf_kernel(x, particles_matrix, bandwidth), particles_matrix
    )
    # Summing along the 'particles axis' to compute the per-particle gradients, see
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    kernel_grads = kernel_vjp(-jnp.ones(kxx.shape))[0]
    stein_grads = (
        -(kxx @ _to_matrix(log_prob_grads, num_particles)[0] + kernel_grads)
        / num_particles
    )
    stein_grads = reconstruct_tree(stein_grads.ravel())
    updates, new_opt_state = optimizer.update(stein_grads, opt_state)
    new_params = optax.apply_updates(hyper_posterior, updates)
    return (ParamsDistribution(*new_params), new_opt_state, log_probs.mean())


def _to_matrix(params, num_particles):
    flattened_params, reconstruct_tree = jax.flatten_util.ravel_pytree(params)
    matrix = flattened_params.reshape((num_particles, -1))
    return matrix, reconstruct_tree


def particle_likelihood(
    meta_batch_x,
    meta_batch_y,
    prior,
    hyper_prior,
    key,
    n_prior_samples,
    prior_weight,
):
    def estimate_mll(x, y):
        prior_samples = prior.sample(key, n_prior_samples)
        y_hat, stddevs = predict(prior_samples, x)
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
    log_prob_prior = hyper_prior.log_prob(prior) * prior_weight
    return (mll + log_prob_prior).mean()


# Based on tf-probability implementation of batched pairwise matrices:
# https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/math/psd_kernels/internal/util.py#L190
def rbf_kernel(x, y, bandwidth=None):
    row_norm_x = (x**2).sum(-1)[..., None]
    row_norm_y = (y**2).sum(-1)[..., None, :]
    pairwise = jnp.clip(row_norm_x + row_norm_y - 2.0 * jnp.matmul(x, y.T), 0.0)
    bandwidth = bandwidth if bandwidth is not None else jnp.median(pairwise)
    n_x = pairwise.shape[-2]
    bandwidth = 0.5 * bandwidth / jnp.log(n_x + 1)
    bandwidth = jnp.maximum(jax.lax.stop_gradient(bandwidth), 1e-5)
    k_xy = jnp.exp(-pairwise / bandwidth)
    return k_xy


def infer_posterior(
    x,
    y,
    prior,
    update_steps,
    learning_rate,
):
    # Sample prior parameters from the hyper-posterior to form an ensemble
    # of neural networks.
    optimizer = optax.flatten(optax.adam(learning_rate))
    opt_state = optimizer.init(prior)

    def loss(model):
        y_hat, stddevs = predict(model, x)
        log_likelihood = distrax.MultivariateNormalDiag(y_hat, stddevs).log_prob(y)
        return -log_likelihood.mean()

    def update(carry, _):
        model, opt_state = carry
        values, grads = jax.value_and_grad(loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = optax.apply_updates(model, updates)
        return (model, opt_state), values.mean()

    (posterior, _), losses = jax.lax.scan(
        update, (prior, opt_state), None, update_steps
    )
    return posterior, losses


def predict(model, x):
    # First vmap along the batch dimension.
    ensemble_predict = lambda model, x: jax.vmap(model)(x)
    # then vmap over members of the ensemble.
    ensemble_predict = eqx.filter_vmap(
        ensemble_predict, in_axes=(eqx.if_array(0), None)
    )
    y_hat, stddev = ensemble_predict(model, x)
    return y_hat, stddev


def make_hyper_prior(model):
    mean_prior_over_mus = jax.tree_map(jnp.zeros_like, model)
    stddev_prior_over_mus = jax.tree_map(lambda x: jnp.ones_like(x) * 1.0, model)
    mean_prior_over_stddevs = jax.tree_map(lambda x: jnp.ones_like(x) * 10.0, model)
    stddev_prior_over_stddevs = jax.tree_map(lambda x: jnp.ones_like(x) * 10.0, model)
    # Create a distribution over a Mean Field distribution of NNs.
    hyper_prior = ParamsDistribution(
        ParamsDistribution(mean_prior_over_mus, stddev_prior_over_mus),
        ParamsDistribution(mean_prior_over_stddevs, stddev_prior_over_stddevs),
    )
    return hyper_prior


def make_hyper_posterior(make_model, key, n_particles, stddev=1e-7):
    stddev_scale = inv_softplus(stddev)
    init_ensemble = jax.vmap(make_model)
    particles_mus = init_ensemble(jnp.asarray(jax.random.split(key, n_particles)))
    particles_mus = jax.tree_map(lambda x: jnp.zeros_like(x), particles_mus)
    particle_stddevs = jax.tree_map(
        lambda x: jnp.ones_like(x) * stddev_scale, particles_mus
    )
    # Create a distribution over ensembles of NNs.
    hyper_posterior = ParamsDistribution(particles_mus, particle_stddevs)
    return hyper_posterior
