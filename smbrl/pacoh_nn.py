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
        # vmap to compute the grads for each svgd particle.
        mll_fn = jax.vmap(
            jax.value_and_grad(
                lambda hyper_posterior_particle: meta_mll(
                    meta_batch_x,
                    meta_batch_y,
                    hyper_posterior_particle,
                    hyper_prior,
                    key,
                    n_prior_samples,
                    prior_weight,
                )
            )
        )
        hyper_posterior, opt_state, log_probs = svgd(
            hyper_posterior, mll_fn, bandwidth, optimizer, opt_state
        )
        return (hyper_posterior, opt_state), log_probs

    num_examples = data[0].shape[0]
    return jax.lax.scan(
        _update,
        (hyper_posterior, opt_state),
        jnp.asarray(jax.random.split(key, iterations)),
    )


def svgd(model, mll_fn, bandwidth, optimizer, opt_state):
    log_probs, log_prob_grads = mll_fn(model)
    # Compute the particles' kernel matrix and its per-particle gradients.
    num_particles = jax.tree_util.tree_flatten(log_prob_grads)[0][0].shape[0]
    particles_matrix, reconstruct_tree = _to_matrix(model, num_particles)
    kxx, kernel_vjp = jax.vjp(
        lambda x: rbf_kernel(x, particles_matrix, bandwidth), particles_matrix
    )
    # Summing along the 'particles axis' to compute the per-particle gradients, see
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    kernel_grads = kernel_vjp(-jnp.ones(kxx.shape))[0]
    score = _to_matrix(log_prob_grads, num_particles)[0]
    stein_grads = -(kxx @ score + kernel_grads) / num_particles
    stein_grads = reconstruct_tree(stein_grads.ravel())
    updates, new_opt_state = optimizer.update(stein_grads, opt_state)
    new_model = optax.apply_updates(model, updates)
    return new_model, new_opt_state, log_probs.mean()


def _to_matrix(params, num_particles):
    flattened_params, reconstruct_tree = jax.flatten_util.ravel_pytree(params)
    matrix = flattened_params.reshape((num_particles, -1))
    return matrix, reconstruct_tree


def meta_mll(
    meta_batch_x,
    meta_batch_y,
    hyper_posterior_particle,
    hyper_prior_particle,
    key,
    n_prior_samples,
    prior_weight,
):
    def estimate_mll(x, y):
        prior_samples = hyper_posterior_particle.sample(key, n_prior_samples)
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
    log_prob_prior = (
        hyper_prior_particle.log_prob(hyper_posterior_particle) * prior_weight
    )
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


def mll(batch_x, batch_y, model, prior, prior_weight):
    y_hat, stddevs = jax.vmap(model)(batch_x)
    log_likelihood = (
        distrax.MultivariateNormalDiag(y_hat, stddevs).log_prob(batch_y).mean()
    )
    log_prior = prior.log_prob(model)
    return log_likelihood + log_prior * prior_weight


def infer_posterior(
    x,
    y,
    hyper_posterior,
    update_steps,
    learning_rate,
    n_prior_samples,
    key,
    prior_weight=1e-7,
    bandwidth=10.0,
):
    # Sample prior parameters from the hyper-posterior to form an ensemble
    # of neural networks.
    priors = hyper_posterior.sample(key, n_prior_samples)
    # Marginalize over samples of each particle in the hyper-posterior.
    prior = jax.tree_map(lambda x: x.mean(0), priors)
    optimizer = optax.flatten(optax.adam(learning_rate))
    opt_state = optimizer.init(prior)
    mll_grads = jax.value_and_grad(
        lambda model, prior: mll(x, y, model, prior, prior_weight)
    )
    # vmap mll over svgd particles.
    mll_fn = jax.vmap(mll_grads)
    partial_mll_fn = lambda model: mll_fn(model, hyper_posterior)

    def update(carry, _):
        model, opt_state = carry
        model, opt_state, mll_value = svgd(
            model, partial_mll_fn, bandwidth, optimizer, opt_state
        )
        return (model, opt_state), mll_value

    (posterior, _), mll_values = jax.lax.scan(
        update, (prior, opt_state), None, update_steps
    )
    return posterior, mll_values


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
