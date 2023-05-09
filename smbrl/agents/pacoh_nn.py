import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from smbrl.agents.models import ParamsDistribution
from smbrl.utils import all_finite, ensemble_predict, update_if


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
        if num_examples > 1:
            ids = jax.random.choice(next_key, num_examples)
            meta_batch_x, meta_batch_y = jax.tree_map(lambda x: x[ids], data)
        else:
            meta_batch_x, meta_batch_y = jax.tree_map(lambda x: x[0], data)
        # vmap to compute the grads for each svgd particle.
        mll_fn = jax.vmap(
            jax.value_and_grad(
                lambda hyper_posterior_particle, key: meta_mll(
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
            hyper_posterior, mll_fn, bandwidth, optimizer, opt_state, key=key
        )
        return (hyper_posterior, opt_state), log_probs

    num_examples = data[0].shape[0]
    if num_examples == 1:
        return _update((hyper_posterior, opt_state), key)
    return jax.lax.scan(
        _update,
        (hyper_posterior, opt_state),
        jnp.asarray(jax.random.split(key, iterations)),
    )


def svgd(model, mll_fn, bandwidth, optimizer, opt_state, *, key=None):
    num_particles = jax.tree_util.tree_flatten(model)[0][0].shape[0]
    if key is not None:
        log_probs, log_prob_grads = mll_fn(
            model, key=jax.random.split(key, num_particles)
        )
    else:
        log_probs, log_prob_grads = mll_fn(model)
    # Compute the particles' kernel matrix and its per-particle gradients.
    particles_matrix, reconstruct_tree = _to_matrix(model, num_particles)
    kxx, kernel_vjp = jax.vjp(
        lambda x: rbf_kernel(x, particles_matrix, bandwidth), particles_matrix  # type: ignore # noqa E501
    )
    # Summing along the 'particles axis' to compute the per-particle gradients, see
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    kernel_grads = kernel_vjp(-jnp.ones(kxx.shape))[0]
    score = _to_matrix(log_prob_grads, num_particles)[0]
    stein_grads = -(kxx @ score + kernel_grads) / num_particles
    stein_grads = reconstruct_tree(stein_grads.ravel())
    updates, new_opt_state = optimizer.update(stein_grads, opt_state)
    all_ok = all_finite(stein_grads)
    updates = update_if(
        all_ok, updates, jax.tree_map(lambda x: jnp.zeros_like(x), updates)
    )
    new_opt_state = update_if(all_ok, new_opt_state, opt_state)
    new_model = optax.apply_updates(model, updates)
    return new_model, new_opt_state, log_probs.mean()


def _to_matrix(params, num_particles):
    flattened_params, reconstruct_tree = jax.flatten_util.ravel_pytree(params)  # type: ignore # noqa E501
    matrix = flattened_params.reshape((num_particles, -1))
    return matrix, reconstruct_tree


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
        prior_samples = jax.vmap(hyper_posterior_particle.sample)(
            jnp.asarray(jax.random.split(key, n_prior_samples))
        )
        y_hat, stddevs = ensemble_predict(prior_samples)(x)
        log_likelihood = distrax.MultivariateNormalDiag(y_hat, stddevs).log_prob(
            y[None]
        )
        batch_size = x.shape[0]
        mll = jax.scipy.special.logsumexp(
            log_likelihood, axis=0, b=jnp.sqrt(batch_size)
        )
        return mll

    # vmap estimate_mll over the task batch dimension, as specified
    # @ Algorithm 1 PACOH with SVGD approximation of Q∗ (MLL_Estimator)
    # @ https://arxiv.org/pdf/2002.05551.pdf.
    mll = jax.vmap(estimate_mll)(meta_batch_x, meta_batch_y)
    log_prob_prior = (
        hyper_prior_particle.log_prob(hyper_posterior_particle) * prior_weight
    )
    return (mll + log_prob_prior).mean()


def mll(batch_x, batch_y, model, prior, prior_weight):
    y_hat, stddevs = jax.vmap(model)(batch_x)
    log_likelihood = (
        distrax.MultivariateNormalDiag(y_hat, stddevs).log_prob(batch_y).mean()
    )
    log_prior = prior.log_prob(model)
    return log_likelihood + log_prior * prior_weight


def infer_posterior(
    data,
    prior,
    posterior,
    iterations,
    learning_rate,
    key,
    prior_weight=1e-7,
    bandwidth=10.0,
):
    optimizer = optax.flatten(optax.adam(learning_rate))
    opt_state = optimizer.init(posterior)
    num_examples = data[0].shape[0]

    def update(carry, inputs):
        model, opt_state = carry
        key = inputs
        key, next_key = jax.random.split(key)
        ids = jax.random.choice(next_key, num_examples)
        x, y = jax.tree_map(lambda x: x[ids], data)
        mll_grads = jax.value_and_grad(
            lambda model: mll(x, y, model, prior, prior_weight)
        )
        # vmap mll over svgd particles.
        mll_fn = jax.vmap(mll_grads)
        model, opt_state, mll_value = svgd(
            model, mll_fn, bandwidth, optimizer, opt_state
        )
        return (model, opt_state), mll_value

    (posterior, _), mll_values = jax.lax.scan(
        update, (posterior, opt_state), jnp.asarray(jax.random.split(key, iterations))
    )
    return posterior, mll_values


def sample_prior_models(hyper_posterior, key, n_prior_samples):
    # Sample model instances from the hyper-posterior to form an ensemble
    # of neural networks.
    model = jax.vmap(hyper_posterior.sample)(jax.random.split(key, n_prior_samples))
    model = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), model)
    return model


def compute_prior(hyper_posterior):
    # Marginalize over samples of each particle in the hyper-posterior.
    prior = jax.tree_map(lambda x: x.mean(0), hyper_posterior)
    return prior


def make_hyper_prior(model):
    mean_prior_over_mus = jax.tree_map(jnp.zeros_like, model)
    stddev_prior_over_mus = jax.tree_map(lambda x: jnp.ones_like(x) * 0.5, model)
    mean_prior_over_stddevs = jax.tree_map(lambda x: jnp.ones_like(x) * -3.0, model)
    stddev_prior_over_stddevs = jax.tree_map(lambda x: jnp.ones_like(x) * 0.5, model)
    # Create a distribution over a Mean Field distribution of NNs.
    hyper_prior = ParamsDistribution(
        ParamsDistribution(mean_prior_over_mus, stddev_prior_over_mus),
        ParamsDistribution(mean_prior_over_stddevs, stddev_prior_over_stddevs),
    )
    return hyper_prior


def make_hyper_posterior(make_model, key, n_particles, stddev=0.1):
    particle_fn = lambda key: make_hyper_posterior_particle(make_model, key, stddev)
    return jax.vmap(particle_fn)(jnp.asarray(jax.random.split(key, n_particles)))


def make_hyper_posterior_particle(make_model, key, stddev=1e-7):
    stddev_scale = jnp.log(stddev)
    particle_mus = make_model(key)
    mus_empirical_stddev = jax.flatten_util.ravel_pytree(particle_mus)[0].std()  # type: ignore # noqa E501
    particle_stddevs = jax.tree_map(
        lambda x: jnp.ones_like(x) * jnp.log(mus_empirical_stddev * stddev + 1e-8),
        particle_mus,
    )
    particle_mus = _init_bias(particle_mus, lambda b: jnp.zeros_like(b))
    particle_stddevs = _init_bias(
        particle_stddevs, lambda b: jnp.ones_like(b) * stddev_scale
    )
    hyper_posterior_particle = ParamsDistribution(particle_mus, particle_stddevs)
    return hyper_posterior_particle


def _init_bias(model, init_fn):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_biases = lambda m: [
        x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)
    ]
    new_biases = [init_fn(b) for b in get_biases(model)]
    new_model = eqx.tree_at(get_biases, model, new_biases)
    return new_model
