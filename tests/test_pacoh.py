from typing import Iterator, Tuple

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax

import smbrl.pacoh_nn as pacoh
from smbrl.utils import inv_softplus


class SinusoidRegression:
    def __init__(
        self,
        meta_batch_size: int,
        num_train_shots: int,
        num_test_shots: int,
        seed: int = 666,
    ):
        self.meta_batch_size = meta_batch_size
        self.num_train_shots = num_train_shots
        self.num_test_shots = num_test_shots
        self.rs = np.random.RandomState(seed)

    @property
    def train_set(
        self,
    ):
        while True:
            yield self._make_batch(self.num_train_shots)[0]

    @property
    def test_set(
        self,
    ) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        while True:
            yield self._make_batch(self.num_test_shots)

    def _make_batch(self, num_shots):
        # Select amplitude and phase for the task
        amplitudes = []
        phases = []
        for _ in range(self.meta_batch_size):
            amplitudes.append(self.rs.uniform(low=0.75, high=1.0))
            phases.append(self.rs.uniform(low=-2.0, high=2.0))

        def get_batch(num_shots):
            xs, ys = [], []
            for amplitude, phase in zip(amplitudes, phases):
                if num_shots > 0:
                    x = self.rs.uniform(low=-4.0, high=4.0, size=(num_shots, 1))
                else:
                    x = np.linspace(-4.0, 4.0, 1000)[:, None]
                y = amplitude * np.sin(3.0 * x + phase)
                xs.append(x)
                ys.append(y)
            return np.stack(xs), np.stack(ys)

        (x1, y1), (x2, y2) = get_batch(num_shots), get_batch(-1)
        return (x1, y1), (x2, y2)


class Homoscedastic(eqx.Module):
    layers: list[eqx.nn.Linear]
    mu: eqx.nn.Linear
    stddev: jax.Array

    def __init__(self, in_size, hidden_size, out_size, key) -> None:
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(in_size, hidden_size, key=keys[0]),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[2]),
        ]
        self.mu = eqx.nn.Linear(hidden_size, out_size, key=keys[3])
        self.stddev = jnp.ones((out_size,)) * 1e-5

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = jnn.relu(layer(x))
        return self.mu(x), _clip_stddev(self.stddev, 0.1, 0.1, 0.1)


def _clip_stddev(stddev, stddev_min, stddev_max, stddev_scale):
    stddev = jnp.clip(
        (stddev + inv_softplus(0.1)) * stddev_scale,
        inv_softplus(stddev_min),
        inv_softplus(stddev_max),
    )
    return jnn.softplus(stddev)


def sample_data(data_generating_process, num_tasks):
    out = []
    for _ in range(num_tasks):
        out.append(next(data_generating_process.train_set))
    return tuple(map(np.stack, zip(*out)))


def test_training():
    data_generating_process = SinusoidRegression(16, 5, 5)
    train_dataset = sample_data(data_generating_process, 200)
    make_model = lambda key: Homoscedastic(1, 32, 1, key)
    key = jax.random.PRNGKey(0)
    hyper_prior = pacoh.make_hyper_prior(make_model(key))
    hyper_posterior = pacoh.make_hyper_posterior(make_model, key, 10, 1e-4)
    opt = optax.flatten(optax.adam(5e-4))
    opt_state = opt.init(hyper_posterior)
    for _ in range(2):
        key, key_next = jax.random.split(key)
        (hyper_posterior, opt_state), _ = eqx.filter_jit(pacoh.meta_train)(
            train_dataset,
            hyper_prior,
            hyper_posterior,
            opt,
            opt_state,
            5000,
            10,
            key_next,
        )
        key, key_next = jax.random.split(key)
        priors = hyper_posterior.sample(key_next, 10)
        priors = jax.tree_map(lambda x: x.mean(0), priors)
        plot_prior(*next(data_generating_process.train_set), priors)
    (context_x, context_y), (test_x, test_y) = next(data_generating_process.test_set)
    key, key_next = jax.random.split(key)
    infer_posteriors = lambda x, y: pacoh.infer_posterior(
        x, y, hyper_posterior, 5000, 3e-4, 10, key_next, 1e-8
    )
    infer_posteriors = jax.jit(jax.vmap(infer_posteriors))
    posteriors, losses = infer_posteriors(context_x, context_y)
    predict = jax.vmap(pacoh.predict)
    predictions = predict(posteriors, test_x)
    plot(context_x, context_y, test_x, test_y, predictions)


def plot_prior(x, y, priors):
    import matplotlib.pyplot as plt

    x_pred = np.linspace(-4.0, 4.0, 1000)[:, None]
    predictions = pacoh.predict(priors, x_pred)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    mu, _ = predictions
    _, (ax) = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(x, y, label="observed", s=5, alpha=0.4)
    ax.plot(
        np.tile(x_pred, (mu.shape[0], 1, 1)).squeeze(-1).T,
        mu.squeeze(-1).T,
        label="ensemble means",
        color="green",
        alpha=0.3,
        linewidth=1.0,
    )
    plt.show()


def plot(x, y, x_tst, y_tst, yhats):
    import matplotlib.pyplot as plt

    mus, stddevs = yhats
    plt.figure()
    for task in range(6):
        plt.subplot(2, 3, task + 1)
        avgm = np.zeros_like(x_tst[task, :, 0])
        for i, (mu, stddev) in enumerate(zip(mus[task], stddevs[task])):
            m = np.squeeze(mu)
            s = np.squeeze(stddev)
            if i < 15:
                plt.plot(
                    x_tst[task],
                    m,
                    "r",
                    label="ensemble means" if i == 0 else None,
                    linewidth=1.0,
                )
                plt.plot(
                    x_tst[task],
                    m + 3 * s,
                    "g",
                    linewidth=0.5,
                    label="ensemble means + 3 ensemble stdev" if i == 0 else None,
                )
                plt.plot(
                    x_tst[task],
                    m - 3 * s,
                    "g",
                    linewidth=0.5,
                    label="ensemble means - 3 ensemble stdev" if i == 0 else None,
                )
            avgm += m
        avgm = avgm / (i + 1)
        epistemic = mus[task].std(0).squeeze(-1)
        plt.plot(x_tst[task], avgm, "r", label="overall mean", linewidth=4)
        plt.fill_between(
            x_tst[task].squeeze(1),
            avgm - 3 * epistemic,
            avgm + 3 * epistemic,
            alpha=0.2,
        )
        plt.plot(x_tst[task], y_tst[task], "b", label="overall mean", linewidth=1)
        plt.scatter(x[task], y[task], c="b", label="observed")
        ax = plt.gca()
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.show()
