from typing import Iterator, Tuple

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax

import smbrl.pacoh_nn as pacoh


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
            amplitudes.append(self.rs.uniform(low=0.1, high=0.75))
            phases.append(self.rs.uniform(low=-np.pi, high=np.pi))

        def get_batch(num_shots):
            xs, ys = [], []
            for amplitude, phase in zip(amplitudes, phases):
                if num_shots > 0:
                    x = self.rs.uniform(low=-5.0, high=5.0, size=(num_shots, 1))
                else:
                    x = np.linspace(-5.0, 5.0, 1000)[:, None]
                y = amplitude * np.sin(x + phase)
                xs.append(x)
                ys.append(y)
            return np.stack(xs), np.stack(ys)

        (x1, y1), (x2, y2) = get_batch(num_shots), get_batch(-1)
        return (x1, y1), (x2, y2)


class Heterescedastic(eqx.Module):
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
        return self.mu(x), self.stddev


def test_training():
    dataset = SinusoidRegression(16, 5, 5)
    make_model = lambda key: Heterescedastic(1, 16, 1, key)
    key = jax.random.PRNGKey(0)
    hyper_prior = pacoh.make_hyper_prior(make_model(key))
    hyper_posterior = pacoh.make_hyper_posterior(make_model, key, 10)
    opt = optax.flatten(optax.adam(3e-4))
    opt_state = opt.init(hyper_posterior)
    hyper_posterior = pacoh.meta_train(
        dataset.train_set, hyper_prior, hyper_posterior, opt, opt_state, 1000, 10
    )
    key, next_key = jax.random.split(key)
    (context_x, context_y), (test_x, test_y) = next(dataset.test_set)
    infer_posteriors = lambda x, y: pacoh.infer_posterior(
        x, y, hyper_posterior, next_key, 1000, 3e-4
    )
    infer_posteriors = jax.vmap(infer_posteriors)
    posteriors, losses = infer_posteriors(context_x, context_y)
    predict = jax.vmap(pacoh.predict)
    predictions = predict(posteriors, test_x)
    assert predictions[0].shape == test_y.shape
