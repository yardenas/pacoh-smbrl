from itertools import cycle

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from optax import l2_loss

from smbrl.agents import model_learning as ml
from smbrl.agents.asmbrl import PACOHLearner as PCHLearner
from smbrl.agents.models import FeedForwardModel, S4Model
from smbrl.utils import Learner, PRNGSequence, ensemble_predict

ACTION_SPACE_DIM = 1
OBSERVATION_SPACE_DIM = 3
BATCH_SIZE = 32
CONTEXT = 12
SEQUENCE_LENGTH = 84

KEY = PRNGSequence(0)


def split_obs_acs(x):
    return x[..., :OBSERVATION_SPACE_DIM], x[..., OBSERVATION_SPACE_DIM:]


def split_context(x, y, context):
    pre = lambda x: x[:, :, :context]
    post = lambda x: x[:, :, context:]
    return tuple(map(pre, (x, y))), tuple(map(post, (x, y)))


def flat(prediction):
    prediction = jnp.concatenate(
        [prediction.next_state, prediction.reward[..., None]], axis=-1
    )
    return prediction


regression_step = eqx.filter_jit(ml.regression_step)


class PACOHLearner:
    def __init__(self):
        model_factory = lambda key: FeedForwardModel(
            state_dim=OBSERVATION_SPACE_DIM,
            action_dim=ACTION_SPACE_DIM,
            key=next(KEY),
            n_layers=3,
            hidden_size=64,
        )
        self.config = OmegaConf.create(
            dict(
                agent=dict(
                    pacoh=dict(
                        num_examples=100,
                        n_particles=3,
                        posterior_stddev=0.01,
                        n_prior_samples=5,
                        prior_weight=1e-4,
                        bandwidth=1000.0,
                    ),
                    model_optimizer=dict(lr=1e-3),
                )
            )
        )
        self.learner = PCHLearner(model_factory, next(KEY), self.config)
        self.posterior = self.learner.sample_prior(
            next(KEY), self.config.agent.pacoh.n_prior_samples, BATCH_SIZE
        )

    def train_step(self, data):
        data = tuple(map(lambda x: x[None], data))
        logprobs = self.learner.update_hyper_posterior(
            data,
            next(KEY),
            self.config.agent.pacoh.n_prior_samples,
            1,
            self.config.agent.pacoh.prior_weight,
            self.config.agent.pacoh.bandwidth,
        )
        return -logprobs

    def adapt(self, data):
        train_steps = 100
        train_data = tuple(map(lambda x: np.repeat(x[None], train_steps, axis=0), data))
        self.posterior, _ = self.learner.infer_posteriors(
            train_data, self.posterior, train_steps, 3e-4, next(KEY), 1e-7, 1000.0
        )

    def predict(self, x):
        horizon = x.shape[2]
        sample = lambda m, o, a: ensemble_predict(m.sample, (None, 0, None, 0))(
            horizon,
            o,
            jax.random.PRNGKey(0),
            a,
        )
        vmaped_sample = jax.vmap(sample)
        partial_sample = lambda o, a: vmaped_sample(self.posterior, o, a)
        mean_sample = lambda o, a: jax.tree_map(
            lambda x: x.mean(1), partial_sample(o[:, :, 0], a)
        )
        pred = mean_sample(*split_obs_acs(x))
        y_hat = flat(pred)
        return y_hat


class S4Learner:
    def __init__(self):
        self.model = S4Model(
            state_dim=OBSERVATION_SPACE_DIM,
            action_dim=ACTION_SPACE_DIM,
            key=next(KEY),
            sequence_length=SEQUENCE_LENGTH,
            n_layers=2,
            hidden_size=64,
            hippo_n=16,
        )
        self.learner = Learner(self.model, dict(lr=1e-3))
        self.hidden = None

    def train_step(self, data):
        (self.model, self.learner.state), loss = regression_step(
            data, self.model, self.learner, self.learner.state
        )
        return loss

    def adapt(self, data):
        def unroll_step(o, a):
            def f(carry, x):
                prev_hidden = carry
                observation, action = x
                hidden, out = self.model.step(observation, action, ssm, prev_hidden)
                return hidden, out

            init_hidden = self.model.init_state
            return jax.lax.scan(f, init_hidden, (o, a))

        ssm = self.model.ssm
        o, a = split_obs_acs(data[0])
        self.hidden, _ = jax.vmap(jax.vmap(unroll_step))(o, a)

    def predict(self, x):
        def sample(o, a, h):
            out = self.model.sample(
                horizon,
                o[0],
                jax.random.PRNGKey(0),
                a,
                ssm,
                h,
            )
            return out

        horizon = x.shape[2]
        ssm = self.model.ssm
        vmaped_sample = jax.vmap(jax.vmap(sample))
        pred = vmaped_sample(*split_obs_acs(x), self.hidden)
        y_hat = flat(pred)
        return y_hat


class VanillaLearner:
    def __init__(self):
        self.model = FeedForwardModel(
            state_dim=OBSERVATION_SPACE_DIM,
            action_dim=ACTION_SPACE_DIM,
            key=next(KEY),
            n_layers=3,
            hidden_size=64,
        )
        self.learner = Learner(self.model, dict(lr=3e-4))

    def train_step(self, data):
        (self.model, self.learner.state), loss = regression_step(
            data, self.model, self.learner, self.learner.state
        )
        return loss

    def adapt(self, data):
        pass

    def predict(self, x):
        horizon = x.shape[2]
        sample = lambda obs, acs: self.model.sample(
            horizon,
            obs[0],
            jax.random.PRNGKey(0),
            acs,
        )
        vmaped_sample = jax.vmap(jax.vmap(sample))
        pred = vmaped_sample(*split_obs_acs(x))
        y_hat = flat(pred)
        return y_hat


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jax.random.permutation(key, indices)
        (key,) = jax.random.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            obs, next_obs, acs, rews = tuple(array[batch_perm] for array in arrays)
            yield np.concatenate((obs, acs), -1), np.concatenate(
                (next_obs, rews[..., None]), -1
            )
            start = end
            end = start + batch_size


def get_data(data_path, sequence_length, split=slice(0, None)):
    obs, action, reward = np.load(data_path).values()

    def normalize(x):
        mean = x.mean(axis=(0))
        stddev = x.std(axis=(0))
        return (x - mean) / (stddev + 1e-8), mean, stddev

    obs, *_ = normalize(obs)
    action, *_ = normalize(action)
    reward, *_ = normalize(reward)
    all_obs, all_next_obs, all_acs, all_rews = [], [], [], []
    for t in range(action.shape[2] - sequence_length):
        all_obs.append(obs[split, :, t : t + sequence_length])
        all_next_obs.append(obs[split, :, t + 1 : t + sequence_length + 1])
        all_rews.append(reward[split, :, t : t + sequence_length])
        all_acs.append(action[split, :, t : t + sequence_length])
    obs, next_obs, acs, rews = map(
        lambda x: np.concatenate(x, axis=0), (all_obs, all_next_obs, all_acs, all_rews)  # type: ignore # noqa: E501
    )
    return obs, next_obs, acs, rews


def plot(context, y, y_hat, context_t):
    import matplotlib.pyplot as plt
    import numpy as np

    t_test = np.arange(y.shape[2]) + context.shape[2]
    t_context = np.arange(context.shape[2])

    plt.figure(figsize=(10, 5), dpi=600)
    for i in range(6):
        plt.subplot(3, 4, i + 1)
        plt.plot(t_context, context[i, 0, :, 2], "b.", label="context")
        plt.plot(
            t_test,
            y_hat[i, 0, :, 2],
            "r",
            label="prediction",
            linewidth=1.0,
        )
        plt.plot(
            t_test,
            y[i, 0, :, 2],
            "c",
            label="ground truth",
            linewidth=1.0,
        )
        ax = plt.gca()
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(context_t, color="k", linestyle="--", linewidth=1.0)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def run_algo(learner, train_data, test_data, steps):
    test_data = cycle(iter(test_data))
    for step, batch in zip(range(steps), train_data):
        loss = learner.train_step(batch)
        loss = loss.item()
        print(f"step={step}, loss={loss}")
        if step % 100 == 0:
            context, (x, y) = split_context(*next(test_data), CONTEXT)
            mse = learner.adapt(context)
            y_hat = learner.predict(x)
            mse = l2_loss(y_hat, y).mean()
            print(f"MSE={mse.item()}")
            plot(context[1], y, y_hat, CONTEXT)


def main():
    train_data = get_data("data-200-multi.npz", SEQUENCE_LENGTH, split=slice(0, 150))
    train_loader = dataloader(train_data, BATCH_SIZE, key=next(KEY))
    test_data = get_data("data-200-multi.npz", SEQUENCE_LENGTH, split=slice(150, None))
    test_loader = dataloader(test_data, BATCH_SIZE, key=next(KEY))
    for learner in [PACOHLearner()]:
        run_algo(learner, train_loader, test_loader, 1000)


if __name__ == "__main__":
    main()
