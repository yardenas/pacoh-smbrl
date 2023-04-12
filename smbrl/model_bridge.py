import equinox as eqx
import jax.numpy as jnp

from smbrl.models import Model


class ModelBridge(eqx.Module):
    model: Model

    def __call__(self, x):
        preds = self.model(x)
        y_hat = jnp.concatenate([preds.next_state, preds.reward], -1)
        stddev = jnp.concatenate([preds.next_state_stddev, preds.reward_stddev], -1)
        return y_hat, stddev
