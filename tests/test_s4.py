import jax
import numpy as np
import pytest

import smbrl.agents.s4 as s4


def test_rnn_cnn(n=8, sequence_length=16, input_size=5):
    u = np.ones((sequence_length, input_size))
    cell = s4.S4Cell(n, input_size, sequence_length, key=jax.random.PRNGKey(666))
    y_cnn = cell.convolve(u)
    ssm = cell.ssm

    def sequence(x):
        def fn(carry, x):
            return cell(carry, x, ssm)

        return jax.lax.scan(fn, np.zeros((input_size, n), dtype=np.complex64), x)

    _, y_rnn = sequence(u)
    assert np.allclose(y_rnn.real, y_cnn.real, atol=1e-4, rtol=1e-4)


def test_output_dimension_not_equal(n=8, sequence_length=16, input_size=5):
    u = np.ones((sequence_length, input_size))
    cell = s4.S4Cell(n, input_size, sequence_length, key=jax.random.PRNGKey(666))
    y_cnn = cell.convolve(u).real
    sequence_summary = y_cnn.sum(0)
    pairwise_diffs = sequence_summary[:, None] - sequence_summary[None, :]
    # If all the pairwise differences are zero, then the model predicts exactly
    # the same for each dimension in the output
    assert not np.all(np.nonzero(pairwise_diffs))


@pytest.mark.parametrize(
    "mode",
    ["cell", "other"],
    ids=["cell", "other"],
)
def test_kernels(mode, n=8, sequence_length=16, input_size=5):
    # Compute a HiPPO NPLR matrix.
    if mode == "cell":
        cell = s4.S4Cell(n, input_size, sequence_length, key=jax.random.PRNGKey(666))
        _lambda = cell.lambda_real + 1j * cell.lambda_imag
        p = cell.p
        b = cell.b
        c = cell.c[..., 0] + 1j * cell.c[..., 1]
        step = np.exp(cell.step)
    else:
        cell = None
        step = np.ones((input_size,)) * 1.0 / sequence_length
        # Compute a HiPPO NPLR matrix.
        _lambda, p, b, _ = [np.tile(x, (input_size, 1)) for x in s4.make_DPLR_HiPPO(n)]
        # Random complex Ct
        c = jax.random.normal(jax.random.PRNGKey(666), (input_size, n, 2))
        c = c[..., 0] + 1j * c[..., 1]
    # Random complex Ct
    k = jax.vmap(s4.kernel_DPLR, (0, 0, 0, 0, 0, 0, None))(
        _lambda, p, p, b, c, step, sequence_length
    )
    # RNN form.
    ab, bb, cb = jax.vmap(s4.discrete_DPLR, (0, 0, 0, 0, 0, 0, None))(
        _lambda, p, p, b, c, step, sequence_length
    )

    if mode == "cell":
        assert cell is not None
        assert all(np.allclose(x1, x2) for x1, x2 in zip((ab, bb, cb), cell.ssm))

    def K_conv(Ab, Bb, Cb, L):
        return jax.numpy.array(
            [
                (Cb @ jax.numpy.linalg.matrix_power(Ab, length) @ Bb).reshape()
                for length in range(L)
            ]
        )

    k2 = jax.vmap(K_conv, (0, 0, 0, None))(ab, bb, cb, sequence_length)
    assert np.allclose(k.real, k2.real, atol=1e-5, rtol=1e-5)


def test_fft_mode(n=8, sequence_length=16, input_size=5):
    cell = s4.S4Cell(n, input_size, sequence_length, key=jax.random.PRNGKey(666))
    _lambda = cell.lambda_real + 1j * cell.lambda_imag
    p = cell.p
    b = cell.b
    c = cell.c[..., 0] + 1j * cell.c[..., 1]
    step = jax.numpy.exp(cell.step)
    k = jax.vmap(s4.kernel_DPLR, (0, 0, 0, 0, 0, 0, None))(
        _lambda, p, p, b, c, step, sequence_length
    )
    u = jax.numpy.ones((sequence_length, input_size))

    def fft(u, k):
        ud = jax.numpy.fft.rfft(jax.numpy.pad(u, (0, k.shape[0])))
        Kd = jax.numpy.fft.rfft(jax.numpy.pad(k, (0, u.shape[0])))
        out = ud * Kd
        return jax.numpy.fft.irfft(out)[: u.shape[0]]

    y1 = jax.vmap(fft, (1, 0), 1)(u, k) + cell.d.T * u
    y2 = cell.convolve(u)
    assert np.allclose(y1.real, y2.real, atol=1e-4, rtol=1e-4)


def test_fft_mode_np(n=8, sequence_length=16, input_size=5):
    cell = s4.S4Cell(n, input_size, sequence_length, key=jax.random.PRNGKey(666))
    u = jax.numpy.ones((sequence_length, input_size))
    y1 = cell.convolve(u, True)
    y2 = cell.convolve(u, False)
    assert jax.numpy.allclose(y1, y2, atol=1e-4, rtol=1e-4)
