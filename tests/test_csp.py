"""Tests of the cumulative shrinkage prior (CSP) module."""
import jax
import jax.numpy as jnp
from jax import random

import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt

import pytest

import jnotype._csp as csp


def calculate_omega_slow(nu: jnp.ndarray) -> jnp.ndarray:
    ret = np.asarray(nu, dtype=float)

    for k in range(len(ret)):
        for j in range(k):
            ret[k] *= 1 - nu[j]
    return jnp.asarray(ret)


@pytest.mark.parametrize("h", [2, 5, 10])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_calculate_omega(alpha: float, h: int) -> None:
    key = jax.random.PRNGKey(0)
    nu = csp.prior_nu(key, alpha=alpha, h=h)
    omega = csp.calculate_omega(nu)
    omega_slow = calculate_omega_slow(nu)
    nptest.assert_allclose(omega, omega_slow, atol=1e-4)


@pytest.mark.parametrize("h", [50, 100])
@pytest.mark.parametrize("alpha", [1.0, 5.0, 8.0])
def test_expected_number_of_active_components(
    tmp_path,
    save_artifact: bool,
    alpha: float,
    h: int,
    draws: int = 300,
) -> None:
    key = random.PRNGKey(123)
    subkeys = random.split(key, draws)

    nus = jax.vmap(csp.prior_nu, in_axes=(0, None, None))(subkeys, alpha, h)
    pis = jax.vmap(csp.calculate_pi_from_nu)(nus)
    ks = jax.vmap(csp.calculate_expected_active_from_pi)(pis)

    # Check if the expected number of active components is `alpha`
    assert ks.mean() == pytest.approx(alpha, abs=0.1)

    if save_artifact:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=150)

        # Plot trajectories
        ax = axs[0]
        x_ax = np.arange(1, h + 1)

        x_ticks = x_ax[:: (1 + h // 10)]

        for pi in pis[:100]:
            ax.plot(x_ax, pi, alpha=0.1, c="k")

        ax.plot(x_ax, pis.mean(axis=0), alpha=1, linestyle="--", c="r")
        ax.axhline(1, c="navy", linewidth=0.4)
        ax.set_xticks(x_ticks, x_ticks)

        # Plot histogram
        ax = axs[1]
        ax.hist(ks, bins=np.arange(0.5, 20.5, 1.0), alpha=0.3, color="k")
        ax.axvline(alpha, linestyle="--", c="crimson")
        ax.axvline(ks.mean(), linestyle="--", c="navy", alpha=0.5)

        ax.set_xticks(x_ticks, x_ticks)

        fig.tight_layout()
        fig.savefig(tmp_path / f"test_csp-{h}-{alpha}.pdf")
