"""Tests of the cumulative shrinkage prior (CSP) module."""
import jax
import jax.numpy as jnp
from jax import random

import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from scipy import stats

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
    nu = csp.sample_prior_nu(key, alpha=alpha, h=h)
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

    nus = jax.vmap(csp.sample_prior_nu, in_axes=(0, None, None))(subkeys, alpha, h)
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


def log_pdf_multivariate_t(x, mask, *, dof: float, multiple: float) -> float:
    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)
    y = x[mask]
    dim = len(y)

    return stats.multivariate_t.logpdf(
        y, loc=np.zeros(dim), shape=multiple * np.eye(dim), df=dof
    )


@pytest.mark.parametrize("dof", [1, 2, 5])
@pytest.mark.parametrize("multiple", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("k", [1, 5, 30])
@pytest.mark.parametrize("sparsity", [0.1, 0.5])
def test_multivariate_student(
    dof: float, multiple: float, k: int, sparsity: float
) -> None:
    key = random.PRNGKey(123)
    key, *subkeys = random.split(key, 3)

    x = random.normal(subkeys[0], shape=(k,))
    mask = random.bernoulli(subkeys[1], p=sparsity, shape=(k,))
    if mask.sum() == 0:
        mask = mask.at[0].set(True)

    assert log_pdf_multivariate_t(x, mask, dof=dof, multiple=multiple) == pytest.approx(
        csp._log_pdf_multivariate_t(x=x, mask=mask, dof=dof, multiple=multiple)
    )


def log_pdf_multivariate_normal(x, mask, multiple) -> float:
    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)
    y = x[mask]
    dim = len(y)
    return stats.multivariate_normal.logpdf(
        y, mean=np.zeros(dim), cov=multiple * np.eye(dim)
    )


@pytest.mark.parametrize("multiple", [0.1, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("k", [1, 10, 30])
@pytest.mark.parametrize("sparsity", [0.1, 0.5])
def test_multivariate_normal(multiple: float, k: int, sparsity: float) -> None:
    key = random.PRNGKey(256)
    key, *subkeys = random.split(key, 3)

    x = random.normal(subkeys[0], shape=(k,))
    mask = random.bernoulli(subkeys[1], p=sparsity, shape=(k,))
    if mask.sum() == 0:
        mask = mask.at[0].set(True)

    assert log_pdf_multivariate_normal(x, mask, multiple=multiple) == pytest.approx(
        csp.log_pdf_multivariate_normal(x=x, mask=mask, multiple=multiple)
    )
