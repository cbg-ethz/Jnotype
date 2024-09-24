"""Tests for discrete Fisher divergence ans pseudolikelihood methods."""

import jax.numpy as jnp

import pytest
import jnotype.energy._dfd as dfd


def linear_model(params, y):
    """Linear (independent) model.

    Args:
        params: vector of shape (G,)
        y: data point of shape (G,)
    """
    return jnp.sum(params * y)


def quadratic_model(params, y):
    """Quadratic (Ising) model.

    Args:
        params: matrix of shape (G, G)
        y: data point of shape (G,)
    """
    return jnp.einsum("ij,i,j->", params, y, y)


SETTINGS = [
    (jnp.zeros(3), linear_model),
    (jnp.zeros(5), linear_model),
    (jnp.zeros((3, 3)), quadratic_model),
    (jnp.zeros((5, 5)), quadratic_model),
]


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize(
    "divergence",
    [
        dfd.discrete_fisher_divergence,
        dfd.besag_pseudolikelihood_mean,
        dfd.besag_pseudolikelihood_sum,
    ],
)
def test_quasidivergence_smoke_test(setting, divergence, n_points: int = 2):
    theta, model = setting
    G = theta.shape[0]

    def logq(y):
        return model(theta, y)

    ys = jnp.zeros((n_points, G), dtype=int)

    value = divergence(logq, ys)
    assert value.shape == ()
