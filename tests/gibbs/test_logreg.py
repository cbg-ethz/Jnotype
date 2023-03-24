"""Tests for the logistic regression."""
import jax.numpy as jnp

import baypy.gibbs.logreg as lr


def test_calculate_logits_smoke_test() -> None:
    """The simplest smoke test. Just runs the function
    and checks the shape of the output."""
    intercepts = jnp.asarray([1.0, 1.0])
    mixing = jnp.asarray(
        [
            [1.0, 2.0, 0.0],
            [0.0, 5.0, 3.0],
        ]
    )
    structure = jnp.ones(mixing.shape, dtype=int)

    covariates = jnp.asarray(
        [
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 3.0],
            [4.0, 5.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    logits = lr.calculate_logits(
        intercepts=intercepts,
        coefficients=mixing,
        structure=structure,
        covariates=covariates,
    )

    assert logits.shape == (4, 2)
