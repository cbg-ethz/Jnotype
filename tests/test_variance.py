import pytest
from jax import random
import jax.numpy as jnp

import jnotype._variance as _var


@pytest.mark.parametrize("seed", (21,))
@pytest.mark.parametrize("n_vars", (30_000,))
@pytest.mark.parametrize("prior_shape", (3.5, 7.0))
@pytest.mark.parametrize("prior_scale", (1.5, 2.1, 3.0))
def test_prior(seed: int, n_vars: int, prior_shape: float, prior_scale: float) -> None:
    """Test whether samples
    from prior match the analytical formulae.

    Args:
        prior_shape: should be greater than 2, so that variance exists
        prior_scale: should be strictly positive
    """
    samples = jnp.zeros((3, n_vars))
    # All samples are inactive, i.e., there
    # are no samples and we have just prior
    mask = jnp.zeros_like(samples, dtype=int)

    variances = _var.sample_variances(
        key=random.PRNGKey(seed),
        values=samples,
        mask=mask,
        prior_shape=prior_shape,
        prior_scale=prior_scale,
    )
    assert variances.shape == (n_vars,)

    empirical_mean = jnp.mean(variances)
    empirical_variance = jnp.var(variances)

    analytic_mean = prior_scale / (prior_shape - 1)
    analytic_variance = analytic_mean**2 / (prior_shape - 2)

    assert empirical_mean == pytest.approx(analytic_mean, rel=0.02)
    assert empirical_variance == pytest.approx(analytic_variance, rel=0.05)


def test_infinite_data() -> None:
    """Test whether in the infinite-data
    limit we get the shrinkage around true values."""
    assert False
