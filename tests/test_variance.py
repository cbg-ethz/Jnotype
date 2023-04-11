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


@pytest.mark.parametrize("seed", (42, 12))
@pytest.mark.parametrize("n_points", (100_000,))
def test_infinite_data(
    seed: int,
    n_points: int,
    prior_shape: float = 2.0,
    prior_scale: float = 1.0,
) -> None:
    """Test whether in the infinite-data
    limit we get the shrinkage around true values."""
    keys = random.split(random.PRNGKey(seed), 3)

    var1, var2 = 0.8, 1.2

    samples1 = jnp.sqrt(var1) * random.normal(keys[0], shape=(n_points,))
    samples2 = jnp.sqrt(var2) * random.normal(keys[1], shape=(n_points,))

    samples = jnp.vstack([samples1, samples2]).T
    assert samples.shape == (n_points, 2)

    mask = jnp.ones_like(samples, dtype=int)

    samples = _var.sample_variances(
        key=keys[2],
        values=samples,
        mask=mask,
        prior_shape=prior_shape,
        prior_scale=prior_scale,
    )
    assert samples.shape == (2,)

    assert samples[0] == pytest.approx(var1, rel=0.01)
    assert samples[1] == pytest.approx(var2, rel=0.01)
