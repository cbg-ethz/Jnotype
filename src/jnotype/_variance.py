"""Utilities for sampling variances."""
from jax import random
import jax
import jax.numpy as jnp

from jaxtyping import Array, Float, Int


def _sample_precisions(
    key: random.PRNGKeyArray,
    values: Float[Array, "G F"],
    mask: Int[Array, "G F"],
    prior_shape: float,
    prior_rate: float,
) -> Float[Array, " F"]:
    """Similar to `sample_variances`,
    but we sample precisions.

    Args:
        prior_shape: shape parameter of the Gamma
          prior distribution
        prior_rate: *rate* parameter of the Gamma
          prior distribution
    """
    # Now we'll sample variances
    posterior_shape: Float[Array, " F"] = prior_shape + 0.5 * mask.sum(axis=0)
    posterior_rate: Float[Array, " F"] = prior_rate + 0.5 * (
        mask * jnp.square(values)
    ).sum(axis=0)

    precisions = random.gamma(key, posterior_shape) / posterior_rate
    return precisions


@jax.jit
def sample_variances(
    key: random.PRNGKeyArray,
    values: Float[Array, "G F"],
    mask: Int[Array, "G F"],
    prior_shape: float = 2.0,
    prior_scale: float = 1.0,
) -> Float[Array, " F"]:
    """Consider variables:

    .. math:

       X_{gf} \\sim N(0, \\sigma_f^2)

    with the prior on the variances:

    .. math:

       \\sigma_f^2 ~ InvGamma(\\alpha, \\beta)

    In this case, the posterior will also be sampled
    from :math:`InvGamma` family, parametrized by
    (a function of) the number of points :math:`X`
    and the observed variance.

    To allow different number of points per different
    variance :math:`\\sigma_f^2`, we use a binary `mask`.


    Args:
        key: JAX random key
        values: values :math:`X_{gf}`
        mask: binary mask (values 0 or 1), distinguishing
          whether a point should be counted or not
        prior_shape: shape parameter of the prior InvGamma
          distribution, should be strictly greater than 1
        prior_scale: scale parameter of the prior InvGamma
          distribution. Note that it's the *rate*
          parameter of the related Gamma distribution.
          Should be strictly positive.

    Returns:
        sampled variances :math:`\\sigma_f^2`

    Note:
        The mean of :math:`InvGamma(a, b)` is :math:`b/(a-1)`.
    """
    # We want to sample Var ~ InvGamma(shape=a, scale=b)
    # so we will sample Precision = 1/Var ~ Gamma(shape=a, rate=b)
    precisions = _sample_precisions(
        key=key,
        values=values,
        mask=mask,
        prior_shape=prior_shape,
        prior_rate=prior_scale,
    )
    return jnp.reciprocal(precisions)
