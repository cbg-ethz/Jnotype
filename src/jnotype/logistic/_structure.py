"""Sample structure (spike/slab distinction) variables."""
from typing import cast

import jax
import jax.numpy as jnp
from jax import random

from jaxtyping import Array, Float, Int
from jnotype.logistic.logreg import calculate_loglikelihood_matrix_from_variables


def _softmax_p1(
    *, log_p0: Float[Array, " X"], log_p1: Float[Array, " X"]
) -> Float[Array, " X"]:
    """Having access to *potentially unnormalized* log-probabilities
    :math:`Z + \\log P(Y=0)` and :math:`Z + \\log P(Y=1)`
    calculates the normalized probability :math:`P(Y=1)`.

    This function is vectorized: entries in log-probabilities
    are vectors representing different data points
    """
    concatenated = jnp.vstack(
        [
            log_p0,
            log_p1,
        ]
    ).T
    return jax.nn.softmax(concatenated, axis=1)[:, 1]


def _logpdf_gaussian(xs: Float[Array, " X"], variance: float) -> Float[Array, " X"]:
    """Returns :math:`\\log N(x | 0, variance)` vector for `x` in `xs`
    *up to a numerical constant*."""
    log_stds: Float[Array, " X"] = 0.5 * jnp.log(variance)
    return -0.5 * jnp.square(xs) / variance - log_stds


def sample_structure(
    *,
    key: random.PRNGKeyArray,
    intercepts: Float[Array, " features"],
    coefficients: Float[Array, "features covs"],
    structure: Int[Array, "features covs"],
    covariates: Float[Array, "points covs"],
    observed: Int[Array, "points features"],
    variances: Float[Array, " covs"],
    pseudoprior_variance: float,
    gamma: float,
) -> Int[Array, "features covs"]:
    """We have `covs` covariates (predictors) used
    to model `observed` points with `features` features each
    using logistic regression.

    We assume that the linear mapping
    `covariates -> logits(features)` is sparse,
    so that
    (a) we model the entries using spike-and-slab
    prior:
    P(turned on) * N(0, variance[k]) + P(turned off) * N(0, pseudovariance)
    (b) the contribution to the likelihood is structure * coefficient,
    so that turned off coefficients do not actually contribute anything.

    In this step we sample from P(turned on) for each entry.

    Args:
        key: JAX random key
        intercepts: for each observed feature the intercept term
          in the mapping `covariates -> logits(features)`
        coefficients: coefficients of the linear part of the affine
          mapping `covariates -> logits(features)`
        observed: observed features for each data point
        variances: vector of coefficient variances
        structure: current structure (binary mask with 0s and 1s)
        covariates: covariates for all points
        pseudoprior_variance: variance of the "pseudoprior" (the spike
          part of the prior. However, we sample from pseudoprior,
          the coefficient is turned off and contribution to the likelihood
          is exactly zero)
        gamma: probability of an individual entry to be turned on.
          Controls the sparsity (values near 0 result in a prior to
          be a very sparse matrix)
    """
    K = coefficients.shape[-1]
    keys = jax.random.split(key, K)

    structure = jnp.asarray(structure, dtype=int)

    # This function samples structure[:, k]
    def body_fun(
        k: int, struct: Int[Array, "features covs"]
    ) -> Float[Array, "features covs"]:
        """Samples all entries of `structure[:, k]` as these
        can be calculated in vectorized fashion (different features
        have access to different variables)"""
        # log p(Y | variables, structure[:, k] = 0)
        log_likelihood0: Float[
            Array, " features"
        ] = calculate_loglikelihood_matrix_from_variables(
            intercepts=intercepts,
            coefficients=coefficients,
            covariates=covariates,
            observed=observed,
            structure=struct.at[:, k].set(0),
        ).sum(
            axis=0
        )  # We sum along `samples` dimension

        # log p(Y | variables, structure[:, k] = 1)
        log_likelihood1: Float[
            Array, " features"
        ] = calculate_loglikelihood_matrix_from_variables(
            intercepts=intercepts,
            coefficients=coefficients,
            covariates=covariates,
            observed=observed,
            structure=struct.at[:, k].set(1),
        ).sum(
            axis=0
        )  # We sum along `samples` dimension

        # Now we want to include log-priors for the coefficients.
        # Both will be of shape (features,).
        # We do this for the pseudoprior (structure turned off)...
        log_coef0: Float[Array, " features"] = _logpdf_gaussian(
            coefficients[:, k], variance=pseudoprior_variance
        )

        # It's really an Array of shape (,), but we cast it to float
        # to make type checker work
        variance = cast(float, variances[k])
        # and for the prior with structure turned on:
        log_coef1: Float[Array, " features"] = _logpdf_gaussian(
            coefficients[:, k], variance=variance
        )

        # Finally, we have the prior on coefficients, controlled by gamma
        log_prior0 = jnp.log1p(-gamma)  # This is log(1-gamma)
        log_prior1 = jnp.log(gamma)

        # Now we can calculate unnormalized posteriors. Both are shape (features,)
        log_p0 = log_likelihood0 + log_coef0 + log_prior0
        log_p1 = log_likelihood1 + log_coef1 + log_prior1

        # Finally, we want to get p1/(p0 + p1). We will use softmax for this
        p1 = _softmax_p1(log_p0=log_p0, log_p1=log_p1)
        sampled = jnp.asarray(random.bernoulli(keys[k], p1), dtype=int)
        return struct.at[:, k].set(sampled)

    return jax.lax.fori_loop(0, K, body_fun, structure)