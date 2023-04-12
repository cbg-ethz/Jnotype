"""Logistic regression sampling utilities using Pólya-Gamma augmentation."""
from jax import random
import jax
import jax.numpy as jnp

import numpy as np
import polyagamma as pg

from jaxtyping import Array, Int, Float


def _calculate_logits(
    coefficients: Float[Array, "features covariates"],
    covariates: Float[Array, "points covariates"],
) -> Float[Array, "points features"]:
    return jnp.einsum("FC,NC->NF", coefficients, covariates)


_calculate_logits_jit = jax.jit(_calculate_logits)


def _sample_coefficients(
    *,
    key: random.PRNGKeyArray,
    omega: Float[Array, "points features"],
    covariates: Float[Array, "points covariates"],
    prior_mean: Float[Array, "features covariates"],
    prior_variance: Float[Array, "features covariates"],
    observed: Int[Array, "points features"],
) -> Float[Array, "features covariates"]:
    """The backend of `sample_coefficients`, which has access
    to the auxiliary Pólya-Gamma variables."""
    x_omega_x: Float[Array, "features covariates covariates"] = jnp.einsum(
        "NC,NF,NK->FKC", covariates, omega, covariates
    )
    precision_matrices: Float[Array, "features covariates covariates"] = jax.vmap(
        jnp.diag
    )(jnp.reciprocal(prior_variance))
    posterior_covariances: Float[
        Array, "features covariates covariates"
    ] = jnp.linalg.inv(x_omega_x + precision_matrices)

    kappa: Float[Array, "points features"] = jnp.asarray(observed, dtype=float) - 0.5

    bracket = jnp.einsum("NC,NF->FC", covariates, kappa) + jnp.einsum(
        "FCK,FK->FC", posterior_covariances, prior_mean
    )
    posterior_means: Float[Array, "features covariates"] = jnp.einsum(
        "FCK,FK->FC", posterior_covariances, bracket
    )

    return random.multivariate_normal(
        key,
        mean=posterior_means,
        cov=posterior_covariances,
    )


_sample_coefficients_jit = jax.jit(_sample_coefficients)


def sample_coefficients(
    jax_key: random.PRNGKeyArray,
    numpy_rng: np.random.Generator,
    observed: Int[Array, "points features"],
    design_matrix: Float[Array, "points covariates"],
    coefficients: Float[Array, "features covariates"],
    coefficients_prior_mean: Float[Array, "features covariates"],
    coefficients_prior_variance: Float[Array, "features covariates"],
) -> Float[Array, "features covariates"]:
    """

    Note:
        This function uses Pólya-Gamma sampler, which is *not*
        compatible with JAX! Therefore, it cannot be compiled
        or vmapped.
    """
    # Calculate logits using JITted code
    logits = _calculate_logits_jit(coefficients=coefficients, covariates=design_matrix)

    # Use Pólya-Gamma sampler to sample auxiliary variables omega
    omega: Float[Array, "points features"] = jnp.asarray(
        pg.random_polyagamma(1, z=logits, random_state=numpy_rng), dtype=float
    )

    # Sample the coefficients using JITted code which is given auxiliary variables
    return _sample_coefficients_jit(
        key=jax_key,
        omega=omega,
        covariates=design_matrix,
        prior_mean=coefficients_prior_mean,
        prior_variance=coefficients_prior_variance,
        observed=observed,
    )
