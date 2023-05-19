"""Logistic regression sampling utilities using Pólya-Gamma augmentation."""
from jax import random
import jax
import jax.numpy as jnp

import numpy as np
import polyagamma as pg

from jaxtyping import Array, Int, Float


def _calculate_logits(
    coefficients: Float[Array, "features covariates"],
    structure: Int[Array, "features covariates"],
    covariates: Float[Array, "points covariates"],
) -> Float[Array, "points features"]:
    return jnp.einsum("FC,FC,NC->NF", coefficients, structure, covariates)


_calculate_logits_jit = jax.jit(_calculate_logits)


def _sample_coefficients(
    *,
    key: random.PRNGKeyArray,
    omega: Float[Array, "points features"],
    covariates: Float[Array, "points covariates"],
    structure: Int[Array, "features covariates"],
    prior_mean: Float[Array, "features covariates"],
    prior_variance: Float[Array, "features covariates"],
    observed: Int[Array, "points features"],
) -> Float[Array, "features covariates"]:
    """The backend of `sample_coefficients`, which has access
    to the auxiliary Pólya-Gamma variables."""

    # This bit requires some explanation. If F=1 and we have only one
    # output, we want to have
    # matrix[c, k] = sum_n  X[n, k] omega[n] X[n, c]
    # However, we have F outputs and each of them has access to different
    # covariates due to the sparsity structure!
    # The covariates are U[f, n, c] = X[n, c] * structure[f, c]
    x_omega_x: Float[Array, "features covariates covariates"] = jnp.einsum(
        "NC,FC,NF,NK,FK->FCK",
        covariates,
        structure,
        omega,
        covariates,
        structure,
        # Empirically this seems to be giving 2x speedup
        # over not specifying
        optimize="greedy",
    )
    precision_matrices: Float[Array, "features covariates covariates"] = jax.vmap(
        jnp.diag
    )(jnp.reciprocal(prior_variance))
    posterior_covariances: Float[
        Array, "features covariates covariates"
    ] = jnp.linalg.inv(x_omega_x + precision_matrices)

    kappa: Float[Array, "points features"] = jnp.asarray(observed, dtype=float) - 0.5

    # Check if this is right...
    bracket = jnp.einsum("NC,FC,NF->FC", covariates, structure, kappa) + jnp.einsum(
        "FCK,FK->FC", precision_matrices, prior_mean
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
    *,
    jax_key: random.PRNGKeyArray,
    numpy_rng: np.random.Generator,
    observed: Int[Array, "points features"],
    design_matrix: Float[Array, "points covariates"],
    coefficients: Float[Array, "features covariates"],
    structure: Int[Array, "features covariates"],
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
    logits = _calculate_logits_jit(
        coefficients=coefficients,
        covariates=design_matrix,
        structure=structure,
    )

    # Use Pólya-Gamma sampler to sample auxiliary variables omega
    omega: Float[Array, "points features"] = jnp.asarray(
        pg.random_polyagamma(1, z=logits, random_state=numpy_rng), dtype=float
    )

    # Sample the coefficients using JITted code which is given auxiliary variables
    return _sample_coefficients_jit(
        key=jax_key,
        omega=omega,
        covariates=design_matrix,
        structure=structure,
        prior_mean=coefficients_prior_mean,
        prior_variance=coefficients_prior_variance,
        observed=observed,
    )


def _augment(matrix: Float[Array, "a b"]) -> Float[Array, "a b+1"]:
    n_points = matrix.shape[0]
    return jnp.hstack(
        (
            jnp.ones((n_points, 1)),
            matrix,
        )
    )


def _augmented_mean(
    intercept_prior_mean: float,
    augmented_coefficients: Float[Array, "features entries"],
) -> Float[Array, "features entries"]:
    n_features, n_augmented_covs = augmented_coefficients.shape
    n_covs = n_augmented_covs - 1

    return jnp.hstack(
        (
            jnp.full(shape=(n_features, 1), fill_value=intercept_prior_mean),
            jnp.zeros(shape=(n_features, n_covs)),
        )
    )


def _augmented_variances(
    intercept_variance: float,
    variance_per_covariate: Float[Array, " covariates"],
    structure: Int[Array, "features covariates"],
    pseudoprior_variance: float,
) -> Float[Array, "features covariates+1"]:
    # Shape (features, covariates)
    coefficient_variances = (
        structure * jnp.full(shape=structure.shape, fill_value=variance_per_covariate)
        + (1 - structure) * pseudoprior_variance
    )

    return jnp.hstack(
        (
            jnp.full(shape=(structure.shape[0], 1), fill_value=intercept_variance),
            coefficient_variances,
        )
    )


@jax.jit
def _augment_matrices(
    *,
    intercepts,
    covariates,
    coefficients,
    structure,
    intercept_prior_mean,
    intercept_prior_variance,
    pseudoprior_variance,
    variances,
) -> dict:
    # Create design matrix, storing intercepts
    # using the augmented design
    # Shape (points, covs+1)
    augmented_covariates = _augment(covariates)

    # Shape (features, covs+1)
    augmented_coefficients = jnp.hstack((intercepts.reshape((-1, 1)), coefficients))

    # Shape (features, covs+1)
    augmented_structure = _augment(structure)

    # Now construct the prior matrices,
    # remembering about the pseudoprior
    # controlled by structure
    # Shape (covs+1,)
    augmented_prior_mean = _augmented_mean(
        intercept_prior_mean=intercept_prior_mean,
        augmented_coefficients=augmented_coefficients,
    )
    # Shape (features, covs+1)
    augmented_prior_variances = _augmented_variances(
        pseudoprior_variance=pseudoprior_variance,
        structure=structure,
        intercept_variance=intercept_prior_variance,
        variance_per_covariate=variances,
    )

    return {
        "covariates": augmented_covariates,
        "coefficients": augmented_coefficients,
        "structure": augmented_structure,
        "prior_mean": augmented_prior_mean,
        "prior_variance": augmented_prior_variances,
    }


def sample_intercepts_and_coefficients(
    *,
    jax_key: random.PRNGKeyArray,
    numpy_rng: np.random.Generator,
    observed: Int[Array, "points features"],
    intercepts: Float[Array, " features"],
    coefficients: Float[Array, "features covs"],
    structure: Int[Array, "features covs"],
    covariates: Float[Array, "points covs"],
    variances: Float[Array, " covs"],
    pseudoprior_variance: float,
    intercept_prior_mean: float,
    intercept_prior_variance: float,
) -> tuple[Float[Array, " features"], Float[Array, "features covs"]]:
    """Samples coefficients and intercepts.

    Note:
        This uses the augmentation trick, i.e.
          X_aug = (1, X),
          coefs_aug = (intercept, coefs_aug),
          structure_aug = (1, structure),
        samples new version of `coefs_aug`
        and then splits it again into intercepts
        and coefficients.
    """
    augmented = _augment_matrices(
        intercepts=intercepts,
        covariates=covariates,
        coefficients=coefficients,
        structure=structure,
        intercept_prior_mean=intercept_prior_mean,
        intercept_prior_variance=intercept_prior_variance,
        pseudoprior_variance=pseudoprior_variance,
        variances=variances,
    )

    # Let's sample the augmented (features, covs+1)
    # coefficients
    augmented_sampled = sample_coefficients(
        jax_key=jax_key,
        numpy_rng=numpy_rng,
        observed=observed,
        design_matrix=augmented["covariates"],
        coefficients=augmented["coefficients"],
        structure=augmented["structure"],
        coefficients_prior_mean=augmented["prior_mean"],
        coefficients_prior_variance=augmented["prior_variance"],
    )

    # Now we split the sampled augmented coefficients
    # into intercepts and the rest
    intercepts_sampled = augmented_sampled[:, 0]
    coefficients_sampled = augmented_sampled[:, 1:]

    return intercepts_sampled, coefficients_sampled
