"""Sample binary latent variables."""
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from jaxtyping import Array, Float, Int
from jnotype.logistic.logreg import calculate_loglikelihood_matrix_from_variables
from jnotype.logistic._structure import _softmax_p1


@partial(jax.jit, static_argnames="n_binary_codes")
def sample_binary_codes(
    *,
    key: random.PRNGKeyArray,
    intercepts: Float[Array, " features"],
    coefficients: Float[Array, "features covs"],
    structure: Int[Array, "features covs"],
    covariates: Float[Array, "points covs"],
    observed: Int[Array, "points features"],
    n_binary_codes: int,
    labels: Int[Array, " points"],
    labels_to_codes: Float[Array, "n_binary_codes n_clusters"],
) -> Float[Array, "points covs"]:
    """Samples binary latent variables, producing a new matrix of features.

    For each data point we have some `covariates`.
    used to predict observed `features`.

    These `covariates` are split into two parts:
      - the first `n_binary_codes` entries are binary (0 or 1)
        latent variables.
      - the rest (n_binary_codes, n_binary_codes+1, ...)
        entries are fixed and can take arbitrary real values.

    In this function we resample the binary latent variables,
    (entries 0, 1, ..., n_binary_codes-1 for each data point)
    leaving the fixed covariates untouched.

    Args:
        key: JAX random key
        intercepts: intercepts in the logistic regression,
          one per observed feature
        coefficients: logistic regression mapping from covariates
          (including latent binary codes) to the observed features
        structure: sparsity structure matrix
        covariates: the first ``n_binary_codes`` columns will be sampled
        observed: observed binary features
        n_binary_codes: number of binary codes to be sampled. Note
          that it can be at most the number of covariates
        labels: cluster labels in the Bernoulli mixture model on the upper
          layer, used to specify the prior on the binary latent codes.
          Each entry should be in the set {0, ..., n_clusters-1}
        labels_to_codes: mixing matrix of the Bernoulli mixture model,
          used to specify the prior on the binary latent codes.
          Note that its dimension is `n_binary_codes`, rather than `covs`

    Note:
        To model the binary variables as independent,
        use one cluster label.

    Todo:
        TODO(Pawel): Refactor this function so that:
          (a) any flexible prior on latent codes can be used,
              not only the one from Bernoulli mixture
          (b) any likelihood function can be used
    """
    keys = jax.random.split(key, n_binary_codes)

    # Make sure that the covariates are represented as floats,
    # even if only binary codes are used
    covariates = jnp.asarray(covariates, dtype=float)

    # This function samples covariates[:, k]
    # Note that k should be 0, ..., n_binary_codes-1, rather than ranging to K-1
    def body_fun(
        k: int, covs: Float[Array, "features covariates"]
    ) -> Float[Array, "features covariates"]:
        """This function samples covs[:, k].

        Args:
            k: column to be sampled,
               takes valueus in {0, ..., n_binary_codes-1},
               rather than {0, ..., covariates.shape[1]-1}.
            covs: "local copy" of covariates, used in functional programming.
               Note that `covariates` should not be used in the function body,
               but passed as `covs` in the `fori_loop`.
        """
        # log p(Y | variables, codes[:, k] = 0). Shape (N,)
        log_likelihood0 = calculate_loglikelihood_matrix_from_variables(
            intercepts=intercepts,
            coefficients=coefficients,
            covariates=covs.at[:, k].set(0.0),
            observed=observed,
            structure=structure,
        ).sum(
            axis=1
        )  # We sum along observed features

        # log p(Y | variables, codes[:, k] = 1). Shape (N,)
        log_likelihood1 = calculate_loglikelihood_matrix_from_variables(
            intercepts=intercepts,
            coefficients=coefficients,
            covariates=covs.at[:, k].set(1.0),
            observed=observed,
            structure=structure,
        ).sum(
            axis=1
        )  # We sum along observed features

        # Now we need to calculate the terms related to the prior
        prior_activation_prob: Float[Array, " points"] = labels_to_codes[k, labels]

        # log(prior code[k] = 1), shape (points,)
        log_prior1 = jnp.log(prior_activation_prob)
        # log(prior code[k] = 0) = log(1 - prior code[k] = 1), shape (points,)
        log_prior0 = jnp.log1p(-prior_activation_prob)

        # Now we want to have p1 / (p0 + p1)
        p1 = _softmax_p1(
            log_p0=log_likelihood0 + log_prior0,
            log_p1=log_likelihood1 + log_prior1,
        )

        sampled = jnp.asarray(random.bernoulli(keys[k], p1), dtype=covs.dtype)
        return covs.at[:, k].set(sampled)

    return jax.lax.fori_loop(0, n_binary_codes, body_fun, covariates)
