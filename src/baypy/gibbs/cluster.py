"""Sampling cluster labels and proportions."""
from jaxtyping import Array, Float, Int
import jax.numpy as jnp
from jax import random


def bernoulli_loglikelihood(
    observed: Int[Array, "N K"],
    prob_code_given_cluster: Float[Array, "K B"],
) -> Float[Array, "N B"]:
    """This is the log-likelihood matrix parametrized
    by the (potential) cluster label of each sample.

    Args:
        observed: observed binary codes for each data point. Each point has
          K independently sampled features
        prob_code_given_cluster: probability matrix P(code[k]=1 | cluster = b)

    Returns:
        matrix storing for each sample
          the loglikelihood B-vector P(codes | label)
    """
    # To calculate P(Z | codes) we need to calculate P(codes | Z)
    # This can be done easily using log-probabilities:
    # log P(codes | Z) = sum_k log P(codes[k] | Z)
    # These are the terms corresponding to the successes (ones)
    part1 = jnp.einsum("KB,NK->NB", jnp.log(prob_code_given_cluster), observed)
    # These are the terms corresponding to the failures (zeros)
    part2 = jnp.einsum("KB,NK->NB", jnp.log1p(-prob_code_given_cluster), 1 - observed)

    return part1 + part2


def sample_cluster_labels(
    *,
    key: random.PRNGKeyArray,
    cluster_proportions: Float[Array, " B"],
    prob_code_given_cluster: Float[Array, "K B"],
    binary_codes: Int[Array, "N K"],
) -> Int[Array, " N"]:
    """Samples cluster labels basing on conditionally independent.

    Args:
        key: JAX random key
        cluster_proportions: vector specifying prevalence
          of each class, shape (B,)
        prob_code_given_cluster: probability matrix P(code[k]=1 | cluster = b)
        binary_codes: observed binary codes for each data point

    Returns:
        sampled label for each data point.
          The values are from the set {0, 1, ..., B-1}
    """
    # This is a B-dimensional vector encoding the log-prior of P(Z)
    log_prior: Float[Array, " B"] = jnp.log(cluster_proportions)

    log_likelihood: Float[Array, "N B"] = bernoulli_loglikelihood(
        observed=binary_codes,
        prob_code_given_cluster=prob_code_given_cluster,
    )

    # Now we add the log-prior and have (potentially unnormalized) B-vector
    # log P(Z|codes) for each sample
    logits: Float[Array, "N B"] = log_likelihood + log_prior[None, ...]

    # Finally, we can sample the label for each data point
    return random.categorical(key, logits, axis=1)
