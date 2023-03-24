"""Sampling cluster labels and proportions."""
import jax
import jax.numpy as jnp

from jax import random
from jaxtyping import Array, Float, Int


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
    key: random.PRNGKeyArray,
    *,
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


def calculate_counts(labels: Int[Array, " N"], n_clusters: int) -> Int[Array, " B"]:
    """Calculates the occurrences of each cluster label.

    Args:
        labels: vector of labels. Each from the set ``{0, 1, ..., B-1}``
        n_clusters: number of clusters, also denoted as ``B``

    Returns:
        occurences of eac

    Note:
        This uses ``jax.numpy.bincount``, so any negative values present
          (or too large) may be truncated. No checking on our side is used.
    """
    return jnp.bincount(labels, length=n_clusters)


def sample_cluster_proportions(
    key: random.PRNGKeyArray,
    *,
    labels: Int[Array, " N"],
    dirichlet_prior: Float[Array, " B"],
) -> Float[Array, " B"]:
    """Samples the cluster proportions.

    Args:
        key: JAX random key
        labels: cluster labels, from the set :math:`\\{0, 1, ..., B-1\\}`,
          for each data point
        dirichlet_prior: parameters :math`\\alpha` of the Dirichlet
          distribution specifying the prior

    Returns:
        sampled cluster proportions: entries between (0, 1), summing up to 1
    """
    n_clusters = dirichlet_prior.shape[0]
    counts = calculate_counts(labels=labels, n_clusters=n_clusters)
    return random.dirichlet(key, alpha=counts + dirichlet_prior)


def sample_prob_code_given_cluster(
    key: random.PRNGKeyArray,
    *,
    codes: Int[Array, "N K"],
    labels: Int[Array, " N"],
    n_labels: int,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> Float[Array, "K B"]:
    """Samples the probability matrix
      P(code[k]=1 | cluster = b)
    given the observations.

    Todo:
        Refactor this function into deterministic summary statistic
          calculation and then sampling.
    """
    labels_encoded = jax.nn.one_hot(labels, n_labels)

    statistic_success: Int[Array, "N B"] = jnp.einsum(
        "NK,NB->KB", codes, labels_encoded
    )
    statistic_fails: Int[Array, "N B"] = jnp.einsum(
        "NK,NB->KB", 1 - codes, labels_encoded
    )

    return random.beta(key, prior_a + statistic_success, prior_b + statistic_fails)
