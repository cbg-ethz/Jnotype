"""Sampling cluster labels and proportions."""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from jax import random
from jaxtyping import Array, Float, Int


import jnotype.sampling as js
from jnotype.sampling._chunker import DatasetInterface
from jnotype._utils import JAXRNG


def _bernoulli_loglikelihood(
    observed: Int[Array, "N K"],
    mixing: Float[Array, "K B"],
) -> Float[Array, "N B"]:
    """This is the log-likelihood matrix parametrized
    by the (potential) cluster label of each sample.

    Args:
        observed: observed binary codes for each data point. Each point has
          K independently sampled features
        mixing: probability matrix P(code[k]=1 | cluster = b)

    Returns:
        matrix storing for each sample
          the loglikelihood B-vector P(codes | label)
    """
    # To calculate P(Z | codes) we need to calculate P(codes | Z)
    # This can be done easily using log-probabilities:
    # log P(codes | Z) = sum_k log P(codes[k] | Z)
    # These are the terms corresponding to the successes (ones)
    part1 = jnp.einsum("KB,NK->NB", jnp.log(mixing), observed)
    # These are the terms corresponding to the failures (zeros)
    part2 = jnp.einsum("KB,NK->NB", jnp.log1p(-mixing), 1 - observed)

    return part1 + part2


def sample_cluster_labels(
    key: jax.Array,
    *,
    cluster_proportions: Float[Array, " B"],
    mixing: Float[Array, "K B"],
    binary_codes: Int[Array, "N K"],
) -> Int[Array, " N"]:
    """Samples cluster labels basing on conditionally independent.

    Args:
        key: JAX random key
        cluster_proportions: vector specifying prevalence
          of each class, shape (B,)
        mixing: probability matrix P(code[k]=1 | cluster = b)
        binary_codes: observed binary codes for each data point

    Returns:
        sampled label for each data point.
          The values are from the set {0, 1, ..., B-1}
    """
    # This is a B-dimensional vector encoding the log-prior of P(Z)
    log_prior: Float[Array, " B"] = jnp.log(cluster_proportions)

    log_likelihood: Float[Array, "N B"] = _bernoulli_loglikelihood(
        observed=binary_codes,
        mixing=mixing,
    )

    # Now we add the log-prior and have (potentially unnormalized) B-vector
    # log P(Z|codes) for each sample
    logits: Float[Array, "N B"] = log_likelihood + log_prior[None, ...]

    # Finally, we can sample the label for each data point
    return random.categorical(key, logits, axis=1)


def _calculate_counts(labels: Int[Array, " N"], n_clusters: int) -> Int[Array, " B"]:
    """Calculates the occurrences of each cluster label.

    Args:
        labels: vector of labels. Each from the set ``{0, 1, ..., B-1}``
        n_clusters: number of clusters, also denoted as ``B``

    Returns:
        array of shape (n_clusters,) with occurrences of each cluster label

    Note:
        This uses ``jax.numpy.bincount``, so any negative values present
          (or too large) may be truncated. No checking on our side is used.
    """
    return jnp.bincount(labels, length=n_clusters)


def sample_cluster_proportions(
    key: jax.Array,
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
    counts = _calculate_counts(labels=labels, n_clusters=n_clusters)
    return random.dirichlet(key, alpha=counts + dirichlet_prior)


def sample_mixing(
    key: jax.Array,
    *,
    observations: Int[Array, "N K"],
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
        "NK,NB->KB", observations, labels_encoded
    )
    statistic_fails: Int[Array, "N B"] = jnp.einsum(
        "NK,NB->KB", 1 - observations, labels_encoded
    )

    return random.beta(key, prior_a + statistic_success, prior_b + statistic_fails)


@jax.jit
def single_sampling_step(
    *,
    key: jax.Array,
    observed_data: Int[Array, "N K"],
    proportions: Float[Array, " B"],
    mixing: Float[Array, "K B"],
    dirichlet_prior: Float[Array, " B"],
    beta_prior: tuple[float, float],
) -> tuple[Int[Array, " N"], Float[Array, " B"], Float[Array, "K B"]]:
    """Single sampling step.

    Args:
        key: JAX random key
        observed_data: observed data Y, shape (n_samples, n_features)
        proportions: P(Z), shape (n_clusters,)
        mixing: P(Y[:, g]=1 | Z=k), shape (n_features, n_clusters)
        dirichlet_prior: weights of the Dirichlet prior to sample
          cluster labels, shape (n_clusters,)
        beta_prior: prior on the entries of `mixing`. Note that if it is (a, b),
          then `a` describes the success (Y=1) and `b` the failure (Y=0)

    Returns:
        sampled cluster labels, shape (N,). Each entry in set {0, ..., B-1}
        sampled proportions, shape (B,). Sums up to 1.
        sampled mixing matrix, shape (K, B). Each entry is from the interval (0, 1)
    """
    key1, key2, key3 = random.split(key, 3)

    # Sample labels
    labels = sample_cluster_labels(
        key=key1,
        cluster_proportions=proportions,
        mixing=mixing,
        binary_codes=observed_data,
    )
    # Sample proportions
    proportions = sample_cluster_proportions(
        key=key2,
        labels=labels,
        dirichlet_prior=dirichlet_prior,
    )
    # Sample mixing
    mixing = sample_mixing(
        key3,
        observations=observed_data,
        labels=labels,
        n_labels=proportions.shape[0],
        prior_a=beta_prior[0],
        prior_b=beta_prior[1],
    )

    return labels, proportions, mixing


class BernoulliMixtureGibbsSampler(js.AbstractGibbsSampler):
    """Gibbs sampler for the Bernoulli mixture model."""

    def __init__(
        self,
        datasets: Sequence[DatasetInterface],
        observed_data: Int[Array, "observation feature"],
        dirichlet_prior: Float[Array, " cluster"],
        beta_prior: tuple[float, float] = (1.0, 1.0),
        *,
        jax_rng_key: Optional[jax.Array] = None,
        warmup: int = 2000,
        steps: int = 3000,
        verbose: bool = False,
    ) -> None:
        """
        Args:
          observed_data: observed data, shape (n_points, n_features)
          dirichlet_prior: Dirichlet prior weights
            on the cluster proportions, shape (n_clusters,)
          beta_prior: beta prior weights on the mixing matrix entries
        """
        super().__init__(datasets, warmup=warmup, steps=steps, verbose=verbose)

        jax_rng_key = jax_rng_key or jax.random.PRNGKey(10)
        self._rng = JAXRNG(jax_rng_key)

        self._observed_data = observed_data
        self._dirichlet_prior = dirichlet_prior
        self._beta_prior = beta_prior

    @classmethod
    def dimensions(cls) -> dict:
        """Named dimensions of each sample."""
        return {
            "labels": ["observation"],
            "proportions": ["cluster"],
            "mixing": ["feature", "cluster"],
        }

    def new_sample(self, sample: dict) -> dict:
        """A new sample with keys:
        "labels", "proportions", "mixing".
        """
        labels, proportions, mixing = single_sampling_step(
            key=self._rng.key,
            observed_data=self._observed_data,
            proportions=sample["proportions"],
            mixing=sample["mixing"],
            dirichlet_prior=self._dirichlet_prior,
            beta_prior=self._beta_prior,
        )

        return {
            "labels": labels,
            "proportions": proportions,
            "mixing": mixing,
        }

    def initialise(self) -> dict:
        """Initialises the sample. See `dimensions`
        for description."""
        n_features = self._observed_data.shape[1]
        n_clusters = self._dirichlet_prior.shape[0]

        proportions = random.dirichlet(self._rng.key, self._dirichlet_prior)
        mixing = random.beta(
            self._rng.key,
            self._beta_prior[0],
            self._beta_prior[1],
            shape=(n_features, n_clusters),
        )
        labels = sample_cluster_labels(
            self._rng.key,
            cluster_proportions=proportions,
            mixing=mixing,
            binary_codes=self._observed_data,
        )
        return {
            "labels": labels,
            "proportions": proportions,
            "mixing": mixing,
        }
