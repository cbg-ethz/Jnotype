"""Sampling cluster labels and proportions."""
import time
from typing import Optional

import jax
import jax.numpy as jnp

from jax import random
from jaxtyping import Array, Float, Int


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
    key: random.PRNGKeyArray,
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
    counts = _calculate_counts(labels=labels, n_clusters=n_clusters)
    return random.dirichlet(key, alpha=counts + dirichlet_prior)


def sample_mixing(
    key: random.PRNGKeyArray,
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
    key: random.PRNGKeyArray,
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


def _log(msg: str) -> None:
    """TODO(Pawel): Replace with a logger."""
    print(msg)


def _init_params(
    key: random.PRNGKeyArray,
    dirichlet_prior: Float[Array, " B"],
    beta_prior: tuple[float, float],
    n_features: int,
    proportions: Optional[Float[Array, " B"]],
    mixing: Optional[Float[Array, "K B"]],
) -> tuple[Float[Array, " B"], Float[Array, "K B"],]:
    """Samples initial values for the parameters if they have not been defined."""
    key_prop, key_mixing = random.split(key)

    n_clusters = dirichlet_prior.shape[0]

    sampled_proportions = random.dirichlet(key_prop, dirichlet_prior)
    sampled_mixing = random.beta(
        key_mixing, beta_prior[0], beta_prior[1], shape=(n_features, n_clusters)
    )

    assert sampled_proportions.shape == (n_clusters,)
    assert sampled_mixing.shape == (n_features, n_clusters)

    proportions = sampled_proportions if proportions is None else proportions
    mixing = sampled_mixing if mixing is None else mixing

    assert proportions.shape == (n_clusters,)
    assert mixing.shape == (n_features, n_clusters)

    return proportions, mixing


def gibbs_sampler(
    *,
    key: random.PRNGKeyArray,
    observed_data: Int[Array, "N K"],
    dirichlet_prior: Float[Array, " B"],
    beta_prior: tuple[float, float] = (1.0, 1.0),
    n_samples: int = 5_000,
    thinning: int = 1,
    burnin: int = 1_000,
    _verbose: bool = False,
    _report_every: int = 1_000,
    proportions: Optional[Float[Array, " B"]] = None,
    mixing: Optional[Float[Array, "K B"]] = None,
) -> dict:
    """Gibbs sampler for Bernoulli mixture model.

    Args:
        key: JAX random key

    Returns:
        TODO(Pawel): xarray with the samples
    """
    key_init, key_burnin, key_sampling = random.split(key, 3)

    proportions, mixing = _init_params(
        key=key_init,
        dirichlet_prior=dirichlet_prior,
        beta_prior=beta_prior,
        proportions=proportions,
        mixing=mixing,
        n_features=observed_data.shape[1],
    )

    if _verbose:
        _log("Starting the burn-in phase sampling...")

    # Run burn in samples
    for key in random.split(key_burnin, burnin):
        _, proportions, mixing = single_sampling_step(
            key=key,
            observed_data=observed_data,
            proportions=proportions,
            mixing=mixing,
            dirichlet_prior=dirichlet_prior,
            beta_prior=beta_prior,
        )

    if _verbose:
        _log("Burn-in phase finished. Starting proper sampling...")

    t0 = time.time()

    n_steps = thinning * n_samples
    keys = random.split(key_sampling, n_steps)

    storage = {
        "labels": [],
        "proportions": [],
        "mixing": [],
    }

    for step, key in enumerate(keys, 1):
        labels, proportions, mixing = single_sampling_step(
            key=key,
            observed_data=observed_data,
            proportions=proportions,
            mixing=mixing,
            dirichlet_prior=dirichlet_prior,
            beta_prior=beta_prior,
        )

        # Save samples every `thinning` steps
        if step % thinning == 0:
            storage["labels"].append(labels)
            storage["proportions"].append(proportions)
            storage["mixing"].append(mixing)

        # Decide whether to print
        if _verbose and step % _report_every == 0:
            delta_t = time.time() - t0
            n_collected = len(storage["labels"])
            _log(
                f"In {delta_t:.2f} performed {step} steps "
                f"and collected {n_collected}/{n_samples} samples. "
                f"Current sampling speed: {n_collected/delta_t:.1f} samples/second."
            )

    # TODO(Pawel): Change the return type.
    return storage
