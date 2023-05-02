"""Gibbs sampler for Bayesian pyramids."""
import time

import numpy as np
import tqdm

import jax.numpy as jnp
from jax import random

from jaxtyping import Array, Float, Int

# Import sampling steps
from jnotype.bmm import sample_bmm
from jnotype.logistic import (
    sample_gamma,
    sample_structure,
    sample_binary_codes,
    sample_intercepts_and_coefficients,
)
from jnotype._variance import sample_variances


def single_sampling_step(
    *,
    # Auxiliary: random keys, static specification
    jax_key: random.PRNGKeyArray,
    numpy_rng: np.random.Generator,
    n_binary_codes: int,
    # Observed values
    observed: Int[Array, "points observed"],
    # Sampled variables
    intercepts: Float[Array, " observed"],
    coefficients: Float[Array, "observed covariates"],
    structure: Int[Array, "observed covariates"],
    covariates: Float[Array, "points covariates"],
    variances: Float[Array, " covariates"],
    gamma: Float[Array, ""],
    cluster_labels: Int[Array, " points"],
    mixing: Float[Array, "n_binary_codes n_clusters"],
    proportions: Float[Array, " n_clusters"],
    # Priors
    dirichlet_prior: Float[Array, " n_clusters"],
    pseudoprior_variance: float = 0.01,
    intercept_prior_mean: float = 0.0,
    intercept_prior_variance: float = 1.0,
    gamma_prior_a: float = 1.0,
    gamma_prior_b: float = 1.0,
    variances_prior_shape: float = 2.0,
    variances_prior_scale: float = 1.0,
    mixing_beta_prior: tuple[float, float] = (1.0, 1.0),
) -> dict:
    t0 = time.time()

    # --- Sample the sparse logistic regression layer ---
    # Sample intercepts and coefficients
    key, subkey = random.split(jax_key)
    intercepts, coefficients = sample_intercepts_and_coefficients(
        jax_key=subkey,
        numpy_rng=numpy_rng,
        intercepts=intercepts,
        coefficients=coefficients,
        structure=structure,
        covariates=covariates,
        variances=variances,
        pseudoprior_variance=pseudoprior_variance,
        intercept_prior_mean=intercept_prior_mean,
        intercept_prior_variance=intercept_prior_variance,
        observed=observed,
    )
    t1 = time.time()
    # print(f"Sampling intercepts and coefs: {t1-t0:.2f}")

    # Sample structure and the sparsity
    key, subkey_structure, subkey_gamma = random.split(key, 3)
    structure = sample_structure(
        key=subkey_structure,
        intercepts=intercepts,
        coefficients=coefficients,
        structure=structure,
        covariates=covariates,
        observed=observed,
        variances=variances,
        pseudoprior_variance=pseudoprior_variance,
        gamma=gamma,
    )
    gamma = sample_gamma(
        key=subkey_gamma,
        structure=structure,
        prior_a=gamma_prior_a,
        prior_b=gamma_prior_b,
    )

    t2 = time.time()
    # print(f"Sampling structure: {t2-t1:.2f}")

    # Sample prior variances for coefficients
    key, subkey = random.split(key)
    variances = sample_variances(
        key=subkey,
        values=coefficients,
        mask=structure,
        prior_shape=variances_prior_shape,
        prior_scale=variances_prior_scale,
    )

    t3 = time.time()
    # print(f"Sampling variances: {t3-t2:.2f}")

    # Sample binary latent variables
    key, subkey = random.split(key)
    covariates = sample_binary_codes(
        key=subkey,
        intercepts=intercepts,
        coefficients=coefficients,
        structure=structure,
        covariates=covariates,
        observed=observed,
        n_binary_codes=n_binary_codes,
        labels=cluster_labels,
        labels_to_codes=mixing,
    )

    t4 = time.time()
    # print(f"Sampling binary latent variables: {t4-t3:.2f}")

    # --- Sample the Bernoulli mixture model layer ---
    key, subkey = random.split(key)
    cluster_labels, proportions, mixing = sample_bmm(
        key=subkey,
        # Bernoulli mixture model sees the binary latent codes
        observed_data=covariates[:, :n_binary_codes],
        proportions=proportions,
        mixing=mixing,
        dirichlet_prior=dirichlet_prior,
        beta_prior=mixing_beta_prior,
    )

    t5 = time.time()
    # print(f"Sampling BMM: {t5-t4:.2f}")

    return {
        "intercepts": intercepts,
        "coefficients": coefficients,
        "structure": structure,
        "gamma": gamma,
        "variances": variances,
        "covariates": covariates,
        "cluster_labels": cluster_labels,
        "proportions": proportions,
        "mixing": mixing,
    }


class JAXRNG:
    def __init__(self, key: random.PRNGKeyArray) -> None:
        self._key = key

    @property
    def key(self) -> random.PRNGKeyArray:
        key, subkey = random.split(self._key)
        self._key = key
        return subkey


def _initialize(
    key: random.PRNGKeyArray,
    n_clusters: int,
    n_points: int,
    n_outputs: int,
    n_covariates: int,
    n_binary_codes: int,
):
    rng = JAXRNG(key)
    return {
        "intercepts": random.normal(rng.key, shape=(n_outputs,)),
        "coefficients": random.normal(rng.key, shape=(n_outputs, n_covariates)),
        "structure": random.bernoulli(rng.key, p=0.1, shape=(n_outputs, n_covariates)),
        "gamma": jnp.asarray(0.1),
        "variances": jnp.ones(n_covariates),
        "covariates": jnp.asarray(
            random.bernoulli(rng.key, p=0.5, shape=(n_points, n_covariates)),
            dtype=float,
        ),
        "cluster_labels": random.categorical(
            rng.key, logits=jnp.zeros(n_clusters), shape=(n_points,)
        ),
        "proportions": jnp.full(fill_value=1 / n_clusters, shape=(n_clusters,)),
        "mixing": random.beta(rng.key, a=1, b=1, shape=(n_binary_codes, n_clusters)),
    }


def do_steps(
    n_steps: int = 100,
    n_clusters: int = 4,
    n_points: int = 8_000,
    n_outputs: int = 200,
    n_covariates: int = 20,
    n_binary_codes: int = 5,
    burnin: int = 2,
):
    """Temporary method, will be removed/refactored into unit test.

    TODO(Pawel): refactor/remove.
    """

    rng = np.random.default_rng(32)
    key_init, key_observed, *keys = random.split(random.PRNGKey(12), n_steps + 2)
    sample = _initialize(
        key=key_init,
        n_binary_codes=n_binary_codes,
        n_clusters=n_clusters,
        n_points=n_points,
        n_outputs=n_outputs,
        n_covariates=n_covariates,
    )

    observed = random.bernoulli(key_observed, p=0.1, shape=(n_points, n_outputs))

    for _ in range(burnin):
        sample = single_sampling_step(
            jax_key=random.PRNGKey(10),  # TODO(Pawel): This is wrong
            numpy_rng=rng,
            n_binary_codes=n_binary_codes,
            observed=observed,
            intercepts=sample["intercepts"],
            coefficients=sample["coefficients"],
            structure=sample["structure"],
            covariates=sample["covariates"],
            variances=sample["variances"],
            gamma=sample["gamma"],
            cluster_labels=sample["cluster_labels"],
            mixing=sample["mixing"],
            proportions=sample["proportions"],
            dirichlet_prior=jnp.ones(n_clusters),
        )

    all_samples = []
    thinning = 10

    t0 = time.time()
    for i, key in tqdm.tqdm(enumerate(keys, 1), total=len(keys)):
        sample = single_sampling_step(
            jax_key=key,
            numpy_rng=rng,
            n_binary_codes=n_binary_codes,
            observed=observed,
            intercepts=sample["intercepts"],
            coefficients=sample["coefficients"],
            structure=sample["structure"],
            covariates=sample["covariates"],
            variances=sample["variances"],
            gamma=sample["gamma"],
            cluster_labels=sample["cluster_labels"],
            mixing=sample["mixing"],
            proportions=sample["proportions"],
            dirichlet_prior=jnp.ones(n_clusters),
        )

        if i % thinning == 0:
            all_samples.append(sample)

    t1 = time.time()
    delta_t = t1 - t0

    print(f"Executed {len(keys)} steps in {delta_t:.2f} seconds")
    print(f"Iteration speed: {len(keys) / delta_t:.2f} steps/second.")

    print(f"Collected {len(all_samples)} in {delta_t:.2f} seconds.")
    print(f"Sampling speed: {len(all_samples) / delta_t:.2f} samples/seconds.")


if __name__ == "__main__":
    do_steps()
