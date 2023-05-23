"""Sampler for two-layer Bayesian pyramids
with fixed number of latent binary codes."""
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxtyping import Array, Float, Int

from jnotype.sampling import AbstractGibbsSampler, DatasetInterface
from jnotype._utils import JAXRNG

from jnotype.bmm import sample_bmm
from jnotype.logistic import (
    sample_gamma,
    sample_structure,
    sample_binary_codes,
    sample_intercepts_and_coefficients,
)
from jnotype._variance import sample_variances


def _single_sampling_step(
    *,
    # Auxiliary: random keys, static specification
    jax_key: jax.random.PRNGKeyArray,
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
    """Single sampling step of two-layer Bayesian pyramid.

    Note:
        It cannot be JITed and it uses JAX random numbers
        as well as NumPy's. This is necessary to use
        the Pólya-Gamma sampler (coefficients).
    """
    # --- Sample the sparse logistic regression layer ---
    # Sample intercepts and coefficients
    key, subkey = jax.random.split(jax_key)
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

    # Sample structure and the sparsity
    key, subkey_structure, subkey_gamma = jax.random.split(key, 3)
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

    # Sample prior variances for coefficients
    key, subkey = jax.random.split(key)
    variances = sample_variances(
        key=subkey,
        values=coefficients,
        mask=structure,
        prior_shape=variances_prior_shape,
        prior_scale=variances_prior_scale,
    )

    # Sample binary latent variables
    key, subkey = jax.random.split(key)
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

    # --- Sample the Bernoulli mixture model layer ---
    key, subkey = jax.random.split(key)
    cluster_labels, proportions, mixing = sample_bmm(
        key=subkey,
        # Bernoulli mixture model sees the binary latent codes
        observed_data=covariates[:, :n_binary_codes],
        proportions=proportions,
        mixing=mixing,
        dirichlet_prior=dirichlet_prior,
        beta_prior=mixing_beta_prior,
    )

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


class TwoLayerPyramidSampler(AbstractGibbsSampler):
    """A prototype of a Gibbs sampler for a two-layer
    Bayesian pyramid.

    Current limitations:
      - No option to provide additional covariates.
      - Shrinkage on latent binary layer is not used.
    """

    def __init__(
        self,
        datasets: Sequence[DatasetInterface],
        *,
        # Observed data and dimension specification
        observed: Int[Array, "points features"],
        n_binary_codes: int = 8,
        n_clusters: int = 10,
        # Prior
        dirichlet_prior: Float[Array, " clusters"],
        gamma_prior: tuple[float, float] = (1.0, 1.0),
        variances_prior_scale: float = 2.0,
        variances_prior_shape: float = 1.0,
        pseudoprior_variance: float = 0.1**2,
        mixing_beta_prior: tuple[float, float] = (1.0, 5.0),
        intercept_prior_mean: float = -3,
        intercept_prior_variance: float = 1.0**2,
        # Gibbs sampling
        warmup: int = 5_000,
        steps: int = 10_000,
        verbose: bool = False,
        seed: int = 195,
    ) -> None:
        super().__init__(datasets, warmup=warmup, steps=steps, verbose=verbose)

        # Initialize two random number generators: we cannot just use JAX
        # here
        self._jax_rng = JAXRNG(jax.random.PRNGKey(seed))
        self._np_rng = np.random.default_rng(seed + 3)

        self._observed_data = observed
        assert n_clusters >= 1
        self._n_clusters = n_clusters
        assert n_binary_codes >= 1
        self._n_binary_codes = n_binary_codes

        self._dirichlet_prior = dirichlet_prior
        self._gamma_prior = gamma_prior

        # Variances of coefficients per each covariate in middle layer
        self._variances_prior_scale = variances_prior_scale
        self._variances_prior_shape = variances_prior_shape
        self._pseudoprior_variance = pseudoprior_variance

        self._mixing_beta_prior = mixing_beta_prior

        self._intercept_prior_mean = intercept_prior_mean
        self._intercept_prior_variance = intercept_prior_variance

    @classmethod
    def dimensions(cls) -> dict:
        """The sites in each sample with annotated dimensions."""
        return {
            "intercepts": ["features"],
            "coefficients": ["features", "latents"],
            "structure": ["features", "latents"],
            "gamma": [],  # Float, no named dimensions
            "variances": ["latents"],
            "covariates": ["points", "latents"],
            "cluster_labels": ["points"],
            "proportions": ["clusters"],
            "mixing": ["latent_codes", "clusters"],
        }

    def new_sample(self, sample: dict) -> dict:
        """A new sample."""

        return _single_sampling_step(
            jax_key=self._jax_rng.key,
            numpy_rng=self._np_rng,
            n_binary_codes=self._n_binary_codes,
            observed=self._observed_data,
            intercepts=sample["intercepts"],
            coefficients=sample["coefficients"],
            structure=sample["structure"],
            covariates=sample["covariates"],
            variances=sample["variances"],
            gamma=sample["gamma"],
            cluster_labels=sample["cluster_labels"],
            mixing=sample["mixing"],
            proportions=sample["proportions"],
            dirichlet_prior=self._dirichlet_prior,
            gamma_prior_a=self._gamma_prior[0],
            gamma_prior_b=self._gamma_prior[1],
            variances_prior_scale=self._variances_prior_scale,
            variances_prior_shape=self._variances_prior_shape,
            mixing_beta_prior=self._mixing_beta_prior,
            pseudoprior_variance=self._pseudoprior_variance,
            intercept_prior_mean=self._intercept_prior_mean,
            intercept_prior_variance=self._intercept_prior_variance,
        )

    def _initialise_intercepts(self) -> Float[Array, " covariates"]:
        """Initializes the intercepts."""
        n_outputs = self._observed_data.shape[1]

        mean = self._intercept_prior_mean
        std = np.sqrt(self._intercept_prior_variance)
        normal_noise = jax.random.normal(self._jax_rng.key, shape=(n_outputs,))
        return mean + std * normal_noise

    def _initialise_gamma(self) -> Float[Array, ""]:
        return jax.random.beta(
            self._jax_rng.key, self._gamma_prior[0], self._gamma_prior[1]
        )

    def initialise(self) -> dict:
        """Initialises the sample."""
        # TODO(Pawel): This initialisation can be much improved.
        #   Hopefully it does not matter in the end, but assessment of
        #   chain mixing is very much required.

        n_binary_codes = self._n_binary_codes
        # TODO(Pawel): Update when covariates are allowed
        n_covariates = self._n_binary_codes
        n_points, n_outputs = self._observed_data.shape
        n_clusters = self._n_clusters

        gamma = self._initialise_gamma()
        return {
            "intercepts": self._initialise_intercepts(),
            # TODO(Pawel): This doesn't really work this way:
            #   some coefficients should be sampled from pseudoprior
            #   and the others should be sampled from the prior
            "coefficients": jnp.sqrt(self._pseudoprior_variance)
            * jax.random.normal(self._jax_rng.key, shape=(n_outputs, n_covariates)),
            "structure": jax.random.bernoulli(
                self._jax_rng.key, p=gamma, shape=(n_outputs, n_covariates)
            ),
            "gamma": gamma,
            "variances": jnp.ones(n_covariates),
            # TODO(Pawel): Update when we allow additional covariates.
            "covariates": jnp.asarray(
                jax.random.bernoulli(
                    self._jax_rng.key, p=0.1, shape=(n_points, n_covariates)
                ),
                dtype=float,
            ),
            "cluster_labels": jax.random.categorical(
                self._jax_rng.key, logits=jnp.zeros(n_clusters), shape=(n_points,)
            ),
            "proportions": jnp.full(fill_value=1.0 / n_clusters, shape=(n_clusters,)),
            "mixing": jax.random.beta(
                self._jax_rng.key,
                a=self._mixing_beta_prior[0],
                b=self._mixing_beta_prior[1],
                shape=(n_binary_codes, n_clusters),
            ),
        }
