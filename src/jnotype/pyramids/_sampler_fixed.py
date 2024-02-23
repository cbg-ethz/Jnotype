"""Sampler for two-layer Bayesian pyramids
with fixed number of latent binary codes."""

from typing import Optional, Sequence, Union, NewType

import jax
import jax.numpy as jnp
import numpy as np

from jaxtyping import Array, Float, Int

from jnotype.sampling import AbstractGibbsSampler, DatasetInterface
from jnotype._utils import JAXRNG

from jnotype.bmm import sample_bmm
from jnotype.logistic import (
    sample_gamma,
    sample_gamma_individual,
    sample_structure,
    sample_binary_codes,
    sample_intercepts_and_coefficients,
)
from jnotype._variance import sample_variances

_JointSample = NewType("_JointSample", dict)
_SplitSample = NewType("_SplitSample", dict)


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
    nu: Float[Array, " observed_covariates"],
    cluster_labels: Int[Array, " points"],
    mixing: Float[Array, "n_binary_codes n_clusters"],
    proportions: Float[Array, " n_clusters"],
    # Priors
    dirichlet_prior: Float[Array, " n_clusters"],
    nu_prior_a: Union[float, Float[Array, " observed_covariates"]] = 1.0,
    nu_prior_b: Union[float, Float[Array, " observed_covariates"]] = 1.0,
    pseudoprior_variance: float = 0.01,
    intercept_prior_mean: float = 0.0,
    intercept_prior_variance: float = 1.0,
    gamma_prior_a: float = 1.0,
    gamma_prior_b: float = 1.0,
    variances_prior_shape: float = 2.0,
    variances_prior_scale: float = 1.0,
    mixing_beta_prior: tuple[float, float] = (1.0, 1.0),
) -> _JointSample:
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
    key, subkey_structure, subkey_gamma, subkey_nu = jax.random.split(key, 4)

    sparsity_vector = jnp.concatenate(
        (
            jnp.full(shape=(n_binary_codes,), fill_value=gamma),
            nu,
        )
    )
    structure = sample_structure(
        key=subkey_structure,
        intercepts=intercepts,
        coefficients=coefficients,
        structure=structure,
        covariates=covariates,
        observed=observed,
        variances=variances,
        pseudoprior_variance=pseudoprior_variance,
        gamma=sparsity_vector,
    )
    gamma = sample_gamma(
        key=subkey_gamma,
        structure=structure[..., :n_binary_codes],
        prior_a=gamma_prior_a,
        prior_b=gamma_prior_b,
    )
    nu = sample_gamma_individual(
        key=subkey_nu,
        structure=structure[..., n_binary_codes:],
        prior_a=nu_prior_a,
        prior_b=nu_prior_b,
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
        "nu": nu,
        "variances": variances,
        "covariates": covariates,
        "cluster_labels": cluster_labels,
        "proportions": proportions,
        "mixing": mixing,
    }


def _split_sample(n_binary_codes: int, sample: _JointSample) -> _SplitSample:
    return {
        "intercepts": sample["intercepts"],
        "coefficients_latent": sample["coefficients"][:, :n_binary_codes],
        "coefficients_observed": sample["coefficients"][:, n_binary_codes:],
        "structure_latent": sample["structure"][:, :n_binary_codes],
        "structure_observed": sample["structure"][:, n_binary_codes:],
        "gamma": sample["gamma"],
        "nu": sample["nu"],
        "latent_variances": sample["variances"][:n_binary_codes],
        "observed_variances": sample["variances"][n_binary_codes:],
        "latent_traits": sample["covariates"][:, :n_binary_codes],
        "cluster_labels": sample["cluster_labels"],
        "proportions": sample["proportions"],
        "mixing": sample["mixing"],
    }


def _merge_sample(
    sample: _SplitSample,
    observed_covariates: Float[Array, "points observed_covariates"],
) -> _JointSample:
    return {
        "intercepts": sample["intercepts"],
        "coefficients": jnp.hstack(
            (sample["coefficients_latent"], sample["coefficients_observed"])
        ),
        "structure": jnp.hstack(
            (sample["structure_latent"], sample["structure_observed"])
        ),
        "gamma": sample["gamma"],
        "nu": sample["nu"],
        "variances": jnp.concatenate(
            (sample["latent_variances"], sample["observed_variances"])
        ),
        "covariates": jnp.hstack((sample["latent_traits"], observed_covariates)),
        "cluster_labels": sample["cluster_labels"],
        "proportions": sample["proportions"],
        "mixing": sample["mixing"],
    }


class TwoLayerPyramidSampler(AbstractGibbsSampler):
    """A prototype of a Gibbs sampler for a two-layer
    Bayesian pyramid.

    Current limitations:
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
        observed_covariates: Optional[
            Float[Array, "points observed_covariates"]
        ] = None,
        # Prior
        dirichlet_prior: Float[Array, " clusters"],
        gamma_prior: tuple[float, float] = (1.0, 1.0),
        nu_prior: tuple[float, float] = (1.0, 1.0),
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
        self._observed_covariates = (
            observed_covariates
            if observed_covariates is not None
            else jnp.zeros((observed.shape[0], 0))
        )
        self._n_observed_covariates = self._observed_covariates.shape[1]

        assert n_clusters >= 1
        self._n_clusters = n_clusters
        assert n_binary_codes >= 1
        self._n_binary_codes = n_binary_codes

        self._dirichlet_prior = dirichlet_prior
        self._gamma_prior = gamma_prior
        self._nu_prior = nu_prior

        # Variances of coefficients per each covariate in middle layer
        self._variances_prior_scale = variances_prior_scale
        self._variances_prior_shape = variances_prior_shape
        self._pseudoprior_variance = pseudoprior_variance

        self._mixing_beta_prior = mixing_beta_prior

        self._intercept_prior_mean = intercept_prior_mean
        self._intercept_prior_variance = intercept_prior_variance

    @classmethod
    def dimensions(cls) -> _SplitSample:
        """The sites in each sample with annotated dimensions."""
        return {
            "intercepts": ["features"],
            "coefficients_latent": ["features", "latents"],
            "coefficients_observed": ["features", "observed_covariates"],
            "structure_latent": ["features", "latents"],
            "structure_observed": ["features", "observed_covariates"],
            "gamma": [],  # Float, no named dimensions
            "nu": ["observed_covariates"],
            "latent_variances": ["latents"],
            "observed_variances": ["observed_covariates"],
            "latent_traits": ["points", "latents"],
            "cluster_labels": ["points"],
            "proportions": ["clusters"],
            "mixing": ["latents", "clusters"],
        }

    def new_sample(self, sample: _SplitSample) -> _SplitSample:
        """A new sample."""
        sample: _JointSample = _merge_sample(
            sample=sample, observed_covariates=self._observed_covariates
        )
        new_sample: _JointSample = _single_sampling_step(
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
            nu=sample["nu"],
            cluster_labels=sample["cluster_labels"],
            mixing=sample["mixing"],
            proportions=sample["proportions"],
            dirichlet_prior=self._dirichlet_prior,
            gamma_prior_a=self._gamma_prior[0],
            gamma_prior_b=self._gamma_prior[1],
            nu_prior_a=self._nu_prior[0],
            nu_prior_b=self._nu_prior[1],
            variances_prior_scale=self._variances_prior_scale,
            variances_prior_shape=self._variances_prior_shape,
            mixing_beta_prior=self._mixing_beta_prior,
            pseudoprior_variance=self._pseudoprior_variance,
            intercept_prior_mean=self._intercept_prior_mean,
            intercept_prior_variance=self._intercept_prior_variance,
        )
        return _split_sample(n_binary_codes=self._n_binary_codes, sample=new_sample)

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

    def _initialise_nu(self) -> Float[Array, ""]:
        return jax.random.beta(
            self._jax_rng.key,
            self._nu_prior[0],
            self._nu_prior[1],
            shape=(self._n_observed_covariates,),
        )

    def _initialise_structure(
        self, gamma: Float[Array, ""], nu: Float[Array, " observed_covariates"]
    ) -> Int[Array, ""]:
        """Initialises the structure."""
        n_outputs = self._observed_data.shape[1]
        structure_codes = jax.random.bernoulli(
            self._jax_rng.key, p=gamma, shape=(n_outputs, self._n_binary_codes)
        )
        structure_observed_features = jax.random.bernoulli(
            self._jax_rng.key, p=nu, shape=(n_outputs, self._n_observed_covariates)
        )
        return jnp.hstack((structure_codes, structure_observed_features))

    def _initialise_full_sample(self) -> _JointSample:
        # TODO(Pawel): This initialisation can be much improved.
        #   Hopefully it does not matter in the end, but assessment of
        #   chain mixing is very much required.

        n_covariates = self._n_binary_codes + self._n_observed_covariates
        n_points, n_outputs = self._observed_data.shape
        n_clusters = self._n_clusters

        gamma = self._initialise_gamma()
        nu = self._initialise_nu()

        initial_binary_codes = jnp.asarray(
            # TODO(Pawel): There could be a better way to initialize binary codes
            jax.random.bernoulli(
                self._jax_rng.key, p=0.1, shape=(n_points, self._n_binary_codes)
            ),
            dtype=float,
        )
        initial_covariates = jnp.hstack(
            (initial_binary_codes, self._observed_covariates)
        )

        structure = self._initialise_structure(gamma=gamma, nu=nu)
        variances = jnp.ones(n_covariates)

        _noise = jax.random.normal(self._jax_rng.key, shape=(n_outputs, n_covariates))
        _entries_variances = variances[
            None, :
        ] * structure + self._pseudoprior_variance * (1 - structure)
        coefficients = jnp.sqrt(_entries_variances) * _noise

        return {
            "intercepts": self._initialise_intercepts(),
            "coefficients": coefficients,
            "structure": structure,
            "gamma": gamma,
            "nu": nu,
            "variances": variances,
            "covariates": initial_covariates,
            "cluster_labels": jax.random.categorical(
                self._jax_rng.key, logits=jnp.zeros(n_clusters), shape=(n_points,)
            ),
            "proportions": jnp.full(fill_value=1.0 / n_clusters, shape=(n_clusters,)),
            "mixing": jax.random.beta(
                self._jax_rng.key,
                a=self._mixing_beta_prior[0],
                b=self._mixing_beta_prior[1],
                shape=(self._n_binary_codes, n_clusters),
            ),
        }

    def initialise(self) -> _SplitSample:
        """Initialises the sample."""
        return _split_sample(
            n_binary_codes=self._n_binary_codes, sample=self._initialise_full_sample()
        )
