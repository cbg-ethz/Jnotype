"""Sampler for two-layer Bayesian pyramids
with cumulative shrinkage process (CSP) prior
on latent binary codes."""

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
from jnotype._csp import sample_csp_gibbs, sample_csp_prior
from jnotype._variance import sample_variances

Sample = NewType("Sample", dict)


_sample_csp_gibbs_jit = jax.jit(sample_csp_gibbs)


def _single_sampling_step(
    *,
    # Auxiliary: random keys, static specification
    jax_key: jax.Array,
    numpy_rng: np.random.Generator,
    n_binary_codes: int,
    # Observed values
    observed: Int[Array, "points observed"],
    # Sampled variables
    intercepts: Float[Array, " observed"],
    coefficients: Float[Array, "observed covariates"],
    structure: Int[Array, "observed covariates"],
    covariates: Float[Array, "points covariates"],
    observed_variances: Float[Array, " known_covariates"],
    gamma: Float[Array, ""],
    nu: Float[Array, " known_covariates"],
    cluster_labels: Int[Array, " points"],
    mixing: Float[Array, "n_binary_codes n_clusters"],
    proportions: Float[Array, " n_clusters"],
    csp_omega: Float[Array, " n_binary_codes"],
    csp_expected_occupied: float,
    # Priors
    dirichlet_prior: Float[Array, " n_clusters"],
    nu_prior_a: Union[float, Float[Array, " known_covariates"]],
    nu_prior_b: Union[float, Float[Array, " known_covariates"]],
    pseudoprior_variance: float,
    intercept_prior_mean: float,
    intercept_prior_variance: float,
    gamma_prior_a: float,
    gamma_prior_b: float,
    mixing_beta_prior: tuple[float, float],
    variances_prior_shape: float,
    variances_prior_scale: float,
    csp_theta_inf: float,
) -> Sample:
    """Single sampling step of two-layer Bayesian pyramid.

    Note:
        It cannot be JITed and it uses JAX random numbers
        as well as NumPy's. This is necessary to use
        the Pólya-Gamma sampler (coefficients).
    """
    # --- Sample variances for latent variables from the CSP prior ---
    key, subkey = jax.random.split(jax_key)
    csp_sample = _sample_csp_gibbs_jit(
        key=subkey,
        coefficients=coefficients[:, :n_binary_codes],
        structure=structure[:, :n_binary_codes],
        omega=csp_omega,
        expected_occupied=csp_expected_occupied,
        prior_shape=variances_prior_shape,
        prior_scale=variances_prior_scale,
        theta_inf=csp_theta_inf,
    )
    # --- Sample variances for observed variables from the usual prior ---
    key, subkey = jax.random.split(key)
    observed_variances = sample_variances(
        key=subkey,
        values=coefficients[:, n_binary_codes:],
        mask=structure[:, n_binary_codes:],
        prior_shape=variances_prior_shape,
        prior_scale=variances_prior_scale,
    )
    variances = jnp.concatenate((csp_sample["variance"], observed_variances))

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
        # Intercepts and coefficients
        "intercepts": intercepts,
        "coefficients_latent": coefficients[:, :n_binary_codes],
        "coefficients_observed": coefficients[:, n_binary_codes:],
        # Structure and sparsities
        "structure_latent": structure[:, :n_binary_codes],
        "structure_observed": structure[:, n_binary_codes:],
        "gamma": gamma,
        "nu": nu,
        # Latent traits
        "latent_traits": covariates[:, :n_binary_codes],
        # Clustering
        "cluster_labels": cluster_labels,
        "proportions": proportions,
        "mixing": mixing,
        # Variances:
        #   - Variances for observed covariates
        "observed_variances": observed_variances,
        #   - Variances for latent binary codes, using CSP prior
        "latent_variances": csp_sample["variance"],
        "csp_omega": csp_sample["omega"],
        "csp_nu": csp_sample["nu"],
        "csp_indicators": csp_sample["indicators"],
        "csp_active_traits": csp_sample["active_traits"],
        "csp_n_active_traits": csp_sample["n_active"],
    }


class TwoLayerPyramidSamplerNonparametric(AbstractGibbsSampler):
    """A prototype of a Gibbs sampler for a two-layer
    Bayesian pyramid with CSP prior.
    """

    def __init__(
        self,
        datasets: Sequence[DatasetInterface],
        *,
        # Observed data and dimension specification
        observed: Int[Array, "points features"],
        expected_binary_codes: float = 4,
        max_binary_codes: int = 8,
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
        inactive_latent_variance_theta_inf: float = 0.1**2,
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

        assert inactive_latent_variance_theta_inf > 0
        self._csp_theta_inf = inactive_latent_variance_theta_inf

        assert n_clusters >= 1
        self._n_clusters = n_clusters

        # We have number of binary codes modelled (maximum one)
        # and number of binary codes expected a priori, as the rest
        # will be marked as inactive.
        assert max_binary_codes >= 1
        self._n_binary_codes = max_binary_codes
        assert expected_binary_codes > 0
        self._kappa_0 = expected_binary_codes

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
    def dimensions(cls) -> Sample:
        """The sites in each sample with annotated dimensions."""
        return {
            "intercepts": ["features"],
            "coefficients_latent": ["features", "latents"],
            "coefficients_observed": ["features", "observed_covariates"],
            "structure_latent": ["features", "latents"],
            "structure_observed": ["features", "observed_covariates"],
            "gamma": [],  # Float, no named dimensions
            "nu": ["observed_covariates"],
            "latent_traits": ["points", "latents"],
            "cluster_labels": ["points"],
            "proportions": ["clusters"],
            "mixing": ["latents", "clusters"],
            # Obseved variances
            "observed_variances": ["observed_covariates"],
            # Latent variances are modelled using the CSP prior
            "latent_variances": ["latents"],
            "csp_omega": ["latents"],
            "csp_nu": ["latents"],
            "csp_indicators": ["latents"],
            "csp_active_traits": ["latents"],
            "csp_n_active_traits": [],  # Int, no named dimensions
        }

    def new_sample(self, sample: Sample) -> Sample:
        """A new sample."""
        coefficients = jnp.hstack(
            (sample["coefficients_latent"], sample["coefficients_observed"])
        )
        structure = jnp.hstack(
            (sample["structure_latent"], sample["structure_observed"])
        )
        covariates = jnp.hstack((sample["latent_traits"], self._observed_covariates))
        return _single_sampling_step(
            jax_key=self._jax_rng.key,
            numpy_rng=self._np_rng,
            n_binary_codes=self._n_binary_codes,
            observed=self._observed_data,
            intercepts=sample["intercepts"],
            coefficients=coefficients,
            structure=structure,
            covariates=covariates,
            observed_variances=sample["observed_variances"],
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
            csp_omega=sample["csp_omega"],
            csp_expected_occupied=self._kappa_0,
            csp_theta_inf=self._csp_theta_inf,
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

    def initialise(self) -> Sample:
        """Initialises the sample."""
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

        csp_sample = sample_csp_prior(
            self._jax_rng.key,
            k=self._n_binary_codes,
            expected_occupied=self._kappa_0,
            theta_inf=self._csp_theta_inf,
        )

        variances = jnp.concatenate(
            (csp_sample["variance"], jnp.ones(self._n_observed_covariates))
        )

        _noise = jax.random.normal(self._jax_rng.key, shape=(n_outputs, n_covariates))
        _entries_variances = variances[
            None, :
        ] * structure + self._pseudoprior_variance * (1 - structure)
        coefficients = jnp.sqrt(_entries_variances) * _noise

        return {
            "intercepts": self._initialise_intercepts(),
            "coefficients_latent": coefficients[:, : self._n_binary_codes],
            "coefficients_observed": coefficients[:, self._n_binary_codes :],
            "structure_latent": structure[:, : self._n_binary_codes],
            "structure_observed": structure[:, self._n_binary_codes :],
            "gamma": gamma,
            "nu": nu,
            "latent_traits": initial_binary_codes,
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
            "observed_variances": variances[self._n_binary_codes :],
            "latent_variances": csp_sample["variance"],
            "csp_omega": csp_sample["omega"],
            "csp_nu": csp_sample["nu"],
            "csp_indicators": csp_sample["indicators"],
            "csp_active_traits": csp_sample["active_traits"],
            "csp_n_active_traits": csp_sample["n_active"],
        }
