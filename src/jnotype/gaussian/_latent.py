"""Latent Gaussian graphical model for binary data."""

import jax
import jax.numpy as jnp
import jax.random as jrandom

import polyagamma as pg
from jaxtyping import Float, Array, Int

from typing import NewType, Optional, Sequence

import jnotype.gaussian._numeric as num
import jnotype.gaussian._spike_and_slab as sas
import numpy as np
from jnotype.sampling import AbstractGibbsSampler, DatasetInterface
from jnotype._utils import JAXRNG


_N_TRIALS: int = 1  # Use 1 for Bernoulli model


def _sample_polya_gamma(
    rng,
    mu: Float[Array, " G"],
    Z: Float[Array, "N G"],
) -> Float[Array, "N G"]:
    logits: Float[Array, "N G"] = Z + mu[None, :]

    # Use Pólya-Gamma sampler to sample auxiliary variables omega
    omega: Float[Array, "N G"] = jnp.asarray(
        pg.random_polyagamma(_N_TRIALS, z=logits, random_state=rng), dtype=float
    )
    return omega


def _sample_mu(
    key,
    Y: Int[Array, "N G"],
    Z: Float[Array, "N G"],
    polya_gamma: Float[Array, "N G"],
    sigma2: float,
):
    inv_var = jnp.reciprocal(sigma2) + jnp.sum(polya_gamma, axis=0)
    var: Float[Array, " G"] = jnp.reciprocal(inv_var)
    mu = var * jnp.sum(Y - 0.5 * _N_TRIALS - polya_gamma * Z, axis=0)

    G = Z.shape[1]
    norm = jrandom.normal(key, shape=(G,))

    return mu + jnp.sqrt(var) * norm


def _sample_Z_row(
    key,
    Y_row: Float[Array, " G"],
    mu: Float[Array, " G"],
    precision: Float[Array, "G G"],
    polya_gamma_row: Float[Array, " G"],
) -> Float[Array, " G"]:
    inv_V = precision + jnp.diag(polya_gamma_row)
    V: Float[Array, "G G"] = jnp.linalg.inv(inv_V)
    m: Float[Array, " G"] = V @ (Y_row - 0.5 * _N_TRIALS - polya_gamma_row * mu)

    return jrandom.multivariate_normal(key, mean=m, cov=V)


def _sample_Z(
    key,
    Y: Int[Array, "N G"],
    mu: Float[Array, " G"],
    precision: Float[Array, "G G"],
    polya_gamma: Float[Array, "N G"],
) -> Float[Array, "N G"]:
    N = Y.shape[0]

    keys = jrandom.split(key, N)

    return jax.vmap(_sample_Z_row, in_axes=(0, 0, None, None, 0))(
        keys, Y, mu, precision, polya_gamma
    )


@jax.jit
def _sample_mu_and_Z_fast(
    key,
    Y: Int[Array, "N G"],
    Z: Float[Array, "N G"],
    precision: Float[Array, "G G"],
    polya_gamma: Float[Array, "N G"],
    sigma2: float,
) -> tuple[Float[Array, " G"], Float[Array, "N G"]]:
    key_mu, key_Z = jrandom.split(key)

    mu = _sample_mu(
        key=key_mu,
        Y=Y,
        Z=Z,
        polya_gamma=polya_gamma,
        sigma2=sigma2,
    )

    Z = _sample_Z(
        key=key_Z,
        Y=Y,
        mu=mu,
        precision=precision,
        polya_gamma=polya_gamma,
    )

    return mu, Z


def _sample_mu_and_Z_with_polya_gamma(
    key,
    rng,
    Y: Int[Array, "N G"],
    mu: Float[Array, " G"],
    Z: Float[Array, "N G"],
    precision: Float[Array, "G G"],
    sigma2: float,
) -> tuple[Float[Array, " G"], Float[Array, "N G"]]:
    polya_gamma = _sample_polya_gamma(
        rng=rng,
        mu=mu,
        Z=Z,
    )
    return _sample_mu_and_Z_fast(
        key=key,
        Y=Y,
        Z=Z,
        precision=precision,
        polya_gamma=polya_gamma,
        sigma2=sigma2,
    )


Sample = NewType("Sample", dict)


class LatentGaussianModelSpikeAndSlabSampler(AbstractGibbsSampler):
    """A Gibbs sampler to learn a sparse precision matrix of a latent
    Gaussian model from binary data.
    """

    def __init__(
        self,
        datasets: Sequence[DatasetInterface],
        *,
        data: Int[Array, "N G"],
        # Gibbs sampling
        warmup: int = 5_000,
        steps: int = 10_000,
        verbose: bool = False,
        seed: int = 195,
        # Initialisation
        deterministic_init: bool = False,
        # Prior hyperparameters
        lambd: float = 1.0,
        pi: Optional[float] = None,
        std0: float = 0.1,
        std1: float = 1.0,
        mu_sigma2: float = 5.0**2,
    ) -> None:
        """

        Args:
            datasets: data sets in which the samples are stored
            data: binary data. Shape (n_points, n_features)
            warmup: number of warmup steps in Gibbs sampling
            steps: number of Gibbs steps after the warmup
            verbose: whether the sampler should print out the sampling status
            seed: random seed
            lambd: prior parameter which regularises the diagonal entries
            pi: prior parameter between 0 and 1 controlling the sparsity
                (lower `pi` should result in sparser matrices).
                By default (None) is set to `2 / (n_features - 1)`.
            std0: standard deviation of the spike prior component
            std1: standard deviation of the slab prior component
            mu_sigma2: variance of the normal prior on mu
        """
        super().__init__(datasets, warmup=warmup, steps=steps, verbose=verbose)

        # Initialize a random number generator
        self._jax_rng = JAXRNG(jax.random.PRNGKey(seed))
        self._numpy_rng = np.random.default_rng(101 * seed)

        self._data = data
        self._n_points: int = self._data.shape[0]
        self._n_features: int = self._data.shape[1]

        if lambd <= 0:
            raise ValueError(f"The lambd value has to be positive, but is {lambd}.")
        self._lambd = lambd

        if pi is None:
            pi = 2 / (self._n_features - 1)

        if pi <= 0 or pi >= 1:
            raise ValueError(
                f"The pi value has to be from the open inverval (0, 1), but is {pi}."
            )
        self._pi = pi

        if std0 <= 0:
            raise ValueError(f"The std0 has to be positive, but is {std0}.")
        if std1 <= 0:
            raise ValueError(f"The std1 has to be positive, but is {std1}.")
        self._std0 = std0
        self._std1 = std1

        if mu_sigma2 <= 0:
            raise ValueError(f"The mu_sigma2 has to be positive, but is {mu_sigma2}.")
        self._mu_sigma2 = mu_sigma2

    @classmethod
    def dimensions(cls) -> Sample:
        """The sites in each sample with annotated dimensions."""
        return {
            "precision": ["features_dim0", "features_dim1"],
            "indicators": ["features_dim0", "features_dim1"],
            "mu": ["features_dim0"],
            "latents": ["points", "features_dim0"],
        }

    def new_sample(self, sample: Sample) -> Sample:
        """A new sample."""

        # Sample indicators and precision
        scatter = num.construct_scatter_matrix(sample["latents"])

        indicators, precision = sas.sample_indicators_and_precision(
            key=self._jax_rng.key,
            indicators=sample["indicators"],
            precision=sample["precision"],
            scatter=scatter,
            n_samples=self._n_points,
            lambd=self._lambd,
            pi=self._pi,
            std0=self._std0,
            std1=self._std1,
        )

        mu, latents = _sample_mu_and_Z_with_polya_gamma(
            key=self._jax_rng.key,
            rng=self._numpy_rng,
            Y=self._data,
            mu=sample["mu"],
            Z=sample["latents"],
            precision=precision,
            sigma2=self._mu_sigma2,
        )

        return {
            "precision": precision,
            "indicators": indicators,
            "mu": mu,
            "latents": latents,
        }

    def initialise(self) -> Sample:
        """Initialises the sample."""
        N = self._n_points
        G = self._n_features

        coinflips = jrandom.bernoulli(
            self._jax_rng.key, p=self._pi, shape=(G * (G - 1) // 2,)
        )
        return {
            "precision": jnp.eye(self._n_features)
            * (0.5 + jrandom.gamma(self._jax_rng.key, 1.0) / 0.5),
            "indicators": num.symmetrize_utzd(num.vector_to_utzd(coinflips, G)),
            "mu": jnp.sqrt(self._mu_sigma2)
            * jrandom.normal(self._jax_rng.key, shape=(G,)),
            "latents": jrandom.normal(self._jax_rng.key, shape=(N, G)),
        }
