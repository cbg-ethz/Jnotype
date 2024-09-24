"""This module implements the Gibbs sampler from

Hao Wang, "Scaling it up: Stochastic search structure
learning in graphical models", Bayesian Analysis (2015)
"""

from typing import NewType, Optional, Sequence

import jax
import jax.numpy as jnp
import jax.random as jrandom

import numpyro.distributions as dist

from jaxtyping import Float, Array, Int

import jnotype.gaussian._numeric as num
from jnotype.logistic._structure import _softmax_p1
from jnotype.sampling import AbstractGibbsSampler, DatasetInterface
from jnotype._utils import JAXRNG


def _normal_logp(x: float, std: float) -> float:
    """Evaluates log-PDF of `N(0, std^2)`$ at `x`"""
    return dist.Normal(0.0, scale=std).log_prob(x)


def _sample_indicators(
    key,
    precision: Float[Array, "G G"],
    pi: float,
    std0: float,
    std1: float,
) -> Int[Array, "G G"]:
    """Samples the indicator matrix:

    Args:
        key: JAX random key
        precision: precision matrix of shape `(G, G)`
        pi: value between 0 and 1 controlling the sparsity
            (lower `pi` should result in sparser matrices)
        std0: standard deviation of the spike prior component
        std1: standard deviation of the slab prior component

    Returns:
        an indicator matrix of shape (G, G).
        Note that it is a *symmetric* matrix with zero diagonal.
    """
    G = precision.shape[0]
    prec = num.utzd_to_vector(precision)

    logp_slab = _normal_logp(prec, std1) + jnp.log(pi)
    logp_spike = _normal_logp(prec, std0) + jnp.log1p(-pi)

    p_slab = _softmax_p1(log_p0=logp_spike, log_p1=logp_slab)
    indicators = jnp.asarray(jrandom.bernoulli(key, p=p_slab), dtype=int)

    a = num.vector_to_utzd(indicators, G)
    return num.symmetrize_utzd(a)


def _generate_variance_matrix(
    indicators: Int[Array, "G G"],
    std0: float,
    std1: float,
) -> Float[Array, "G G"]:
    """Auxiliary function creating the variance matrix.

    Args:
        indicators: symmetric indicator matrix with zero diagonal
        std0: standard deviation of the spike prior component
        std1: standard deviation of the slab prior component
    """
    a = jnp.triu(
        indicators * jnp.square(std1) + (1 - indicators) * jnp.square(std0), k=1
    )
    return num.symmetrize_utzd(a)


def _sample_last_precision_column(
    key,
    precision: Float[Array, "G G"],
    scatter: Float[Array, "G G"],
    variances: Float[Array, "G G"],
    lambd: float,
    n: int,
) -> Float[Array, " G"]:
    """Samples the last column.

    Args:
        key: JAX random key
        precision: precision matrix
        scatter: the scatter matrix
        variances: variances matrix
                   (obtained using the latent indicators)
        lambd: penalisation on the diagonal entries.
               The larger `lambd`, the more shrinkage to 0 is encouraged.
        n: number of data points

    Returns:
        A sample from the conditional distribution of the last column (row)
        of the precision matrix.
    """
    inv_omega11 = jnp.linalg.inv(precision[:-1, :-1])  # (G-1, G-1)

    v12 = variances[-1, :-1]  # (G-1,)
    s12 = scatter[-1, :-1]  # (G-1,)
    s22: float = scatter[-1, -1]

    inv_C = (s22 + lambd) * inv_omega11 + jnp.diag(jnp.reciprocal(v12))
    rate = 0.5 * (s22 + lambd)

    return num.sample_precision_column(
        key,
        inv_omega11=inv_omega11,
        inv_C=inv_C,
        scatter12=s12,
        n_samples=n,
        rate=rate,
    )


def _sample_precision_matrix_column_by_column(
    key,
    precision: Float[Array, "G G"],
    scatter: Float[Array, "G G"],
    variances: Float[Array, "G G"],
    lambd: float,
    n: int,
) -> Float[Array, "G G"]:
    """Samples the precision matrix by sampling
       columns one after the other.

    Args:
        key: JAX random key
        precision: precision matrix
        scatter: the scatter matrix
        variances: variances matrix
                   (obtained using the latent indicators)
        lambd: penalisation on the diagonal entries.
               The larger `lambd`, the more shrinkage to 0 is encouraged.
        n: number of data points

    Returns:
        A precision matrix.
    """

    def update_column(carry: tuple, k: int) -> tuple:
        """Function sampling the `k`th column (row) and updating it.

        Args:
            carry: tuple (key, precision)
            k: the index of the column (row) to be updated
        """
        key, precision = carry

        # Reorder the variables
        precision = num.swap_with_last(precision, k)
        scatter_ = num.swap_with_last(scatter, k)
        variances_ = num.swap_with_last(variances, k)

        # Sample the new last row/column
        key, subkey = jrandom.split(key)
        new_col = _sample_last_precision_column(
            key=subkey,
            precision=precision,
            scatter=scatter_,
            variances=variances_,
            lambd=lambd,
            n=n,
        )
        # Update both the row and the column
        precision = precision.at[:, -1].set(new_col)
        precision = precision.at[-1, :].set(new_col)

        # Reorder the variables to the original order
        precision = num.swap_with_last(precision, k)

        return (key, precision), None

    carry, _ = jax.lax.scan(
        update_column,
        (key, precision),
        jnp.arange(precision.shape[0]),
    )
    _, precision = carry
    return precision


@jax.jit
def sample_indicators_and_precision(
    key,
    indicators: Int[Array, "G G"],
    precision: Float[Array, "G G"],
    scatter: Float[Array, "G G"],
    lambd: float,
    n_samples: int,
    pi: float,
    std0: float,
    std1: float,
) -> tuple[Int[Array, "G G"], Float[Array, "G G"]]:
    """Jointly samples indicator variables and precision matrix.

    Args:
        key: JAX random key
        indicators: current indicator matrix
        precision: current precision matrix
        scatter: the scatter matrix
        lambd: penalisation on the diagonal entries.
               The larger `lambd`, the more shrinkage to 0 is encouraged.
        n_samples: number of data points
        pi: value between 0 and 1 controlling the sparsity
            (lower `pi` should result in sparser matrices)
        std0: standard deviation of the spike prior component
        std1: standard deviation of the slab prior component

    Returns:
        indicators: symmetric 0-1 matrix of shape (G, G)
        precision: symmetric real matrix of shape (G, G)
    """
    subkey_indicators, subkey_precision = jrandom.split(key)

    indicators = _sample_indicators(
        key=subkey_indicators,
        precision=precision,
        pi=pi,
        std0=std0,
        std1=std1,
    )

    precision = _sample_precision_matrix_column_by_column(
        key=subkey_precision,
        precision=precision,
        scatter=scatter,
        variances=_generate_variance_matrix(
            indicators=indicators, std0=std0, std1=std1
        ),
        lambd=lambd,
        n=n_samples,
    )

    return indicators, precision


Sample = NewType("Sample", dict)


class PrecisionMatrixSpikeAndSlabSampler(AbstractGibbsSampler):
    """A Gibbs sampler to learn a precision matrix from centered (zero-mean)
    normally distributed data.
    """

    def __init__(
        self,
        datasets: Sequence[DatasetInterface],
        *,
        # Data
        data: Optional[Float[Array, "points features"]] = None,
        scatter_matrix: Optional[Float[Array, "features features"]] = None,
        n_points: Optional[int] = None,
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
    ) -> None:
        """

        Args:
            datasets: data sets in which the samples are stored
            data: data, assumed to be multivariate normal with zero mean.
                Shape (n_points, n_features)
            scatter_matrix: scatter matrix (can be provided instead of data).
                Shape (n_features, n_features)
            n_points: number of points in the data set
                (use only when the scatter matrix is provided)
            warmup: number of warmup steps in Gibbs sampling
            steps: number of Gibbs steps after the warmup
            verbose: whether the sampler should print out the sampling status
            seed: random seed
            deterministic_init: whether to use deterministic initialisation
            lambd: prior parameter which regularises the diagonal entries
            pi: prior parameter between 0 and 1 controlling the sparsity
                (lower `pi` should result in sparser matrices).
                By default (None) is set to `2 / (n_features - 1)`.
            std0: standard deviation of the spike prior component
            std1: standard deviation of the slab prior component
        """
        super().__init__(datasets, warmup=warmup, steps=steps, verbose=verbose)

        self._deterministic_init = deterministic_init

        # Initialize a random number generator
        self._jax_rng = JAXRNG(jax.random.PRNGKey(seed))

        scatter, n_points = num.prepare_data(
            data=data,
            scatter=scatter_matrix,
            n_points=n_points,
        )

        self._scatter_matrix: Float[Array, "features features"] = scatter
        self._n_points: int = n_points
        self._n_features: int = self._scatter_matrix.shape[0]

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

    @classmethod
    def dimensions(cls) -> Sample:
        """The sites in each sample with annotated dimensions."""
        return {
            "precision": ["features_dim0", "features_dim1"],
            "indicators": ["features_dim0", "features_dim1"],
        }

    def new_sample(self, sample: Sample) -> Sample:
        """A new sample."""
        indicators, precision = sample_indicators_and_precision(
            key=self._jax_rng.key,
            indicators=sample["indicators"],
            precision=sample["precision"],
            scatter=self._scatter_matrix,
            n_samples=self._n_points,
            lambd=self._lambd,
            pi=self._pi,
            std0=self._std0,
            std1=self._std1,
        )
        return {
            "precision": precision,
            "indicators": indicators,
        }

    def initialise(self) -> Sample:
        """Initialises the sample."""
        G = self._n_features

        if self._deterministic_init:
            return {
                "precision": jnp.eye(G, dtype=float),
                "indicators": jnp.zeros((G, G), dtype=int),
            }
        else:
            coinflips = jrandom.bernoulli(
                self._jax_rng.key, p=self._pi, shape=(G * (G - 1) // 2,)
            )
            return {
                "precision": jnp.eye(self._n_features)
                * (0.5 + jrandom.gamma(self._jax_rng.key, 1.0) / 0.5),
                "indicators": num.symmetrize_utzd(num.vector_to_utzd(coinflips, G)),
            }
