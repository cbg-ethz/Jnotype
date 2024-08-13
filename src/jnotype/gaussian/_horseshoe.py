"""Implements the sampler employing graphical horseshoe
prior as proposed by
Y. Li, B.A. Craig and A. Bhadra,
The graphical horseshoe estimator for inverse covariance matrices (2019)
"""

from typing import NamedTuple

from jaxtyping import Float, Array

import jax
import jax.random as jrandom
import jax.numpy as jnp

import jnotype.gaussian._numeric as num


class _HorseshoeRowSample(NamedTuple):
    """Internal sample of the last row of all three local arrays"""

    precision: Float[Array, " G"]
    lambda2: Float[Array, " G-1"]
    nu: Float[Array, " G-1"]


def _sample_inverse_gamma(
    key,
    shape: float,
    scale: Float[Array, " N"],
) -> Float[Array, " N"]:
    """Samples from the inverse gamma distribution.

    Args:
        key: JAX random key
        shape: shape parameter of the inverse gamma distribution
        scale: scale parameter of the inverse gamma distribution

    Note that:
        X ~ InvGamma(shape=a, scale=b)
    is equivalent to
        1/X ~ Gamma(shape=a, rate=b)
    """
    samples_gamma = jrandom.gamma(key, shape, shape=scale.shape)
    return scale * jnp.reciprocal(samples_gamma)


def sample_horseshoe_row(
    key,
    *,
    scatter_row: Float[Array, " G"],
    precision: Float[Array, "G G"],
    lambda2_row: Float[Array, " G-1"],
    nu_row: Float[Array, " G-1"],
    n_points: int,
    tau2: float,
    _jitter: float,
) -> _HorseshoeRowSample:
    """Samples the last row.

    Args:
        _jitter: a small numerical jitter to make the matrix inversion more stable
    """
    s12: Float[Array, " G-1"] = scatter_row[:-1]
    s22: float = scatter_row[-1]
    Gm1 = s12.shape[0]  # G - 1

    inv_omega11 = jnp.linalg.inv(precision[:-1, :-1] + _jitter * jnp.eye(Gm1))
    v12 = lambda2_row * tau2
    inv_C = s22 * inv_omega11 + jnp.diag(jnp.reciprocal(v12)) + _jitter * jnp.eye(Gm1)
    rate = 0.5 * s22

    key_omega, key_lambda, key_nu = jrandom.split(key, 3)

    omega12: Float[Array, " G"] = num.sample_precision_column(
        key=key_omega,
        inv_omega11=inv_omega11,
        inv_C=inv_C,
        scatter12=s12,
        n_samples=n_points,
        rate=rate,
    )

    lambda2: Float[Array, " G-1"] = _sample_inverse_gamma(
        key_lambda,
        shape=1,
        scale=jnp.reciprocal(nu_row) + 0.5 * jnp.square(omega12[:-1]) / tau2,
    )

    nu: Float[Array, " G-1"] = _sample_inverse_gamma(
        key_nu, shape=1.0, scale=1 + jnp.reciprocal(lambda2)
    )

    return _HorseshoeRowSample(
        precision=omega12,
        lambda2=lambda2,
        nu=nu,
    )


class _HorseshoeMatricesSample(NamedTuple):
    """Internal object representing the matrices."""

    precision: Float[Array, "G G"]
    lambda2: Float[Array, "G G"]
    nu: Float[Array, "G G"]

    @property
    def dim(self) -> int:
        return self.precision.shape[0]


def sample_precision_matrix_column_by_column(
    key,
    *,
    n_samples: int,
    scatter: Float[Array, "G G"],
    sample: _HorseshoeMatricesSample,
    tau2: float,
    _jitter: float,
) -> _HorseshoeMatricesSample:
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
        n_samples: number of data points
        _jitter: a small numerical jitter to make the matrix inversion more stable

    Returns:
        A precision matrix.
    """

    def update_column(carry: tuple, k: int) -> tuple:
        """Function sampling the `k`th column (row) and updating it.

        Args:
            carry: tuple (key, HorseshoeMatricesSample)
            k: the index of the column (row) to be updated
        """
        key = carry[0]
        matrices: _HorseshoeMatricesSample = carry[1]

        # Reorder the variables,
        # so that the updated column is the last one
        scatter_ = num.swap_with_last(scatter, k)
        precision = num.swap_with_last(matrices.precision, k)
        lambda2 = num.swap_with_last(matrices.lambda2, k)
        nu = num.swap_with_last(matrices.nu, k)

        # Sample the new last row/column
        key, subkey = jrandom.split(key)
        new_cols: _HorseshoeRowSample = sample_horseshoe_row(
            key=subkey,
            precision=precision,  # Full precision matrix
            scatter_row=scatter_[:, -1],
            lambda2_row=lambda2[:-1, -1],
            nu_row=nu[:-1, -1],
            n_points=n_samples,
            tau2=tau2,
            _jitter=_jitter,
        )

        # Update both the row and the column
        _LAST = -1
        precision = precision.at[:, _LAST].set(new_cols.precision)
        precision = precision.at[_LAST, :].set(new_cols.precision)

        lambda2 = lambda2.at[:-1, _LAST].set(new_cols.lambda2)
        lambda2 = lambda2.at[_LAST, :-1].set(new_cols.lambda2)

        nu = nu.at[:-1, _LAST].set(new_cols.nu)
        nu = nu.at[_LAST, :-1].set(new_cols.nu)

        # Reorder the variables to the original order
        precision = num.swap_with_last(precision, k)
        lambda2 = num.swap_with_last(lambda2, k)
        nu = num.swap_with_last(lambda2, k)

        new_matrices = _HorseshoeMatricesSample(
            precision=precision,
            lambda2=lambda2,
            nu=nu,
        )

        return (key, new_matrices), None

    carry, _ = jax.lax.scan(
        update_column,
        (key, sample),
        jnp.arange(sample.dim),
    )
    _, matrices = carry
    return matrices


class HorseshoeSample(NamedTuple):
    """Represents a full sample."""

    precision: Float[Array, "G G"]
    lambda2: Float[Array, "G G"]
    nu: Float[Array, "G G"]
    tau2: Float[Array, ""]
    xi: Float[Array, ""]

    @property
    def dim(self) -> int:
        return self.precision.shape[0]


def sample_horseshoe(
    key,
    scatter: Float[Array, "G G"],
    n_samples: int,
    sample: HorseshoeSample,
    _jitter: float = 1e-6,
) -> HorseshoeSample:
    key_matrices, key_tau, key_xi = jrandom.split(key, 3)

    new_matrices = sample_precision_matrix_column_by_column(
        key=key_matrices,
        n_samples=n_samples,
        scatter=scatter,
        sample=_HorseshoeMatricesSample(
            precision=sample.precision,
            lambda2=sample.lambda2,
            nu=sample.nu,
        ),
        tau2=sample.tau2,
        _jitter=_jitter,
    )

    G = sample.dim
    Gover2 = 0.5 * G * (G - 1)

    offset = 0.5 * jnp.sum(
        jnp.square(num.utzd_to_vector(new_matrices.precision))
        / num.utzd_to_vector(new_matrices.lambda2)
    )

    tau2 = _sample_inverse_gamma(
        key_tau,
        shape=0.5 * (1 + Gover2),
        scale=jnp.reciprocal(sample.xi) + offset,
    )

    xi = _sample_inverse_gamma(
        key_xi,
        shape=1,
        scale=1.0 + jnp.reciprocal(tau2),
    )

    return HorseshoeSample(
        precision=new_matrices.precision,
        lambda2=new_matrices.lambda2,
        nu=new_matrices.nu,
        tau2=tau2,
        xi=xi,
    )
