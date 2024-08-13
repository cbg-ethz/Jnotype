"""Sparse precision matrix learning from Gaussian data.

This module implements the Gibbs sampler from

Hao Wang, "Scaling it up: Stochastic search structure
learning in graphical models", Bayesian Analysis (2015)

Notation: an UTZD stands for an "upper triangular with zero diagonal"
matrix. An UTZD matrix of shape `(G, G)` has therefore G*(G-1)/2
free parameters.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

import numpyro.distributions as dist

from jaxtyping import Float, Array, Int, Num

from jnotype.logistic._structure import _softmax_p1


def normal_logp(x: float, std: float) -> float:
    """Evaluates log-PDF of `N(0, std^2)`$ at `x`"""
    return dist.Normal(0.0, scale=std).log_prob(x)


def utzd_to_vector(matrix: Num[Array, "G G"]) -> Num[Array, " G*(G-1)/2"]:
    """Stores the free parameters of the UTZD matrix in a vector.

    See Also:
        `vector_to_utzd` for the (one-sided) inverse.
    """
    # Get the indices for the upper-triangular part (excluding the diagonal)
    m = matrix.shape[0]
    upper_tri_indices = jnp.triu_indices(m, k=1)

    # Extract the upper-triangular elements and flatten them into a vector
    vector = matrix[upper_tri_indices]
    return vector


def vector_to_utzd(vector: Num[Array, " m*(m-1)/2"], m: int) -> Num[Array, "m m"]:
    """Stores a vector `vector` as
    an upper triangular matrix with zero diagonal.

    See Also:
        `utzd_to_vector` for the (one-sided) inverse
    """
    # Create an empty m x m matrix of zeros
    matrix = jnp.zeros((m, m), dtype=vector.dtype)

    # Get the indices for the upper-triangular part (excluding the diagonal)
    upper_tri_indices = jnp.triu_indices(m, k=1)

    # Assign the vector values to the upper-triangular positions
    matrix = matrix.at[upper_tri_indices].set(vector)
    return matrix


def symmetrize_utzd(a: Num[Array, "G G"]) -> Num[Array, "G G"]:
    """Symmetrizes a UTZD matrix, by copying the entries
    to the lower diagonal.

    Note:
        Do not use this function for a general matrix as e.g., it may
        behave counterintuitively with respect to th diagonal.
    """
    return a + a.T


def sample_indicators(
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
    prec = utzd_to_vector(precision)

    logp_slab = normal_logp(prec, std1) + jnp.log(pi)
    logp_spike = normal_logp(prec, std0) + jnp.log1p(-pi)

    p_slab = _softmax_p1(log_p0=logp_spike, log_p1=logp_slab)
    indicators = jnp.asarray(jrandom.bernoulli(key, p=p_slab), dtype=int)

    a = vector_to_utzd(indicators, G)
    return symmetrize_utzd(a)


def generate_variance_matrix(
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
    return symmetrize_utzd(a)


def construct_scatter_matrix(y: Float[Array, "N G"]) -> Float[Array, "G G"]:
    """Constructs the scatter matrix of a data set, i.e.,

    $$S_{ij} = \\sum_{n=1^N} y_{ni}y_{nj}$$

    for $i, j=1, \\dotsc, G$.
    """
    return jnp.einsum("ng,nh->gh", y, y)


def sample_last_precision_column(
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
    C = jnp.linalg.inv(inv_C)

    key_u, key_v = jrandom.split(key)

    u = jrandom.multivariate_normal(key_u, -C @ s12, C)
    rate = 0.5 * (s22 + lambd)
    v = jrandom.gamma(key_v, 1 + 0.5 * n) / rate

    new_omega22 = v + jnp.einsum("g,gh,h->", u, inv_omega11, u)

    return jnp.append(u, new_omega22)


def swap_with_last(A: Float[Array, "G G"], k: int) -> Float[Array, "G G"]:
    """For a symmetric matrix `A` swaps the `k`th column with the last one."""
    m = -1  # We swap with the last column
    A = A.at[[k, m], :].set(A[[m, k], :])  # Swap rows
    A = A.at[:, [k, m]].set(A[:, [m, k]])  # Swap columns
    return A


def sample_precision_matrix_column_by_column(
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
        precision = swap_with_last(precision, k)
        scatter_ = swap_with_last(scatter, k)
        variances_ = swap_with_last(variances, k)

        # Sample the new last row/column
        key, subkey = jrandom.split(key)
        new_col = sample_last_precision_column(
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
        precision = swap_with_last(precision, k)

        return (key, precision), None

    carry, _ = jax.lax.scan(
        update_column,
        (key, precision),
        jnp.arange(precision.shape[0]),
    )
    _, precision = carry
    return precision


def sample_indicators_and_precision(
    key,
    indicators: Int[Array, "G G"],
    precision: Float[Array, "G G"],
    scatter: Float[Array, "G G"],
    lambd: float,
    n: int,
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
        n: number of data points
        pi: value between 0 and 1 controlling the sparsity
            (lower `pi` should result in sparser matrices)
        std0: standard deviation of the spike prior component
        std1: standard deviation of the slab prior component

    Returns:
        indicators: symmetric 0-1 matrix of shape (G, G)
        precision: symmetric real matrix of shape (G, G)
    """
    subkey_indicators, subkey_precision = jrandom.split(key)

    indicators = sample_indicators(
        key=subkey_indicators,
        precision=precision,
        pi=pi,
        std0=std0,
        std1=std1,
    )

    precision = sample_precision_matrix_column_by_column(
        key=subkey_precision,
        precision=precision,
        scatter=scatter,
        variances=generate_variance_matrix(indicators=indicators, std0=std0, std1=std1),
        lambd=lambd,
        n=n,
    )

    return indicators, precision
