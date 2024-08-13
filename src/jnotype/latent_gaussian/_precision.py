"""Sparse precision matrix learning from Gaussian data.

This module implements the Gibbs sampler from

Hao Wang, "Scaling it up: Stochastic search structure
learning in graphical models", Bayesian Analysis (2015)
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

import numpyro.distributions as dist

from jaxtyping import Float, Array, Int

from jnotype.logistic._structure import _softmax_p1


def normal_logp(x: float, std: float) -> float:
    return dist.Normal(0.0, scale=std).log_prob(x)


def matrix_to_vector(matrix):
    # Get the indices for the upper-triangular part (excluding the diagonal)
    m = matrix.shape[0]
    upper_tri_indices = jnp.triu_indices(m, k=1)

    # Extract the upper-triangular elements and flatten them into a vector
    vector = matrix[upper_tri_indices]
    return vector


def vector_to_matrix(vector, m: int):
    # Create an empty m x m matrix of zeros
    matrix = jnp.zeros((m, m), dtype=vector.dtype)

    # Get the indices for the upper-triangular part (excluding the diagonal)
    upper_tri_indices = jnp.triu_indices(m, k=1)

    # Assign the vector values to the upper-triangular positions
    matrix = matrix.at[upper_tri_indices].set(vector)
    return matrix


def symmetrise_offdiagonal_upper_triangular(a):
    return a + a.T


def sample_indicators(
    key,
    precision: Float[Array, "G G"],
    pi: float,
    std0: float,
    std1: float,
) -> Int[Array, "G G"]:
    G = precision.shape[0]
    prec = matrix_to_vector(precision)

    logp_slab = normal_logp(prec, std1) + jnp.log(pi)
    logp_spike = normal_logp(prec, std0) + jnp.log1p(-pi)

    p_slab = _softmax_p1(log_p0=logp_spike, log_p1=logp_slab)
    indicators = jnp.asarray(jrandom.bernoulli(key, p=p_slab), dtype=int)
    
    a = vector_to_matrix(indicators, G)
    return symmetrise_offdiagonal_upper_triangular(a)


def generate_variance_matrix(
    indicators: Int[Array, "G G"],
    std0: float,
    std1: float,
) -> Float[Array, "G G"]:
    a = jnp.triu(
        indicators * jnp.square(std1) + (1 - indicators) * jnp.square(std0), k=1
    )
    return symmetrise_offdiagonal_upper_triangular(a)


def construct_scatter_matrix(y: Float[Array, "N G"]) -> Float[Array, "G G"]:
    return jnp.einsum("ng,nh->gh", y, y)


def sample_last_precision_column(
    key,
    precision: Float[Array, "G G"],
    scatter: Float[Array, "G G"],
    variances: Float[Array, "G G"],
    lambd: float,
    n: int,
) -> Float[Array, " G"]:
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
    def update_column(carry, k: int):
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
) -> dict:
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
