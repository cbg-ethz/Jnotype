"""Common numeric utilities.

Notation: an UTZD stands for an "upper triangular with zero diagonal"
matrix. An UTZD matrix of shape `(G, G)` has therefore G*(G-1)/2
free parameters.
"""

import jax.numpy as jnp
import jax.random as jrandom

from jaxtyping import Float, Array, Num


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


def swap_with_last(A: Float[Array, "G G"], k: int) -> Float[Array, "G G"]:
    """For a symmetric matrix `A` swaps the `k`th column with the last one."""
    m = -1  # We swap with the last column
    A = A.at[[k, m], :].set(A[[m, k], :])  # Swap rows
    A = A.at[:, [k, m]].set(A[:, [m, k]])  # Swap columns
    return A


def sample_precision_column(
    key,
    inv_omega11: Float[Array, "G-1 G-1"],
    inv_C: Float[Array, "G-1 G-1"],
    scatter12: Float[Array, " G-1"],
    n_samples: int,
    rate: float,
) -> Float[Array, " G"]:
    """Samples the last column (row) using the factorization:
        Normal(first G-1 entries) x Gamma(last entry)

    Args:
        key: JAX random key
        inv_omega11: inverse of the (G-1) x (G-1) block
            of the precision matrix
        inv_C: inverse of the `C` matrix,
            i.e., the precision matrix of the first `G-1` entries
        scatter: column of the scatter matrix, shape (G,)
        n_samples: number of samples, which controls the
            shape parameter of the Gamma distribution
        rate: the rate parameter of the Gamma distribution

    Returns:
        Sampled column of length `G`.
    """
    # Invert `inv_C` to obtain the variance
    C = jnp.linalg.inv(inv_C)

    key_u, key_v = jrandom.split(key)

    u = jrandom.multivariate_normal(key_u, -C @ scatter12, C)

    shape = 1 + 0.5 * n_samples
    v = jrandom.gamma(key_v, shape) / rate

    new_omega22 = v + jnp.einsum("g,gh,h->", u, inv_omega11, u)

    return jnp.append(u, new_omega22)
