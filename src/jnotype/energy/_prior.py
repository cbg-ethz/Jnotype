"""Priors for interpretable energy-based models."""

import jax.numpy as jnp
from jaxtyping import Float, Array


def number_of_interactions_quadratic(G: int) -> int:
    """Number of interactions in a quadratic energy model,
    namely G over 2."""
    return G * (G - 1) // 2


def create_symmetric_interaction_matrix(
    diagonal: Float[Array, " G"],
    offdiagonal: Float[Array, " G*(G-1)//2"],
) -> Float[Array, " G G"]:
    """Generates a symmetric matrix out of one-dimensional
    diagonal and offdiagonal entries."""
    G = diagonal.shape[0]
    S = jnp.zeros((G, G), dtype=diagonal.dtype)
    i1, i2 = jnp.triu_indices(G, 1)

    S = S.at[i1, i2].set(offdiagonal)
    S = S + S.T
    return jnp.diag(diagonal) + S
