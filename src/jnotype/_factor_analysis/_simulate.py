"""Simulate data sets."""

from jaxtyping import Float, Array

import jax.numpy as jnp
from jax import random


def sample_observations(
    key,
    lambd: Float[Array, "observed latent"],
    eta: Float[Array, "points latent"],
    sigma2: Float[Array, " observed"],
) -> Float[Array, "points observed"]:
    """Samples observations.

    Args:
        key: JAX PRNG key
        lambd: Mixing matrix
        eta: Latent traits
        sigma2: noise variance for each observed dimension

    Returns:
        Y matrix, shape (n_points, n_observed)
    """
    # Shape (N, P)
    N = eta.shape[0]
    P = lambd.shape[0]
    noise = random.normal(key, shape=(N, P)) * jnp.sqrt(sigma2)[None, :]
    return jnp.einsum("PH,NH -> NP", lambd, eta) + noise


def sample_latent(key, points: int, latent: int) -> Float[Array, "points latent"]:
    """Samples latent traits.

    Args:
        key: JAX PRNG key
        points: Number of points
        latent: Number of latent traits
    """
    return random.normal(key, shape=(points, latent))


def sample_mixing(
    key, observed: int, theta: Float[Array, " latent"]
) -> Float[Array, "observed latent"]:
    """Samples the mixing matrix.

    Args:
        key: JAX PRNG key
        observed: Number of observed dimensions
        theta: variance of factors attributed to each latent trait
    """
    latent = theta.shape[0]
    lambd = random.normal(key, shape=(observed, latent))
    return lambd * jnp.sqrt(theta)[None, ...]


def covariance_from_mixing(
    lambd: Float[Array, "observed latent"],
    sigma2: Float[Array, " observed"],
) -> Float[Array, "observed observed"]:
    """Compute the covariance matrix of the observed variables
    using the mixing matrix
    and additional variance for each observed dimension."""
    return jnp.einsum("ph,qh -> pq", lambd, lambd) + jnp.diag(sigma2)
