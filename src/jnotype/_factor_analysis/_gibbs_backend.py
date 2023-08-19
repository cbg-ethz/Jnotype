"""Sampling steps for all variables, apart from variances
attributed to latent traits, which are sampled with CSP module."""
from typing import Callable

import jax
import jax.numpy as jnp
from jax import random


from jaxtyping import Float, Array


def _v_j_generator(
    theta: Float[Array, " traits"],
    eta: Float[Array, "points traits"],
) -> Callable:
    """Generates a function which is used to sample
    covariance matrix, used to sample mixing matrix."""
    # Both arrays have shape (traits, traits)
    d_inv = jnp.diag(jnp.reciprocal(theta))
    eta_eta = eta.T @ eta

    def v_j(sigma2: float) -> Float[Array, "traits1 traits2"]:
        return jnp.linalg.inv(d_inv + eta_eta / sigma2)

    return v_j


def gibbs_sample_mixing(
    key,
    theta: Float[Array, " traits"],
    sigma2: Float[Array, " observed"],
    eta: Float[Array, "points traits"],
    Y: Float[Array, "points observed"],
) -> Float[Array, "observed traits"]:
    """Samples the mixing matrix.

    Args:
        key: JAX PRNG key
        theta: variances attributed to each latent trait
        sigma2: noise variance for each observed dimension
        eta: Latent traits
        Y: observed data

    Returns:
        mixing matrix, shape (observed, traits)
    """

    # Shape (observed, traits, traits)
    V = jax.vmap(_v_j_generator(theta, eta))(sigma2)

    temp1 = jnp.einsum("phk,nk->pnh", V, eta)
    temp2 = jnp.einsum("pnh,np->ph", temp1, Y)
    mu = temp2 / sigma2[:, None]  # Shape (observed, traits)

    subkeys = random.split(key, V.shape[0])
    return jax.vmap(random.multivariate_normal, in_axes=(0, 0, 0))(subkeys, mu, V)


def gibbs_sample_traits(
    key,
    lambd: Float[Array, "observed traits"],
    sigma2: Float[Array, " observed"],
    Y: Float[Array, "points observed"],
) -> Float[Array, "points traits"]:
    """Samples the latent traits."""
    # Shape (traits, traits)
    Sigma_inv = jnp.diag(jnp.reciprocal(sigma2))
    Id = jnp.eye(lambd.shape[1])
    V = jnp.linalg.inv(Id + lambd.T @ Sigma_inv @ lambd)

    # Shape (points, traits)
    mu = (V @ lambd.T @ Sigma_inv @ Y.T).T

    n_points = mu.shape[0]
    subkeys = random.split(key, n_points)

    return jax.vmap(random.multivariate_normal, in_axes=(0, 0, None))(subkeys, mu, V)
