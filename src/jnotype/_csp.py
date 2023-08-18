"""Cumulative shrinkage prior."""
from jax import random
import jax.numpy as jnp

from jaxtyping import Float, Array


def prior_nu(key, alpha: float, h: int) -> Float[Array, " h"]:
    """Samples beta variables.

    Args:
        key: random key
        alpha: expected number of active components
        h: truncation

    Returns:
        array of length `h`, with the last entry set to 1.0,
          for the truncation purpose
    """
    nu = random.beta(key, 1.0, alpha, shape=(h,))
    return nu.at[h - 1].set(1.0)


def calculate_omega(nu: Float[Array, " h"]) -> Float[Array, " h"]:
    """Calculates weights from beta variables, in a stick-breaking fashion.

    Adapted from
    https://dirmeier.github.io/etudes/stick_breaking_constructions.html
    """
    one = jnp.ones((1,), dtype=float)
    # Product of (1-nu_k) for k < l.
    # For l = 0 we have just 1
    prods = jnp.concatenate([one, jnp.cumprod(1 - nu)[:-1]])
    return nu * prods


def calculate_pi_from_omega(omega: Float[Array, " h"]) -> Float[Array, " h"]:
    """Calculates shrinking probabilities.

    Args:
        omega: weights, shape (h,)

    Returns:
        shrinking probabilities, shape (h,)
    """
    return jnp.cumsum(omega)


def calculate_pi_from_nu(nu: jnp.ndarray) -> jnp.ndarray:
    """Calculates shrinking probabilities from beta variables.

    Note:
        Due to numerical issues, `pi` may marginally exceed 1.0.
    """
    return calculate_pi_from_omega(calculate_omega(nu))


def calculate_expected_active_from_pi(pi: jnp.ndarray) -> float:
    """Calculates expected number of active components from shrinking probabilities."""
    return jnp.sum(1.0 - pi)
