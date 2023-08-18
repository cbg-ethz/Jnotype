"""Cumulative shrinkage prior."""
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp

from jaxtyping import Float, Int, Array


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


def _log_pdf_multivariate_t(
    x: Float[Array, " K"],
    mask: Int[Array, " K"],
    *,
    dof: float,
    multiple: float,
) -> float:
    """Evaluates the log-PDF of the multivariate Student's t distribution
    at `x[mask]`.

    Assumes that the t distribution has the format:
    t(location=0, dispersion=multiple * identity, dof=dof)

    Args:
        x: position vector, with some possibly redundant coordinates
        mask: used to select the active coordinates
        dof: degrees of freedom
        multiple: positive scalar used to define the dispersion matrix
    """
    p = jnp.sum(mask)  # The effective dimension

    quadratic_form = jnp.sum(jnp.square(x * mask)) / (multiple * dof)

    log_contrib_quadratic_form = -0.5 * (p + dof) * jnp.log1p(quadratic_form)
    log_contrib_determinant = -0.5 * p * jnp.log(multiple)
    log_contrib_else = (
        jsp.special.gammaln(0.5 * (dof + p))
        - jsp.special.gammaln(0.5 * dof)
        - 0.5 * p * jnp.log(dof * jnp.pi)
    )

    return log_contrib_quadratic_form + log_contrib_determinant + log_contrib_else


def log_pdf_multivariate_t_cusp(
    x: Float[Array, " K"],
    mask: Int[Array, " K"],
    a: float,
    b: float,
) -> float:
    """Evaluates the log-PDF of the multivariate Student's t distribution
    at `x[mask]`, assuming that the prior on the slab part of the prior
    is InvGamma(a, b)."""
    return _log_pdf_multivariate_t(
        x=x,
        mask=mask,
        dof=2 * a,
        multiple=b / a,
    )
