"""Cumulative shrinkage prior."""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random

from jaxtyping import Float, Int, Array

from jnotype._variance import sample_variances, sample_inverse_gamma

# ----- Beta variables -----


def sample_prior_nu(key, alpha: float, h: int) -> Float[Array, " h"]:
    """Samples beta variables from the prior.

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


def _calculate_nu_posterior_coefficients(
    zs: jnp.ndarray, alpha: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    K = zs.shape[0]
    coeffs1 = jnp.ones(K - 1, dtype=float)
    coeffs2 = jnp.full(shape=coeffs1.shape, fill_value=alpha, dtype=coeffs1.dtype)

    def body_fun(carry, z):
        coeffs1, coeffs2 = carry
        lt_mask = jnp.arange(K - 1) < z
        eq_mask = jnp.arange(K - 1) == z
        coeffs1 = coeffs1 + eq_mask
        coeffs2 = coeffs2 + lt_mask
        return (coeffs1, coeffs2), ()

    (coeffs1, coeffs2), _ = jax.lax.scan(body_fun, (coeffs1, coeffs2), zs)
    return coeffs1, coeffs2


def sample_posterior_nu(
    key, zs: Float[Array, " K"], alpha: float
) -> Float[Array, " K"]:
    """Samples beta variables from the posterior,
    conditioned on the latent indicators.

    Args:
        zs: latent indicators, shape (K,)
          with values {0, 1, ..., K-1}
        alpha: sparsity parameter

    Returns:
        array of length `K` with sampled `nu` vector.
          The last entry set to 1.0,
    """
    coeffs1, coeffs2 = _calculate_nu_posterior_coefficients(zs, alpha)
    nus = random.beta(key, coeffs1, coeffs2)
    return jnp.append(nus, 1.0)


# ----- Omega variables -----


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


# ----- Shrinking probabilities -----


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


def log_pdf_multivariate_normal(
    x: Float[Array, " K"],
    mask: Int[Array, " K"],
    multiple: float,
) -> float:
    """Evaluates the log-PDF of the multivariate normal distribution
    at `x[mask]`."""
    p = jnp.sum(mask)  # The effective dimension
    quadratic_form = -0.5 * jnp.sum(jnp.square(x * mask)) / multiple
    log_else = -0.5 * p * jnp.log(multiple * 2 * jnp.pi)
    return quadratic_form + log_else


# ----- Indicator variables -----


def calculate_indicator_logits(
    coefficients: Float[Array, "features codes"],
    structure: Int[Array, "features codes"],
    omega: Float[Array, " values"],
    a: float = 2.0,
    b: float = 1.0,
    theta_inf: float = 0.01,
) -> Float[Array, "codes values"]:
    """Calculates log-probabilities
    of the indicators.

    Args:
        coefficients: shape (coefficients, traits)
        structure: shape (coefficients, traits)
        omega: shape (traits,)
        a: parameter of inverse gamma prior
        b: parameter of inverse gamma prior
        theta_inf: vanishing variance

    Returns:
        logits, shape (codes, values).
          It should be used to sample (codes,)
          vector with logprobabilities of (values,)
    """
    n_codes, n_values = coefficients.shape[1], coefficients.shape[1]

    coefficients_normal = jax.vmap(log_pdf_multivariate_normal, in_axes=(1, 1, None))(
        coefficients, structure, jnp.asarray(theta_inf)
    )

    coefficients_t = jax.vmap(log_pdf_multivariate_t_cusp, in_axes=(1, 1, None, None))(
        coefficients, structure, jnp.asarray(a), jnp.asarray(b)
    )

    mask = jnp.less_equal(jnp.arange(n_values), jnp.arange(n_codes)[:, jnp.newaxis])
    normal_2d = jnp.expand_dims(coefficients_normal, axis=-1)
    student_2d = jnp.expand_dims(coefficients_t, axis=-1)
    semilogits = jnp.where(mask, normal_2d, student_2d)  # Shape (codes, values)

    # Now we need to add log(omegas)
    return semilogits + jnp.log(omega)[None, :]


def sample_indicators(
    key,
    logits: Float[Array, "codes values"],
) -> Int[Array, " codes"]:
    """Samples the latent indicator variables.

    Args:
        logits, shape (codes, values)

    Returns:
        indicators, shape (codes,).
          Each entry is in {0, 1, ..., values-1}
    """
    return random.categorical(key, logits, axis=1)


# ----- Sampling variances -----


def _select_variances_active(
    indicators: Int[Array, " codes"],
    variances_active: Float[Array, " codes"],
    theta_inf: float,
) -> Float[Array, " codes"]:
    variances_inactive = jnp.full(shape=variances_active.shape, fill_value=theta_inf)
    # If zs[h] <= h, the feature is inactive. Otherwise, it is active.
    inactive = jnp.less_equal(indicators, jnp.arange(indicators.shape[0]))
    return jnp.where(inactive, variances_inactive, variances_active)


def compute_active_traits(indicators: Int[Array, " codes"]) -> Int[Array, " codes"]:
    """Annotates with 1 which traits are active."""
    active = jnp.greater(indicators, jnp.arange(indicators.shape[0]))
    return jnp.asarray(active, dtype=int)


def _sample_variances_conditioned_on_indicators(
    key,
    indicators: Int[Array, " codes"],
    coefficients: Float[Array, "features codes"],
    structure: Int[Array, "features codes"],
    prior_shape: float,
    prior_scale: float,
    theta_inf: float,
) -> Float[Array, " codes"]:
    variances_active = sample_variances(
        key,
        values=coefficients,
        mask=structure,
        prior_shape=prior_shape,
        prior_scale=prior_scale,
    )

    return _select_variances_active(
        indicators=indicators,
        variances_active=variances_active,
        theta_inf=theta_inf,
    )


def sample_csp_prior(
    key,
    k: int,
    expected_occupied: float,
    prior_shape: float = 2.0,
    prior_scale: float = 1.0,
    theta_inf: float = 1e-2,
) -> dict:
    """One sample from the prior to initialize."""
    key_nu, key_indicators, key_var = random.split(key, 3)
    nus = sample_prior_nu(key_nu, alpha=expected_occupied, h=k)
    omega = calculate_omega(nus)
    indicators = random.categorical(key_indicators, jnp.log(omega), shape=(k,))

    variances_active = sample_inverse_gamma(
        key=key_var,
        n_points=k,
        a=prior_shape,
        b=prior_scale,
    )

    variance = _select_variances_active(
        indicators=indicators,
        variances_active=variances_active,
        theta_inf=theta_inf,
    )

    return {
        "variance": variance,
        "nu": nus,
        "omega": omega,
        "indicators": indicators,
        "active_traits": compute_active_traits(indicators),
        "n_active": jnp.sum(compute_active_traits(indicators)),
    }


def sample_csp_gibbs(
    key,
    coefficients: Float[Array, "features codes"],
    structure: Int[Array, "features codes"],
    omega: Float[Array, " codes"],
    expected_occupied: float,
    prior_shape: float = 2.0,
    prior_scale: float = 1.0,
    theta_inf: float = 1e-2,
) -> dict:
    """One sample from the posterior."""
    key_indicator, key_nu, key_var = random.split(key, 3)

    logits = calculate_indicator_logits(
        coefficients=coefficients,
        structure=structure,
        omega=omega,
        a=prior_shape,
        b=prior_scale,
        theta_inf=theta_inf,
    )
    indicators = sample_indicators(key_indicator, logits)

    nus = sample_posterior_nu(key_nu, zs=indicators, alpha=expected_occupied)
    omega = calculate_omega(nus)

    variance = _sample_variances_conditioned_on_indicators(
        key_var,
        indicators=indicators,
        coefficients=coefficients,
        structure=structure,
        prior_shape=prior_shape,
        prior_scale=prior_scale,
        theta_inf=theta_inf,
    )

    return {
        "variance": variance,
        "nu": nus,
        "omega": omega,
        "indicators": indicators,
        "active_traits": compute_active_traits(indicators),
        "n_active": jnp.sum(compute_active_traits(indicators)),
    }
