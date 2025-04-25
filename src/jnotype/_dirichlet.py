"""The Dirichlet process/Dirichlet-multinomial model."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import jnotype._reparam as reparam
from functools import partial
from typing import Callable, Sequence, TypeVar
from jaxtyping import Float, Array


_Float = Float[Array, " "]
_Param = TypeVar("_Param")


def log_factorial(x):
    return gammaln(x + 1)


def log_multinomial(x, log_p):
    n = jnp.sum(x)
    return log_factorial(n) - jnp.sum(log_factorial(x)) + jnp.sum(x * log_p)


def construct_multinomial_loglikelihood(
    data: Sequence[reparam.BinaryArray],
    loglikelihood: Callable[[_Param, reparam.BinaryArray], float],
) -> Callable[[_Param], float]:
    """Binds the loglikelihood function to the data set
    in the multinomial model.

    Args:
        data: binary array of shape (n_samples, n_loci)
        loglikelihood: function of signature (params, genotype) -> float
            calculating the loglikelihood for a single data point

    Returns:
        loglikelihood function for the whole data set of signature
            params -> float
    """
    dataset = reparam.empirical_binary_vector_distribution(data)

    def f(param: _Param) -> _Float:
        """Function to be returned."""
        ll = partial(loglikelihood, param)
        logps = jax.vmap(ll)(dataset.atoms)
        return log_multinomial(dataset.counts, logps)

    return f


def _log_gamma_ratio(log_a, k):
    """Numerically stable  log Γ(exp(log_a)+k) - log Γ(exp(log_a)),
    computed in log-space as  Σ_{j=0}^{k-1} log(exp(log_a)+j)
    using log-add-exp trick.
    """
    k = jnp.asarray(k, dtype=jnp.int32)

    def body(i, acc):
        i_f = i.astype(log_a.dtype)
        return acc + jnp.logaddexp(log_a, jnp.log(i_f))

    return jax.lax.fori_loop(0, k, body, 0.0)


# vectorised version for a whole batch of counts
_vmap_gamma_ratio = jax.vmap(_log_gamma_ratio)


def log_dirichlet_multinomial(x, log_p, log_alpha):
    """
    Log-pmf of Dirichlet–multinomial with concentration α_i = exp(log_alpha + log_p_i),
    implemented fully in log-space for numerical stability.

    Args:
        x: observed counts, integer array of shape (K,)
        log_p: log-probabilities, float array of shape (K,)
        log_alpha: scalar representing log(total concentration)

    Return:
        log_prob: log-likelihood
    """
    x = jnp.asarray(x)
    log_p = jnp.asarray(log_p)
    log_alpha = jnp.asarray(log_alpha)

    n = jnp.sum(x, axis=-1)

    # Multinomial coefficient term
    log_coeff = log_factorial(n) - jnp.sum(log_factorial(x), axis=-1)

    # Dirichlet normaliser: log Γ(α₀) – log Γ(α₀ + n)
    log_alpha0_term = -_log_gamma_ratio(log_alpha, n.astype(jnp.int32))

    # Component-wise ratio: Σ_i [log Γ(α_i + x_i) – log Γ(α_i)]
    log_ai = log_alpha[..., None] + log_p
    # Flatten, vmap, then restore (..., K) shape
    ratio_terms = _vmap_gamma_ratio(log_ai.reshape(-1), x.reshape(-1)).reshape(x.shape)
    log_ratio = jnp.sum(ratio_terms, axis=-1)

    return log_coeff + log_alpha0_term + log_ratio


def construct_dirichlet_multinomial_loglikelihood(data, loglikelihood):
    """Binds the loglikelihood function to the data set
    in the Dirichlet-multinomial model.

    Args:
        data: binary array of shape (n_samples, n_loci)
        loglikelihood: function of signature (params, genotype) -> float
            calculating the loglikelihood for a single data point

    Returns:
        loglikelihood function for the whole data set
          of signature (params, alpha) -> float
    """
    dataset = reparam.empirical_binary_vector_distribution(data)
    counts = dataset.counts

    def f(param: _Param, alpha: float) -> float:
        ll = partial(loglikelihood, param)
        log_F_theta = jax.vmap(ll)(dataset.atoms)
        log_alpha = jnp.log(alpha)
        return log_dirichlet_multinomial(
            x=counts,
            log_p=log_F_theta,
            log_alpha=log_alpha,
        )

    return f


def construct_perturbed_loglikelihood(data, loglikelihood, _clamp: float = -80.0):
    """Binds the loglikelihood function to the perturbed model.

    Args:
        data: binary array of shape (n_samples, n_loci)
        loglikelihood: function of signature (params, genotype) -> float
            calculating the loglikelihood for a single data point

    Returns:
        loglikelihood function for the whole data set
          of signature (params, alpha, eta) -> float
    """
    dataset = reparam.empirical_binary_vector_distribution(data)
    counts = dataset.counts

    def f(param: _Param, alpha: float, eta: float) -> float:
        ll = partial(loglikelihood, param)
        log_F_theta = jax.vmap(ll)(dataset.atoms)

        logp_parametric = log_multinomial(counts, log_F_theta)
        logp_nonparametric = log_dirichlet_multinomial(
            counts, log_F_theta, jnp.log(alpha)
        )

        log_eta = jnp.log(eta)
        log_1meta = jnp.log1p(-eta)

        return jnp.logaddexp(log_eta + logp_nonparametric, log_1meta + logp_parametric)

    return f
