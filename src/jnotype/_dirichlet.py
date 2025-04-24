"""The Dirichlet process/Dirichlet-multinomial model."""

import jax
import jax.numpy as jnp
from jax.scipy.special import betaln, gammaln
import jnotype._reparam as reparam
from functools import partial
from typing import Callable, Sequence, TypeVar
from jaxtyping import Float, Array

_Float = Float[Array, " "]
_Param = TypeVar("_Param")


def log_factorial(x):
    return gammaln(x + 1)


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
    log_const = log_factorial(dataset.n_datapoints) - jnp.sum(
        log_factorial(dataset.counts)
    )

    def f(param: _Param) -> _Float:
        """Function to be returned."""
        ll = partial(loglikelihood, param)
        return log_const + dataset.calculate_function_sum(ll)

    return f


def _safe_exp(log_x, *, floor: float):
    """
    exp(log_x) but clamps log_x on the left so that
    we never reach exactly zero and still keep correct gradients.
    """
    return jnp.exp(jnp.clip(log_x, a_min=floor))


def construct_dirichlet_multinomial_loglikelihood(
    data, loglikelihood, _clamp: float = -80.0
):
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
    n = dataset.n_datapoints
    counts = dataset.counts

    def f(param: _Param, alpha: float) -> float:
        ll = partial(loglikelihood, param)
        log_F_theta = jax.vmap(ll)(dataset.atoms)

        parametric = _safe_exp(jnp.log(alpha) + log_F_theta, floor=_clamp)
        log_num = jnp.log(n) + betaln(alpha, n)
        log_den = jnp.sum(betaln(parametric, counts) + jnp.log(counts))
        return log_num - log_den

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
    n = dataset.n_datapoints
    counts = dataset.counts
    log_const = log_factorial(dataset.n_datapoints) - jnp.sum(
        log_factorial(dataset.counts)
    )

    def f(param: _Param, alpha: float, eta: float) -> float:
        ll = partial(loglikelihood, param)
        log_F_theta = jax.vmap(ll)(dataset.atoms)
        logp_parametric = jnp.sum(log_F_theta) + log_const

        alpha_F_theta = _safe_exp(jnp.log(alpha) + log_F_theta, floor=_clamp)
        log_num = jnp.log(n) + betaln(alpha, n)
        log_den = jnp.sum(betaln(alpha_F_theta, counts) + jnp.log(counts))
        logp_nonparametric = log_num - log_den

        log_eta = jnp.log(eta)
        log_1meta = jnp.log1p(-eta)

        return jnp.logaddexp(log_eta + logp_nonparametric, log_1meta + logp_parametric)

    return f
