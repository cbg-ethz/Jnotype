"""The Dirichlet process/Dirichlet-multinomial model."""

import jax
import jax.numpy as jnp
from jax.scipy.special import betaln
import jnotype._reparam as reparam
from functools import partial
from typing import Callable, Sequence, TypeVar
from jaxtyping import Float, Array

_Float = Float[Array, " "]
_Param = TypeVar("_Param")


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
        return dataset.calculate_function_sum(ll)

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

    def f(param: _Param, alpha: float) -> float:
        ll = partial(loglikelihood, param)
        log_F_theta = jax.vmap(ll)(dataset.atoms)

        parametric = _safe_exp(jnp.log(alpha) + log_F_theta, floor=_clamp)
        return betaln(alpha, dataset.n_datapoints) - jnp.sum(
            betaln(parametric, dataset.counts)
        )

    return f
