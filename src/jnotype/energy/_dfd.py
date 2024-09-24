from functools import partial
from typing import Callable, Union
from jaxtyping import Float, Int, Array

import jax
import jax.numpy as jnp


Integer = Union[int, Int[Array, " "]]
DataPoint = Int[Array, " G"]
DataSet = Int[Array, "N G"]

LogProbFn = Callable[[DataPoint], Float[Array, " "]]


def bitflip(g: Integer, y: DataPoint) -> DataPoint:
    """Flips a bit at site `g`."""
    return y.at[g].set(1 - y[g])


def _dfd_onepoint(log_q: LogProbFn, y: DataPoint) -> Float[Array, " "]:
    """Calculates the discrete Fisher divergence on a single data point.

    Args:
        log_q: unnormalized log-probability function
        y: data point on which it should be evaluated
    """
    log_qy = log_q(y)

    def log_q_flip_fn(g: Union[int, Int[Array, " "]]):
        return log_q(bitflip(g, y))

    log_qflipped = jax.vmap(log_q_flip_fn)(jnp.arange(y.shape[0]))
    log_ratio = log_qflipped - log_qy
    return jnp.sum(jnp.exp(2 * log_ratio) - 2 * jnp.exp(-log_ratio))


def discrete_fisher_divergence(log_q: LogProbFn, ys: DataSet) -> Float[Array, " "]:
    """Evaluates the discrete Fisher divergence between the model distribution
    and the empirical distribution.

    Note:
        When using in generalised Bayesian inference framework,
        remember that the update is multiplied by the data set size
        (and the temperature), i.e.,
        $$
            P(\\theta | data) \\propto P(\\theta) * \\exp( -\\tau N DFD )
        $$
    """
    f = partial(_dfd_onepoint, log_q)
    return jnp.mean(jax.vmap(f)(ys))


def _besag_pseudolikelihood_onepoint(
    log_q: LogProbFn, y: DataPoint
) -> Float[Array, " "]:
    """Calculates Julian Besag's pseudolikelihood on a single data point.

    Namely,
    $$
        \\log L &= \\sum_g \\log P(Y[i] = y[i] | Y[~i] = y[~i] )
                &= \\sum_g \\log P(Y = y) - \\log( P(Y = y) + P(Y = bitflip(g, y) ))
    $$
    """
    log_qy = log_q(y)

    def log_denominator(g: Union[int, Int[Array, " "]]):
        log_bitflipped = log_q(bitflip(g, y))
        return jnp.logaddexp(log_qy, log_bitflipped)

    log_denominators = jax.vmap(log_denominator)(jnp.arange(y.shape[0]))
    return jnp.sum(log_qy - log_denominators)


def besag_pseudolikelihood_sum(
    log_q: LogProbFn,
    ys: DataSet,
) -> Float[Array, " "]:
    """Besag pseudolikelihood calculated over the whole data set.

    Note that pseudolikelihood is additive.
    """
    n_points = ys.shape[0]
    return n_points * besag_pseudolikelihood_mean(log_q, ys)


def besag_pseudolikelihood_mean(
    log_q: LogProbFn,
    ys: DataSet,
) -> Float[Array, " "]:
    """Average Besag pseudolikelihood.

    Note:
        As the pseudolikelihood is additive, for generalised
        Bayesian inference one should multiply by the data set size.

    See Also:
        `discrete_fisher_divergence`, which also requires multiplication
        by the data set size
    """
    f = partial(_besag_pseudolikelihood_onepoint, log_q)
    return jnp.mean(jax.vmap(f)(ys))
