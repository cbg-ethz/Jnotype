"""This is a complete copy paste from Aaron's code prior to merging.
https://github.com/allenxzhao/jax/blob/961789ec3897c43a6b23b0517fde4c3f88a6e7eb/df-bayes-counts/df_bayes_count_data.py
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from functools import partial

from typing import Callable, Union, Tuple
from jaxtyping import Float, Int, Array, PRNGKeyArray

Integer = Union[int, Int[Array, " "]]
DataPoint = Int[Array, " G"]
DataSet = Int[Array, "N G"]
Params = Float[Array, "G"]

BootstrapDataSet = Int[Array, "B N G"]
BootstrapParams = Float[Array, "B G"]

LogProbFn = Callable[[Params, DataPoint], Float[Array, " "]]
LogPriorFn = Callable[[Params], Float[Array, " "]]
LossFnDataset = Callable[[Params, DataSet], Float[Array, " "]]
LossFnParams = Callable[[Params], Float[Array, " "]]


def increment(g: Integer, y: DataPoint) -> DataPoint:
    """Increments the count at site `g` by 1."""
    return y.at[g].add(1)


def decrement(g: Integer, y: DataPoint) -> DataPoint:
    """Decrements the count at site `g` by 1."""
    return y.at[g].add(-1)


def _dfd_onepoint_count(log_q: LogProbFn, y: DataPoint) -> Float[Array, " "]:
    """
    Calculates the discrete Fisher divergence for count data on a single data point.

    Args:
        log_q: unnormalized log-probability function
        y: data point on which it should be evaluated
    """
    log_qy = log_q(y)
    indices = jnp.arange(y.shape[0])

    # Term with decrements: (p_theta(x_i^j-) / p_theta(x_i))^2
    def log_q_dec_fn(g: Integer):
        return log_q(decrement(g, y))

    log_q_dec = jax.vmap(log_q_dec_fn)(indices)
    log_ratio_dec = log_q_dec - log_qy
    dec_term = jnp.where(
        y > 0, jnp.exp(2 * log_ratio_dec), jnp.array(0.0)
    )  # account for decrementing 0

    # Term with increments: -2 * (p_theta(x_i) / p_theta(x_i^j+))
    def log_q_inc_fn(g: Integer):
        return log_q(increment(g, y))

    log_q_inc = jax.vmap(log_q_inc_fn)(indices)
    log_ratio_inc = log_qy - log_q_inc
    inc_term = -2 * jnp.exp(log_ratio_inc)

    return jnp.sum(dec_term + inc_term)


def dfd_count(log_q: LogProbFn, ys: DataSet) -> Float[Array, " "]:
    """
    Evaluates the discrete Fisher divergence between the model distribution
    and the empirical distribution for count data.
    """
    f = partial(_dfd_onepoint_count, log_q)
    return jnp.mean(jax.vmap(f)(ys))


def _optimal_beta_single(
    param: Params, loss: LossFnParams, logprior: LogPriorFn
) -> Tuple[Float[Array, " "], Float[Array, " "]]:
    """
    Calculates the numerator and denominator terms of `optimal_beta`
    for a single parameter vector.
    """
    loss_grad = jax.grad(loss)(param)
    loss_hessian = jax.hessian(loss)(param)
    logprior_grad = jax.grad(logprior)(param)
    numer = jnp.dot(loss_grad, logprior_grad) + jnp.trace(loss_hessian)
    denom = jnp.linalg.norm(loss_grad, ord=2) ** 2
    return numer, denom


def optimal_beta(
    params: BootstrapParams, loss: LossFnParams, logprior: LogPriorFn
) -> Float:
    """
    Computes an optimal temperature value based on the bootstrap minimizers.

    Adapted from the PyTorch implementation by Takuo Matsubara:
    https://github.com/takuomatsubara/Discrete-Fisher-Bayes/blob/e0ae7bc24a1e32d76ee6100784be4f6521ede987/Source/Posteriors.py#L145

    Args:
        loss: the loss function subject to calibration
        params: the bootstrap minimizers of the loss
    """
    single_partial = partial(_optimal_beta_single, loss=loss, logprior=logprior)
    numer, denom = jax.vmap(single_partial)(params)
    return jnp.sum(numer) / jnp.sum(denom)


def create_bootstrap_samples(
    key: PRNGKeyArray, dataset: DataSet, B: Int
) -> BootstrapDataSet:
    """Creates `B` bootstrap samples from a dataset."""
    N = dataset.shape[0]
    keys = jax.random.split(key, B)

    def _create_one_sample(key: PRNGKeyArray) -> DataSet:
        return dataset[jrandom.choice(key, N, shape=(N,), replace=True), :]

    return jax.vmap(_create_one_sample)(keys)


def find_minimizers(
    dataset: DataSet,
    dfd_loss_fn: LossFnDataset,
    initial_params: Params,
    optimizer: optax.GradientTransformation,
    num_steps: Int,
) -> Params:
    """Finds the params that minimize `dfd_loss_fn`."""

    def loss_fn(params: Params) -> Float[Array, " "]:
        return dfd_loss_fn(params, dataset)

    def scan_fn(carry, _):
        params, opt_state = carry
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return (params_new, opt_state_new), loss_val

    init = initial_params, optimizer.init(initial_params)
    (final_params, _), losses = jax.lax.scan(scan_fn, init, length=num_steps)
    return final_params, losses


def calibrate_beta(
    key: PRNGKeyArray,
    dataset: DataSet,
    B: Int,
    dfd_loss_fn: LossFnDataset,
    logprior_fn: LogPriorFn,
    initial_params: Params,
    optimizer: optax.GradientTransformation,
    num_steps: Int,
    prior_params: Tuple = (),
) -> Float[Array, " "]:
    """Calibrates beta using bootstrap minimizers."""

    # Find minimizers across bootstrap samples
    minimizers, losses = jax.vmap(
        partial(
            find_minimizers,
            dfd_loss_fn=dfd_loss_fn,
            initial_params=initial_params,
            optimizer=optimizer,
            num_steps=num_steps,
        )
    )(create_bootstrap_samples(key, dataset, B))

    # Calculate optimal beta
    beta_star = optimal_beta(
        minimizers,
        partial(dfd_loss_fn, dataset=dataset),
        partial(logprior_fn, *prior_params),
    )
    return beta_star, losses
