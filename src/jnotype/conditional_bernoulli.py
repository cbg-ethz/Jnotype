"""Conditional Bernoulli distribution utilities.

The conditional Bernoulli distribution is defined as the distribution of
independent Bernoulli variables conditioned on their total number of ones.

This module provides:
1. numerically stable dynamic-programming normalizers,
2. log-likelihood utilities for single and mixture models,
3. pure-JAX samplers for single and mixture conditional Bernoulli models,
4. simplex-oriented parameterization helpers for identifiability,
5. lightweight helpers for NumPyro and BlackJAX integration.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, Int


_FloatLike = Union[float, Float[Array, " "]]


def _calculate_logZ_vector(
    logZ_prev: Float[Array, " G"],
    log_theta: Float[Array, " G"],
    n: int,
) -> Float[Array, " G"]:
    """Single dynamic-programming update for `log Z(n, \theta)` by feature index."""
    neg_inf = jnp.finfo(log_theta.dtype).min
    g = logZ_prev.shape[0]

    prev = jnp.where(jnp.arange(g) < n - 1, neg_inf, jnp.roll(logZ_prev, shift=1))
    return jax.lax.cumlogsumexp(prev + log_theta)


@jax.jit
def _calculate_logZ(
    log_theta: Float[Array, " G"],
    n_max_shape: Int[Array, " n_max"],
) -> Float[Array, " n_max+1"]:
    """Calculates `log Z(k, theta)` for `k = 0, ..., n_max`.

    Args:
        log_theta: vector with log-weights of shape `(G,)`
        n_max_shape: shape-only vector, e.g. `jnp.arange(n_max)`

    Returns:
        Vector of shape `(n_max + 1,)` with entries `log Z(k, theta)`.
    """

    def step(logZ_prev: Float[Array, " G"], n: int):
        new_vec = _calculate_logZ_vector(logZ_prev, log_theta=log_theta, n=n)
        return new_vec, new_vec[-1]

    g = log_theta.shape[0]
    n_max = n_max_shape.shape[0]

    logZ0 = jnp.zeros(g, dtype=log_theta.dtype)
    _, logZ = jax.lax.scan(step, logZ0, xs=jnp.arange(1, n_max + 1))
    return jnp.concatenate([logZ0[:1], logZ])


@jax.jit
def _calculate_logZ_all(
    log_theta: Float[Array, " G"],
    n_max_shape: Int[Array, " n_max"],
) -> Float[Array, "n_max+1 G"]:
    """Calculates full DP table used by the sequential sampler.

    Returns array of shape `(n_max+1, G)`.
    """

    def step(logZ_prev: Float[Array, " G"], n: int):
        new_vec = _calculate_logZ_vector(logZ_prev, log_theta=log_theta, n=n)
        return new_vec, new_vec

    g = log_theta.shape[0]
    n_max = n_max_shape.shape[0]

    logZ0 = jnp.zeros(g, dtype=log_theta.dtype)
    _, logZ = jax.lax.scan(step, logZ0, xs=jnp.arange(1, n_max + 1))
    return jnp.concatenate([logZ0[None, :], logZ])


def calculate_logZ(
    log_theta: Float[Array, " G"],
    n_max: int,
) -> Float[Array, " n_max+1"]:
    """User-facing normalizer wrapper.

    Args:
        log_theta: vector of log-weights `(G,)`
        n_max: maximum number of successes to normalize for
    """
    if n_max < 0:
        raise ValueError(f"n_max must be non-negative, got {n_max}.")
    return _calculate_logZ(
        log_theta=log_theta, n_max_shape=jnp.arange(n_max, dtype=int)
    )


def calculate_logZ_all(
    log_theta: Float[Array, " G"],
    n_max: int,
) -> Float[Array, "n_max+1 G"]:
    """User-facing wrapper returning the full DP table for sampling."""
    if n_max < 0:
        raise ValueError(f"n_max must be non-negative, got {n_max}.")
    return _calculate_logZ_all(
        log_theta=log_theta,
        n_max_shape=jnp.arange(n_max, dtype=int),
    )


def calculate_logZ_batched(
    log_theta_batch: Float[Array, "B G"],
    n_max: int,
) -> Float[Array, "B n_max+1"]:
    """Batched version of `calculate_logZ` over the first axis."""
    if n_max < 0:
        raise ValueError(f"n_max must be non-negative, got {n_max}.")
    n_max_shape = jnp.arange(n_max, dtype=int)
    return jax.vmap(lambda log_theta: _calculate_logZ(log_theta, n_max_shape))(
        log_theta_batch
    )


@jax.jit
def simplex_to_log_theta(
    simplex_probs: Float[Array, " *shape G"],
) -> Float[Array, " *shape G"]:
    """Maps simplex probabilities to `log_theta`.

    This gives an identifiable parameterization for conditional Bernoulli models.
    """
    return jnp.log(simplex_probs)


@jax.jit
def logits_to_simplex(logits: Float[Array, " *shape G"]) -> Float[Array, " *shape G"]:
    """Maps unconstrained logits to simplex probabilities."""
    return jax.nn.softmax(logits, axis=-1)


@jax.jit
def logits_to_log_theta(logits: Float[Array, " *shape G"]) -> Float[Array, " *shape G"]:
    """Maps unconstrained logits directly to identified `log_theta`."""
    return jax.nn.log_softmax(logits, axis=-1)


def _to_n_obs(
    ys: Int[Array, "N G"],
    n_obs: Optional[Int[Array, " N"]] = None,
) -> Int[Array, " N"]:
    if n_obs is None:
        return jnp.asarray(ys.sum(axis=-1), dtype=int)
    n_obs = jnp.asarray(n_obs, dtype=int)
    if n_obs.ndim != 1:
        raise ValueError(f"Expected `n_obs` to have ndim=1, got {n_obs.ndim}.")
    if n_obs.shape[0] != ys.shape[0]:
        raise ValueError(
            f"`n_obs` length ({n_obs.shape[0]}) must match sample size ({ys.shape[0]})."
        )
    return n_obs


def conditional_bernoulli_logpmf(
    y: Int[Array, " G"],
    log_theta: Float[Array, " G"],
    n: Optional[int] = None,
    *,
    precomputed_logZ: Optional[Float[Array, " n_max+1"]] = None,
) -> _FloatLike:
    """Evaluates log PMF for a single binary vector.

    If `n` is not provided, it is set to `sum(y)`.
    """
    y = jnp.asarray(y, dtype=int)
    if y.ndim != 1:
        raise ValueError(f"Expected `y` to have shape (G,), got ndim={y.ndim}.")

    if n is None:
        n_index = jnp.asarray(jnp.sum(y), dtype=int)
    else:
        n_index = jnp.asarray(n, dtype=int)

    if precomputed_logZ is None:
        # Use full support to stay compatible with JAX tracing
        # (no Python int conversion from traced values).
        precomputed_logZ = calculate_logZ(log_theta, n_max=y.shape[0])

    safe_index = jnp.clip(n_index, 0, precomputed_logZ.shape[0] - 1)
    logZ_n = precomputed_logZ[safe_index]

    valid_n = jnp.logical_and(n_index >= 0, n_index <= y.shape[0])
    valid_sum = jnp.sum(y) == n_index
    valid_precomputed = n_index < precomputed_logZ.shape[0]

    ll = jnp.dot(y, log_theta) - logZ_n
    neg_inf = jnp.array(-jnp.inf, dtype=log_theta.dtype)
    valid = jnp.logical_and(jnp.logical_and(valid_n, valid_sum), valid_precomputed)
    return jnp.where(valid, ll, neg_inf)


def conditional_bernoulli_loglikelihood(
    ys: Int[Array, "N G"],
    log_theta: Float[Array, " G"],
    n_obs: Optional[Int[Array, " N"]] = None,
) -> _FloatLike:
    """Total log-likelihood of observed data under a single component."""
    ys = jnp.asarray(ys, dtype=int)
    if ys.ndim != 2:
        raise ValueError(f"Expected `ys` to have shape (N, G), got ndim={ys.ndim}.")

    ns = _to_n_obs(ys, n_obs=n_obs)
    # Use full support {0, ..., G} to avoid tracer-to-int conversion
    # in compiled contexts.
    n_max = ys.shape[1]
    logZ = calculate_logZ(log_theta, n_max=n_max)

    ll_unnorm = jnp.einsum("ng,g->n", ys, log_theta)
    safe_n = jnp.clip(ns, 0, logZ.shape[0] - 1)
    ll = ll_unnorm - logZ[safe_n]

    valid_ns = jnp.logical_and(ns >= 0, ns <= ys.shape[1])
    valid_sums = ys.sum(axis=-1) == ns
    valid = jnp.logical_and(valid_ns, valid_sums)

    neg_inf = jnp.finfo(log_theta.dtype).min
    ll = jnp.where(valid, ll, neg_inf)
    return jnp.sum(ll)


def conditional_bernoulli_component_loglikelihood_matrix(
    ys: Int[Array, "N G"],
    component_log_theta: Float[Array, "K G"],
    n_obs: Optional[Int[Array, " N"]] = None,
) -> Float[Array, "N K"]:
    """Component-wise log-likelihood matrix for conditional Bernoulli mixtures."""
    ys = jnp.asarray(ys, dtype=int)
    if ys.ndim != 2:
        raise ValueError(f"Expected `ys` to have shape (N, G), got ndim={ys.ndim}.")
    if component_log_theta.ndim != 2:
        raise ValueError(
            "Expected `component_log_theta` to have shape (K, G), "
            f"got ndim={component_log_theta.ndim}."
        )

    if ys.shape[1] != component_log_theta.shape[1]:
        raise ValueError(
            "Feature mismatch: ys has "
            f"G={ys.shape[1]} and components have G={component_log_theta.shape[1]}."
        )

    ns = _to_n_obs(ys, n_obs=n_obs)
    # Use full support {0, ..., G} to avoid tracer-to-int conversion
    # in compiled contexts.
    n_max = ys.shape[1]

    logZ = calculate_logZ_batched(component_log_theta, n_max=n_max)  # (K, n_max+1)

    ll_unnorm = jnp.einsum("ng,kg->nk", ys, component_log_theta)
    safe_n = jnp.clip(ns, 0, logZ.shape[1] - 1)
    norm_terms = jnp.take(logZ, safe_n, axis=1).T  # (N, K)

    ll = ll_unnorm - norm_terms

    valid_ns = jnp.logical_and(ns >= 0, ns <= ys.shape[1])
    valid_sums = ys.sum(axis=-1) == ns
    valid = jnp.logical_and(valid_ns, valid_sums)[:, None]

    neg_inf = jnp.finfo(component_log_theta.dtype).min
    return jnp.where(valid, ll, neg_inf)


def conditional_bernoulli_mixture_loglikelihood(
    ys: Int[Array, "N G"],
    mixing_logits: Float[Array, " K"],
    component_log_theta: Float[Array, "K G"],
    n_obs: Optional[Int[Array, " N"]] = None,
) -> _FloatLike:
    """Total log-likelihood of a conditional Bernoulli mixture model."""
    component_ll = conditional_bernoulli_component_loglikelihood_matrix(
        ys=ys,
        component_log_theta=component_log_theta,
        n_obs=n_obs,
    )

    log_mixing = jax.nn.log_softmax(mixing_logits)
    return jnp.sum(logsumexp(component_ll + log_mixing[None, :], axis=-1))


def generate_loglikelihood(ys: Int[Array, "N G"], n_max: int = None):
    """Generates a log-likelihood closure for a single-component model.

    Args:
        ys: observed binary data, shape `(N, G)`
        n_max: maximum value used for precomputing `log Z`;
            defaults to `max(sum(ys, axis=-1))`

    Returns:
        A jitted function mapping `log_theta` `(G,)` to scalar total log-likelihood.
    """
    ys = jnp.asarray(ys, dtype=int)
    ns = jnp.asarray(ys.sum(axis=-1), dtype=int)

    if n_max is None:
        n_max = int(ns.max()) if ns.size else 0

    weights = jnp.bincount(ns, length=n_max + 1)

    @jax.jit
    def loglikelihood(log_theta: Float[Array, " G"]) -> _FloatLike:
        """Evaluates total log-likelihood for one conditional Bernoulli component."""
        ll = jnp.sum(ys * log_theta[None, ...])
        logZ = calculate_logZ(log_theta, n_max=n_max)
        return ll - jnp.sum(logZ * weights)

    return loglikelihood


def _logR(n: int, y: Int[Array, " G"], log_theta: Float[Array, " G"]):
    """Calculates `log R(n, Y)` where subset `Y` is represented by binary indicators.

    This helper is mainly useful for debugging/testing.
    """
    size = int(jnp.sum(y))
    if n == 0:
        return 0.0
    if n > size:
        return -jnp.inf

    log_theta_ = log_theta[y == 1]
    return calculate_logZ(log_theta_, n_max=n)[-1]


def _get_logR_from_table(
    log_Rs: Float[Array, "n_max+1 G"],
    n: Int[Array, ""],
    size: Int[Array, ""],
) -> _FloatLike:
    """Reads `log R(n, size)` from a precomputed table."""
    zero = jnp.array(0.0, dtype=log_Rs.dtype)
    neg_inf = jnp.array(-jnp.inf, dtype=log_Rs.dtype)

    def _valid(_):
        return log_Rs[n, size - 1]

    nonzero = jax.lax.cond(
        jnp.logical_and(jnp.logical_and(n > 0, size > 0), n <= size),
        _valid,
        lambda _: neg_inf,
        operand=None,
    )

    return jax.lax.cond(n == 0, lambda _: zero, lambda _: nonzero, operand=None)


@jax.jit
def _sample_single_conditional_bernoulli_with_table(
    key: jax.Array,
    n: Int[Array, ""],
    log_theta: Float[Array, " G"],
    log_Rs: Float[Array, "n_max+1 G"],
) -> Int[Array, " G"]:
    """Samples one observation with fixed `n` and precomputed `log_Rs` table."""
    g = log_theta.shape[0]
    keys = jax.random.split(key, g)

    def step(carry, data):
        n_ones, sample = carry
        k, subkey = data

        active = n_ones < n

        logRnum = _get_logR_from_table(log_Rs, n - n_ones - 1, g - (k + 1))
        logRden = _get_logR_from_table(log_Rs, n - n_ones, g - k)

        log_p = log_theta[k] + logRnum - logRden
        take = jnp.logical_and(active, jnp.log(jax.random.uniform(subkey)) < log_p)

        sample = sample.at[k].set(take.astype(jnp.int32))
        n_ones = n_ones + take.astype(jnp.int32)
        return (n_ones, sample), None

    init_sample = jnp.zeros(g, dtype=jnp.int32)
    (_, sample), _ = jax.lax.scan(
        step, (jnp.array(0, dtype=jnp.int32), init_sample), (jnp.arange(g), keys)
    )

    sample = jax.lax.cond(
        n == 0,
        lambda _: jnp.zeros(g, dtype=jnp.int32),
        lambda _: sample,
        operand=None,
    )
    sample = jax.lax.cond(
        n == g,
        lambda _: jnp.ones(g, dtype=jnp.int32),
        lambda _: sample,
        operand=None,
    )
    return sample


def _prepare_ns(ns: Union[Sequence[int], Int[Array, " N"]]) -> Int[Array, " N"]:
    ns = jnp.asarray(ns, dtype=int)
    if ns.ndim == 0:
        ns = ns[None]
    if ns.ndim != 1:
        raise ValueError(f"Expected `ns` to have ndim=1, got {ns.ndim}.")
    return ns


def sample_conditional_bernoulli_many_n(
    key: jax.Array,
    ns: Union[Sequence[int], Int[Array, " N"]],
    log_theta: Float[Array, " G"],
) -> Int[Array, "N G"]:
    """Samples conditional Bernoulli vectors for observation-specific `n` values."""
    ns = _prepare_ns(ns)

    g = log_theta.shape[0]
    min_n = int(jnp.min(ns)) if ns.size else 0
    max_n = int(jnp.max(ns)) if ns.size else 0

    if min_n < 0:
        raise ValueError(f"All values in `ns` must be non-negative, got min={min_n}.")
    if max_n > g:
        raise ValueError(f"Need all n<=G, but got max(n)={max_n} and G={g}.")

    log_Rs = calculate_logZ_all(log_theta=log_theta[::-1], n_max=max_n)

    keys = jax.random.split(key, ns.shape[0])
    return jax.vmap(
        lambda k, n: _sample_single_conditional_bernoulli_with_table(
            k, n, log_theta, log_Rs
        )
    )(keys, ns)


def sample_conditional_bernoulli(
    key: jax.Array,
    ns: Union[Sequence[int], Int[Array, " N"]],
    log_theta: Float[Array, " G"],
) -> Int[Array, "N G"]:
    """Backwards-compatible alias for sampling conditional Bernoulli vectors.

    This keeps the historical signature used in workflows.
    """
    return sample_conditional_bernoulli_many_n(key=key, ns=ns, log_theta=log_theta)


def sample_conditional_bernoulli_many_n_with_log_theta(
    key: jax.Array,
    ns: Union[Sequence[int], Int[Array, " N"]],
    log_theta: Float[Array, "N G"],
) -> Int[Array, "N G"]:
    """Samples with observation-specific `log_theta` rows."""
    ns = _prepare_ns(ns)
    if log_theta.ndim != 2:
        raise ValueError(
            f"Expected `log_theta` to have shape (N, G), got ndim={log_theta.ndim}."
        )
    if ns.shape[0] != log_theta.shape[0]:
        raise ValueError(
            f"Length mismatch: len(ns)={ns.shape[0]} "
            f"and log_theta has N={log_theta.shape[0]}."
        )

    g = log_theta.shape[1]
    min_n = int(jnp.min(ns)) if ns.size else 0
    max_n = int(jnp.max(ns)) if ns.size else 0

    if min_n < 0:
        raise ValueError(f"All values in `ns` must be non-negative, got min={min_n}.")
    if max_n > g:
        raise ValueError(f"Need all n<=G, but got max(n)={max_n} and G={g}.")

    log_Rs = jax.vmap(
        lambda row_log_theta: calculate_logZ_all(row_log_theta[::-1], n_max=max_n)
    )(log_theta)

    keys = jax.random.split(key, ns.shape[0])
    return jax.vmap(_sample_single_conditional_bernoulli_with_table)(
        keys, ns, log_theta, log_Rs
    )


def sample_conditional_bernoulli_mixture(
    key: jax.Array,
    ns: Union[Sequence[int], Int[Array, " N"]],
    mixing_logits: Float[Array, " K"],
    component_log_theta: Float[Array, "K G"],
    *,
    return_assignments: bool = False,
):
    """Samples from a conditional Bernoulli mixture model.

    Args:
        key: PRNG key
        ns: per-observation constrained numbers of ones, shape `(N,)`
        mixing_logits: logits of mixture weights, shape `(K,)`
        component_log_theta: component log-weights, shape `(K, G)`
        return_assignments: if True, also returns sampled component labels
    """
    ns = _prepare_ns(ns)
    n_samples = ns.shape[0]

    if component_log_theta.ndim != 2:
        raise ValueError(
            "Expected `component_log_theta` with shape (K, G), "
            f"got ndim={component_log_theta.ndim}."
        )
    if mixing_logits.ndim != 1:
        raise ValueError(
            f"Expected `mixing_logits` with shape (K,), got ndim={mixing_logits.ndim}."
        )
    if mixing_logits.shape[0] != component_log_theta.shape[0]:
        raise ValueError(
            "Mismatch between number of components in `mixing_logits` "
            "and `component_log_theta`: "
            f"{mixing_logits.shape[0]} vs {component_log_theta.shape[0]}."
        )

    key_assign, key_sample = jax.random.split(key)
    assignments = jax.random.categorical(key_assign, mixing_logits, shape=(n_samples,))
    selected_log_theta = component_log_theta[assignments, :]

    samples = sample_conditional_bernoulli_many_n_with_log_theta(
        key=key_sample,
        ns=ns,
        log_theta=selected_log_theta,
    )

    if return_assignments:
        return samples, assignments
    return samples


def numpyro_conditional_bernoulli_factor(
    name: str,
    ys: Int[Array, "N G"],
    simplex_probs: Float[Array, " G"],
    n_obs: Optional[Int[Array, " N"]] = None,
) -> None:
    """Adds conditional Bernoulli log-likelihood to a NumPyro model via `factor`.

    This helper expects an identifiable simplex parameterization.
    """
    import numpyro

    ll = conditional_bernoulli_loglikelihood(
        ys=ys,
        log_theta=simplex_to_log_theta(simplex_probs),
        n_obs=n_obs,
    )
    numpyro.factor(name, ll)


def numpyro_conditional_bernoulli_mixture_factor(
    name: str,
    ys: Int[Array, "N G"],
    mixing_simplex: Float[Array, " K"],
    component_simplex: Float[Array, "K G"],
    n_obs: Optional[Int[Array, " N"]] = None,
) -> None:
    """Adds conditional Bernoulli mixture log-likelihood to a NumPyro model."""
    import numpyro

    ll = conditional_bernoulli_mixture_loglikelihood(
        ys=ys,
        mixing_logits=jnp.log(mixing_simplex),
        component_log_theta=simplex_to_log_theta(component_simplex),
        n_obs=n_obs,
    )
    numpyro.factor(name, ll)


def build_blackjax_logdensity_single(
    ys: Int[Array, "N G"],
    *,
    n_obs: Optional[Int[Array, " N"]] = None,
    prior_logprob_fn: Optional[Callable[[dict], _FloatLike]] = None,
) -> Callable[[dict], _FloatLike]:
    """Builds a BlackJAX-compatible log-density function for a single component.

    Expected parameters dictionary:
    - `{"simplex_probs": array(G,)}`
    """
    ys = jnp.asarray(ys, dtype=int)

    if prior_logprob_fn is None:

        def prior_logprob_fn(_):
            """Default prior term returning zero log-density."""
            return 0.0

    def logdensity(params: dict) -> _FloatLike:
        """Evaluates log-prior plus single-component CB log-likelihood."""
        simplex_probs = params["simplex_probs"]
        ll = conditional_bernoulli_loglikelihood(
            ys=ys,
            log_theta=simplex_to_log_theta(simplex_probs),
            n_obs=n_obs,
        )
        return prior_logprob_fn(params) + ll

    return logdensity


def build_blackjax_logdensity_mixture(
    ys: Int[Array, "N G"],
    *,
    n_obs: Optional[Int[Array, " N"]] = None,
    prior_logprob_fn: Optional[Callable[[dict], _FloatLike]] = None,
) -> Callable[[dict], _FloatLike]:
    """Builds a BlackJAX-compatible log-density function for a mixture model.

    Expected parameters dictionary:
    - `{"mixing_simplex": array(K,), "component_simplex": array(K, G)}`
    """
    ys = jnp.asarray(ys, dtype=int)

    if prior_logprob_fn is None:

        def prior_logprob_fn(_):
            """Default prior term returning zero log-density."""
            return 0.0

    def logdensity(params: dict) -> _FloatLike:
        """Evaluates log-prior plus conditional Bernoulli mixture log-likelihood."""
        ll = conditional_bernoulli_mixture_loglikelihood(
            ys=ys,
            mixing_logits=jnp.log(params["mixing_simplex"]),
            component_log_theta=simplex_to_log_theta(params["component_simplex"]),
            n_obs=n_obs,
        )
        return prior_logprob_fn(params) + ll

    return logdensity


__all__ = [
    "_calculate_logZ",
    "_calculate_logZ_all",
    "_calculate_logZ_vector",
    "_logR",
    "build_blackjax_logdensity_mixture",
    "build_blackjax_logdensity_single",
    "calculate_logZ",
    "calculate_logZ_all",
    "calculate_logZ_batched",
    "conditional_bernoulli_component_loglikelihood_matrix",
    "conditional_bernoulli_loglikelihood",
    "conditional_bernoulli_logpmf",
    "conditional_bernoulli_mixture_loglikelihood",
    "generate_loglikelihood",
    "logits_to_log_theta",
    "logits_to_simplex",
    "numpyro_conditional_bernoulli_factor",
    "numpyro_conditional_bernoulli_mixture_factor",
    "sample_conditional_bernoulli",
    "sample_conditional_bernoulli_many_n",
    "sample_conditional_bernoulli_many_n_with_log_theta",
    "sample_conditional_bernoulli_mixture",
    "simplex_to_log_theta",
]
