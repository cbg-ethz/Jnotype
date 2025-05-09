"""The Dirichlet process/Dirichlet-multinomial model."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, digamma, betaln
import jnotype._reparam as reparam
from functools import partial
from typing import Callable, Sequence, TypeVar
from jaxtyping import Float, Array, Bool, Int


_Float = Float[Array, " "]
_Param = TypeVar("_Param")


def log_factorial(x):
    return gammaln(x + 1)


def multinomial_log_coef(x):
    n = jnp.sum(x)
    return log_factorial(n) - jnp.sum(log_factorial(x))


def log_multinomial(x, log_p):
    return multinomial_log_coef(x) + jnp.sum(x * log_p)


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


def _lgamma_ratio_impl(log_a, k):
    def body(i, acc):
        i_f = i.astype(log_a.dtype)
        return acc + jnp.logaddexp(log_a, jnp.log(i_f))

    return jax.lax.fori_loop(0, k, body, 0.0)


@jax.custom_vjp
def _log_gamma_ratio(log_a, k):
    return _lgamma_ratio_impl(log_a, k)


# ---- forward & backward rules ---------------------------------------
def _lgamma_ratio_fwd(log_a, k):
    a = jnp.exp(log_a)
    val = _lgamma_ratio_impl(log_a, k)
    return val, (a, k)  # saved for backward


def _lgamma_ratio_bwd(res, g):
    a, k = res
    grad_log_a = g * a * (digamma(a + k) - digamma(a))
    # no gradient w.r.t. integer k
    return (grad_log_a, jnp.zeros_like(k))


_log_gamma_ratio.defvjp(_lgamma_ratio_fwd, _lgamma_ratio_bwd)


# def _log_gamma_ratio(log_a, k):
#     """Numerically stable  log Γ(exp(log_a)+k) - log Γ(exp(log_a)),
#     computed in log-space as  Σ_{j=0}^{k-1} log(exp(log_a)+j)
#     using log-add-exp trick.
#     """
#     k = jnp.asarray(k, dtype=jnp.int32)

#     def body(i, acc):
#         i_f = i.astype(log_a.dtype)
#         return acc + jnp.logaddexp(log_a, jnp.log(i_f))

#     return jax.lax.fori_loop(0, k, body, 0.0)


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
    log_coeff = multinomial_log_coef(x)

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


def construct_perturbed_loglikelihood(data, loglikelihood):
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


def sum_indicators(
    items: Bool[Array, " N"],
    class_sizes: Int[Array, " K"],
) -> Int[Array, " K"]:
    """For each class there are several binary items.
    This function calculates how many items are active
    within each class.

    Args:
        items: binary items. Array of shape (N,)
        class_sizes: represents the class sizes. The first `class_sizes[0]`
            items belong to class 0, then the next `class_sizes[1]` items
            belong to class 1 etc.

    Returns:
        number of active items per class, shape the same as `class_sizes`
    """
    starts = jnp.concatenate([jnp.array([0]), jnp.cumsum(class_sizes)[:-1]])
    # Note that this works in JAX v0.6.0, but not in v0.4.30
    return jnp.add.reduceat(items, starts)


def construct_mixture_loglikelihood(data, loglikelihood):
    """Constructs the loglikelihood function for the mixture model.

    Args:
        data: binary array of shape (n_samples, n_loci)
        loglikelihood: function of signature (params, genotype) -> float
            calculating the loglikelihood for a single data point

    Returns:
        loglikelihood function `log P(Y | param, alpha, Z)`
    """
    dataset = reparam.empirical_binary_vector_distribution(data)
    counts = dataset.counts

    def f(param: _Param, alpha: float, z: jax.Array):
        """Uses the counts summary statistics
        constructed for both components."""
        # Counts in the nonparametric component
        counts_nonparam = sum_indicators(z, counts)
        # Counts in the parametric component
        counts_param = counts - counts_nonparam

        ll = partial(loglikelihood, param)
        log_F_theta = jax.vmap(ll)(dataset.atoms)

        loglikelihood_param = log_multinomial(
            counts_param, log_F_theta
        ) - multinomial_log_coef(counts_param)
        loglikelihood_nonparam = log_dirichlet_multinomial(
            counts_nonparam, log_F_theta, jnp.log(alpha)
        ) - multinomial_log_coef(counts_nonparam)

        return loglikelihood_param + loglikelihood_nonparam

    return f


def construct_mixture_log_prob_integrated_weight(
    data, loglikelihood, weight_beta_prior: tuple[float, float]
):
    """Constructs the loglikelihood function for the mixture model
    with the mixture weight marginalized out

    Args:
        data: binary array of shape (n_samples, n_loci)
        loglikelihood: function of signature (params, genotype) -> float
            calculating the loglikelihood for a single data point
        weight_beta_prior: the parameters (w_1, w_0) for the beta
            prior on the weight

    Returns:
        loglikelihood function
          log P(Y, Z | param, alpha) = log P(Y | param, alpha, Z) + log P(Z),
          where P(Z) is a model sampling a weight w ~ Beta(w1, w0) and then
          generating independently Z[n] ~ Bernoulli(w).
    """
    n: int = data.shape[0]
    loglike_fn = construct_mixture_loglikelihood(data, loglikelihood)
    w1, w0 = weight_beta_prior
    log_denominator = betaln(w1, w0) + gammaln(n + w1 + w0)

    def f(param: _Param, alpha: float, z: jax.Array):
        """Uses the loglikelihood, correcting by the log-prior
        factor, which has analytic form due to conjugacy."""
        ll = loglike_fn(param, alpha, z)
        n_ = jnp.sum(jnp.asarray(z, dtype=jnp.int32))
        log_prior = gammaln(w1 + n_) + gammaln(w0 + n - n_) - log_denominator
        return ll + log_prior

    return f
