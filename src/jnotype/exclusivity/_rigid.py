"""Model proposed by Szczurek and Beerenwinkel (2014)."""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
from typing import NamedTuple, Tuple, Union

import jnotype.exclusivity._bernoulli_mixtures as bmm


def _bincount(counts: Int[Array, " counts"], n_genes: int) -> Int[Array, " n_genes+1"]:
    """Calculates occurences of different values from the set {0, 1, ..., n_genes}."""
    return jnp.bincount(counts, minlength=n_genes + 1, length=n_genes + 1)


def _calculate_summary_statistic(
    Y: Int[Array, "n_samples n_genes"],
) -> Int[Array, " n_genes+1"]:
    """Calculates the summary statistic, counting the occurences of 1s."""
    return _bincount(counts=Y.sum(axis=-1), n_genes=Y.shape[-1])


_FloatLike = Union[float, Float[Array, " "]]


def _calculate_d(alpha: _FloatLike, beta: _FloatLike, delta: _FloatLike) -> _FloatLike:
    """Calculates auxiliary parameter d."""
    return delta * (1.0 - beta) + (1.0 - delta) * alpha


def _calculate_loglikelihood_single_point(
    k: int,
    G: int,
    alpha: _FloatLike,
    beta: _FloatLike,
    gamma: _FloatLike,
    delta: _FloatLike,
) -> _FloatLike:
    """Calculates loglikelihood on a single point with `k` mutations."""
    log_term1 = jnp.log1p(-gamma) + k * jnp.log(alpha) + (G - k) * jnp.log1p(-alpha)

    d = _calculate_d(alpha=alpha, beta=beta, delta=delta)

    log_term21 = (
        jnp.log(gamma) - jnp.log(G) + (k - 1) * jnp.log(d) + (G - k - 1) * jnp.log1p(-d)
    )

    log_term22 = jnp.logaddexp(
        jnp.log(k) + jnp.log1p(-beta) + jnp.log1p(-d),
        jnp.log(G - k) + jnp.log(beta) + jnp.log(d),
    )

    log_term2 = log_term21 + log_term22

    return jnp.logaddexp(log_term1, log_term2)


class Parameters(NamedTuple):
    """Parameters in the mutual exclusivity model.

    Parameters:
        false_positive_rate: false positive rate
        false_negative_rate: false negative rate
    """

    false_positive_rate: _FloatLike  # Alpha, false positive rate
    false_negative_rate: _FloatLike  # Beta, false negative rate
    coverage: _FloatLike  # Gamma parameter, coverage
    impurity: _FloatLike  # Delta parameter, impurity


def _get_loglikelihood_function_from_counts(
    counts: Int[Array, " n_genes+1"], from_params: bool = False
):
    """Factory for the likelihood function,
    which uses an easy-to-calculate
    summary statistic of the data,
    improving the computation speed."""
    G = counts.shape[0] - 1

    ks = jnp.arange(counts.shape[0])
    assert ks.shape == counts.shape

    def f(
        alpha: _FloatLike, beta: _FloatLike, gamma: _FloatLike, delta: _FloatLike
    ) -> _FloatLike:
        lls = _calculate_loglikelihood_single_point(
            k=ks,  # type: ignore
            G=G,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        return jnp.dot(counts, lls)

    def g(params: Parameters):
        return f(
            alpha=params.false_positive_rate,
            beta=params.false_negative_rate,
            gamma=params.coverage,
            delta=params.impurity,
        )

    if from_params:
        return g
    else:
        return f


def get_loglikelihood_function(
    Y: Int[Array, "n_samples n_genes"], from_params: bool = False
):
    """Factory for the likelihood function,
    which uses an easy-to-calculate
    summary statistic of the data,
    improving the computation speed."""
    counts = _calculate_summary_statistic(Y)
    return _get_loglikelihood_function_from_counts(counts, from_params=from_params)


def estimate_no_errors(Y: Int[Array, "n_samples n_genes"]) -> Parameters:
    """Estimates the parameter values assuming no errors in the data,
    i.e., FPR = FNR = 0."""
    zero = jnp.asarray(0.0)

    statistic = _calculate_summary_statistic(Y)
    n_samples, n_genes = Y.shape[0], Y.shape[1]

    coverage = 1 - statistic[0] / n_samples

    numerator = jnp.sum(jnp.arange(n_genes + 1) * statistic) - n_samples * coverage
    denominator = (n_genes - 1) * n_samples * coverage

    return Parameters(
        false_positive_rate=zero,
        false_negative_rate=zero,
        coverage=coverage,
        impurity=numerator / denominator,
    )


def convert_to_bernoulli_mixture(parameters: Parameters, n_genes: int) -> Tuple[
    Float[Array, " n_genes+1"],
    Float[Array, "n_genes+1 n_genes"],
]:
    """Generates the parameters of the equivalent Bernoulli mixture model,
    with `n_genes+1` components."""
    coverage = parameters.coverage

    weights = jnp.concatenate(
        [jnp.array([1.0 - coverage]), jnp.full((n_genes,), coverage / n_genes)]
    )

    impurity = parameters.impurity
    components = jnp.concatenate(
        [jnp.zeros((1, n_genes)), impurity + (1.0 - impurity) * jnp.eye(n_genes)],
        axis=0,
    )

    return weights, bmm.adjust_mixture_components_for_noise(
        mixture_components=components,
        false_positive_rate=parameters.false_positive_rate,
        false_negative_rate=parameters.false_negative_rate,
    )


def _calculate_c(
    k: Int[Array, " G+1"],
    log_f: Float[Array, " G+1"],
    d: _FloatLike,
    params: Parameters,
) -> Float[Array, " G+1"]:
    G = k.shape[0] - 1
    beta = params.false_negative_rate
    gamma = params.coverage

    first_term = jnp.log(gamma) + (k - 1) * jnp.log(d) + (G - k - 1) * jnp.log1p(-d)
    bracket = k * (1.0 - beta) * (1.0 - d) + (G - k) * beta * d

    denominator = jnp.log(G) + log_f
    return bracket * jnp.exp(first_term - denominator)


def _calculate_t0(
    k: Int[Array, " G+1"],
    log_f: Float[Array, " G+1"],
    d: _FloatLike,
    params: Parameters,
) -> Float[Array, " G+1"]:
    G = k.shape[0] - 1
    beta = params.false_negative_rate
    gamma = params.coverage
    delta = params.impurity

    bracket = (
        beta
        * gamma
        * (
            d * (1 - d)
            + k * delta * (1 - beta) * (1 - d)
            + (G - k - 1) * delta * beta * d
        )
    )

    log_num = (k - 1) * jnp.log(d) + (G - k - 2) * jnp.log1p(-d)
    log_den = log_f + jnp.log(G)

    return bracket * jnp.exp(log_num - log_den)


def _calculate_t1(
    k: Int[Array, " G+1"],
    log_f: Float[Array, " G+1"],
    d: _FloatLike,
    params: Parameters,
) -> Float[Array, " G+1"]:
    G = k.shape[0] - 1
    beta = params.false_negative_rate
    gamma = params.coverage
    delta = params.impurity

    term1 = (
        (1 - beta)
        * gamma
        * (
            d * (1 - d)
            + (k - 1) * delta * (1 - beta) * (1 - d)
            + (G - k) * delta * beta * d
        )
    )

    log_den = jnp.log(G) + log_f
    log_num = (k - 2) * jnp.log(d) + (G - k - 1) * jnp.log1p(-d)

    return term1 * jnp.exp(log_num - log_den)


def _calculate_h0(
    k: Int[Array, " G+1"],
    f: Float[Array, " G+1"],
    d: _FloatLike,
    params: Parameters,
) -> Float[Array, " G+1"]:
    G = k.shape[0] - 1
    beta = params.false_negative_rate
    gamma = params.coverage

    numerator = gamma * beta * jnp.power(d, k) * jnp.power(1.0 - d, G - k - 1)
    denominator = G * f

    return numerator / denominator


def _calculate_h1(
    k: Int[Array, " G+1"],
    f: Float[Array, " G+1"],
    d: _FloatLike,
    params: Parameters,
) -> Float[Array, " G+1"]:
    G = k.shape[0] - 1
    beta = params.false_negative_rate
    gamma = params.coverage

    numerator = gamma * (1 - beta) * jnp.power(d, k - 1) * jnp.power(1.0 - d, G - k)
    denominator = G * f
    return numerator / denominator


class _EMStepResult(NamedTuple):
    new_params: Parameters
    new_loglikelihood: _FloatLike
    old_loglikelihood: _FloatLike


@jax.jit
def em_step(
    *,
    params: Parameters,
    k: Int[Array, " G+1"],
    counts: Int[Array, " G+1"],
    _jitter: float = 1e-8,
):
    G = k.shape[0] - 1
    N = jnp.sum(counts)

    log_f = _calculate_loglikelihood_single_point(
        k=k,
        G=G,
        alpha=params.false_positive_rate,
        beta=params.false_negative_rate,
        gamma=params.coverage,
        delta=params.impurity,
    )
    old_loglikelihood = jnp.dot(log_f, counts)

    d = _calculate_d(
        alpha=params.false_positive_rate,
        beta=params.false_negative_rate,
        delta=params.impurity,
    )
    d = jnp.where(d > _jitter, d, _jitter)

    c = _calculate_c(k=k, log_f=log_f, d=d, params=params)
    c = jnp.where(c > _jitter, c, _jitter)

    t0 = _calculate_t0(k=k, log_f=log_f, d=d, params=params)
    t0 = jnp.where(t0 > _jitter, t0, _jitter)

    t1 = _calculate_t1(k=k, log_f=log_f, d=d, params=params)
    t1 = jnp.where(t1 > _jitter, t1, _jitter)

    # h0 = _calculate_h0(k=k, f=f, d=d, params=params)
    # h1 = _calculate_h1(k=k, f=f, d=d, params=params)

    s = k * t1 + (G - k) * t0

    gamma = jnp.sum(c * counts) / N

    _num_delta = jnp.sum(counts * (s - c))
    _den_delta = (G - 1) * jnp.sum(counts * c)
    delta = _num_delta / _den_delta

    _num_alpha = jnp.sum(counts * k * (1 - t1))
    _den_alpha = N * G - jnp.sum(counts * s)
    alpha = _num_alpha / _den_alpha
    alpha = jnp.where(alpha > _jitter, alpha, _jitter)

    _num_beta = jnp.sum(counts * (G - k) * t0)
    _den_beta = jnp.sum(counts * s)
    beta = _num_beta / _den_beta

    log_f_new = _calculate_loglikelihood_single_point(
        k=k,
        G=G,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
    )
    new_loglikelihood = jnp.dot(log_f_new, counts)

    return _EMStepResult(
        new_params=Parameters(
            false_positive_rate=alpha,
            false_negative_rate=beta,
            coverage=gamma,
            impurity=delta,
        ),
        new_loglikelihood=new_loglikelihood,
        old_loglikelihood=old_loglikelihood,
    )


def em_algorithm(
    Y: Int[Array, "N G"],
    params0: Parameters,
    max_iter: int = 1_000,
    threshold: float = 1e-5,
) -> Tuple[Parameters, list]:
    counts = _calculate_summary_statistic(Y)
    k = jnp.arange(Y.shape[-1] + 1)

    trajectory = [
        _EMStepResult(
            new_params=params0,
            new_loglikelihood=-float("inf"),
            old_loglikelihood=-float("inf"),
        )
    ]
    params = params0
    for _ in range(max_iter):
        result = em_step(params=params, k=k, counts=counts)

        params = result.new_params
        trajectory.append(result)

        if abs(result.new_loglikelihood - result.old_loglikelihood) < threshold:
            break

    return trajectory[-1].new_params, trajectory
