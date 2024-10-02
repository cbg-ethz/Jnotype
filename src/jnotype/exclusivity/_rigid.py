"""Model proposed by Szczurek and Beerenwinkel (2014)."""

import jax.numpy as jnp
from jaxtyping import Float, Int, Array
from typing import NamedTuple, Tuple

import jnotype.exclusivity._bernoulli_mixtures as bmm


def _bincount(counts: Int[Array, " counts"], n_genes: int) -> Int[Array, " n_genes+1"]:
    """Calculates occurences of different values from the set {0, 1, ..., n_genes}."""
    return jnp.bincount(counts, minlength=n_genes + 1, length=n_genes + 1)


def calculate_summary_statistic(
    Y: Int[Array, "n_samples n_genes"]
) -> Int[Array, " n_genes+1"]:
    """Calculates the summary statistic, counting the occurences of 1s."""
    return _bincount(counts=Y.sum(axis=-1), n_genes=Y.shape[-1])


def _calculate_d(alpha: float, beta: float, delta: float) -> float:
    """Calculates auxiliary parameter d."""
    return delta * (1.0 - beta) + (1.0 - delta) * alpha


def calculate_loglikelihood_single_point(
    k: int,
    G: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> float:
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


def get_loglikelihood_function_from_counts(counts: Int[Array, " n_genes+1"]):
    """Factory for the likelihood function,
    which uses an easy-to-calculate
    summary statistic of the data,
    improving the computation speed."""
    G = counts.shape[0] - 1

    ks = jnp.arange(counts.shape[0])
    assert ks.shape == counts.shape

    def f(alpha: float, beta: float, gamma: float, delta: float) -> float:
        lls = calculate_loglikelihood_single_point(
            k=ks,  # type: ignore
            G=G,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        return jnp.dot(counts, lls)

    return f


def get_loglikelihood_function(Y: Int[Array, "n_samples n_genes"]):
    """Factory for the likelihood function,
    which uses an easy-to-calculate
    summary statistic of the data,
    improving the computation speed."""
    counts = calculate_summary_statistic(Y)
    return get_loglikelihood_function_from_counts(counts)


_FloatLike = Float[Array, " "]


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


def estimate_no_errors(Y: Int[Array, "n_samples n_genes"]) -> Parameters:
    """Estimates the parameter values assuming no errors in the data,
    i.e., FPR = FNR = 0."""
    zero = jnp.asarray(0.0)

    statistic = calculate_summary_statistic(Y)
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
