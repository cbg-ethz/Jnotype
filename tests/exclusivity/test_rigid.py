import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp

from jaxtyping import Float, Array, Int

import jnotype.exclusivity._rigid as muex
import jnotype.exclusivity._bernoulli_mixtures as bmm
from jnotype.exclusivity._rigid import _FloatLike

import pytest


@pytest.mark.parametrize("n_genes", [3, 5])
def test_bincount(n_genes: int):
    counts = jnp.asarray([1, 1, 1, 3, 3])

    output = muex._bincount(counts, n_genes=n_genes)
    assert output.shape == (n_genes + 1,), "The shape is wrong."

    assert output[0] == 0
    assert output[1] == 3
    assert output[2] == 0
    assert output[3] == 2


@pytest.mark.parametrize("n_genes", [3, 5])
@pytest.mark.parametrize(
    "params",
    [
        muex.Parameters(
            false_positive_rate=0.05,
            false_negative_rate=0.1,
            coverage=0.6,
            impurity=0.15,
        ),
        muex.Parameters(
            false_positive_rate=0.8, false_negative_rate=0.7, coverage=0.1, impurity=0.6
        ),
    ],
)
def test_convert_to_bernoulli_mixture(n_genes: int, params: muex.Parameters) -> None:
    weights, mixture_components = muex.convert_to_bernoulli_mixture(
        parameters=params, n_genes=n_genes
    )

    coverage = params.coverage
    weights_ = jnp.asarray(
        [1.0 - coverage] + [coverage / n_genes for _ in range(n_genes)]
    )

    _comp = np.zeros((n_genes + 1, n_genes), dtype=float)
    _comp[1:, :] = params.impurity + (1.0 - params.impurity) * np.eye(n_genes)

    mixture_components_ = bmm.adjust_mixture_components_for_noise(
        mixture_components=jnp.asarray(_comp),
        false_positive_rate=params.false_positive_rate,
        false_negative_rate=params.false_negative_rate,
    )

    npt.assert_allclose(weights, weights_)
    npt.assert_allclose(mixture_components, mixture_components_)


@pytest.mark.parametrize("n_samples", [10, 20])
@pytest.mark.parametrize("n_genes", [3, 5])
@pytest.mark.parametrize(
    "params",
    [
        muex.Parameters(
            false_positive_rate=0.05,
            false_negative_rate=0.1,
            coverage=0.6,
            impurity=0.15,
        ),
        muex.Parameters(
            false_positive_rate=0.8, false_negative_rate=0.7, coverage=0.1, impurity=0.6
        ),
    ],
)
@pytest.mark.parametrize("seed", [101])
def test_loglikelihood(
    n_samples: int, n_genes: int, params: muex.Parameters, seed: int
) -> None:

    Y = jax.random.bernoulli(
        jax.random.PRNGKey(seed), p=0.1, shape=(n_samples, n_genes)
    )
    loglikelihood_fn = jax.jit(muex.get_loglikelihood_function(Y, from_params=True))

    ll1 = loglikelihood_fn(params)

    weights, components = muex.convert_to_bernoulli_mixture(
        parameters=params, n_genes=n_genes
    )
    ll2 = bmm.loglikelihood_bernoulli_mixture(
        Y, mixture_weights=weights, mixture_components=components
    )

    assert float(ll1) == pytest.approx(float(ll2))


def _calculate_c(
    k: Int[Array, " G+1"],
    f: Float[Array, " G+1"],
    d: _FloatLike,
    params: muex.Parameters,
) -> Float[Array, " G+1"]:
    G = k.shape[0] - 1
    beta = params.false_negative_rate
    gamma = params.coverage

    first_term = gamma * jnp.power(d, k - 1) * jnp.power(1.0 - d, G - k - 1)
    bracket = k * (1.0 - beta) * (1.0 - d) + (G - k) * beta * d

    denominator = G * f
    return first_term * bracket / denominator


def _calculate_t0(
    k: Int[Array, " G+1"],
    f: Float[Array, " G+1"],
    d: _FloatLike,
    params: muex.Parameters,
) -> Float[Array, " G+1"]:
    G = k.shape[0] - 1
    beta = params.false_negative_rate
    gamma = params.coverage
    delta = params.impurity

    term1 = gamma / (G * f)
    term2 = beta * jnp.power(d, k - 1) * jnp.power(1 - d, G - k - 2)
    bracket = (
        d * (1 - d) + k * delta * (1 - beta) * (1 - d) + (G - k - 1) * delta * beta * d
    )

    return term1 * term2 * bracket


def _calculate_t1(
    k: Int[Array, " G+1"],
    f: Float[Array, " G+1"],
    d: _FloatLike,
    params: muex.Parameters,
) -> Float[Array, " G+1"]:
    G = k.shape[0] - 1
    beta = params.false_negative_rate
    gamma = params.coverage
    delta = params.impurity

    term1 = gamma / (G * f)
    term2 = (1 - beta) * jnp.power(d, k - 2) * jnp.power(1 - d, G - k - 1)
    bracket = (
        d * (1 - d)
        + (k - 1) * delta * (1 - beta) * (1 - d)
        + (G - k) * delta * beta * d
    )

    return term1 * term2 * bracket


@pytest.mark.parametrize("G", [2, 5])
@pytest.mark.parametrize("N", [3, 4])
def calculate_auxiliary(G: int, N: int):
    jax.random.bernoulli(jax.random.PRNGKey(42), p=0.5, shape=(N, G))

    k = jnp.arange(G + 1)
    params = muex.Parameters(
        false_positive_rate=0.1,
        false_negative_rate=0.1,
        coverage=0.5,
        impurity=0.1,
    )

    log_f = muex._calculate_loglikelihood_single_point(
        k=k,
        G=G,
        alpha=params.false_positive_rate,
        beta=params.false_negative_rate,
        gamma=params.coverage,
        delta=params.impurity,
    )
    f = jnp.exp(log_f)

    d = muex._calculate_d(
        alpha=params.false_positive_rate,
        beta=params.false_negative_rate,
        delta=params.impurity,
    )

    c_expected = _calculate_c(k=k, f=f, d=d, params=params)
    t0_expected = _calculate_t0(k=k, f=f, d=d, params=params)
    t1_expected = _calculate_t1(k=k, f=f, d=d, params=params)

    c_obtained = muex._calculate_c(k=k, log_f=log_f, d=d, params=params)
    t0_obtained = muex._calculate_t0(k=k, log_f=log_f, d=d, params=params)
    t1_obtained = muex._calculate_t1(k=k, log_f=log_f, d=d, params=params)

    npt.assert_allclose(c_expected, c_obtained)
    npt.assert_allclose(t0_expected, t0_obtained)
    npt.assert_allclose(t1_expected, t1_obtained)
