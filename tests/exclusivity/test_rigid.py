import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp

import jnotype.exclusivity._rigid as muex
import jnotype.exclusivity._bernoulli_mixtures as bmm

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
        false_positive_rate=params.false_positive_rate,  # type: ignore
        false_negative_rate=params.false_negative_rate,  # type: ignore
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
    loglikelihood_fn = jax.jit(muex.get_loglikelihood_function(Y))

    ll1 = loglikelihood_fn(
        params.false_positive_rate,
        params.false_negative_rate,
        params.coverage,
        params.impurity,
    )
    weights, components = muex.convert_to_bernoulli_mixture(
        parameters=params, n_genes=n_genes
    )
    ll2 = bmm.loglikelihood_bernoulli_mixture(
        Y, mixture_weights=weights, mixture_components=components
    )

    assert float(ll1) == pytest.approx(float(ll2))
