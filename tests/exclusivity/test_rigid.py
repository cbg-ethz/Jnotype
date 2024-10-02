import numpy as np
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


@pytest.mark.parametrize("n_samples", [3, 5])
@pytest.mark.parametrize("n_genes", [10, 20])
@pytest.mark.parametrize(
    "params",
    [
        muex.ParameterEstimates(
            false_positive_rate=0.05,
            false_negative_rate=0.1,
            coverage=0.6,
            impurity=0.15,
        ),
        muex.ParameterEstimates(
            false_positive_rate=0.8, false_negative_rate=0.7, coverage=0.1, impurity=0.6
        ),
    ],
)
@pytest.mark.parametrize("seed", [101])
def test_loglikelihood(
    n_samples: int, n_genes: int, params: muex.ParameterEstimates, seed: int
) -> None:
    coverage = params.coverage
    weights = jnp.asarray(
        [1.0 - coverage] + [coverage / n_genes for _ in range(n_genes)]
    )

    noiseless_components = np.zeros((n_genes + 1, n_genes), dtype=float)
    noiseless_components[1:, :] = params.impurity + (1.0 - params.impurity) * np.eye(
        n_genes
    )

    noisy_components = bmm.adjust_mixture_components_for_noise(
        mixture_components=jnp.asarray(noiseless_components),
        false_positive_rate=params.false_positive_rate,  # type: ignore
        false_negative_rate=params.false_negative_rate,  # type: ignore
    )

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
    ll2 = bmm.loglikelihood_bernoulli_mixture(
        Y, mixture_weights=weights, mixture_components=noisy_components
    )

    assert float(ll1) == pytest.approx(float(ll2))
