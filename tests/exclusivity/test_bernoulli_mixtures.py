import pytest
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import jnotype.exclusivity._bernoulli_mixtures as bmm
from jnotype._utils import JAXRNG


@pytest.mark.parametrize("n_genes", [3, 5])
@pytest.mark.parametrize("seed", [42])
def test_likelihood(seed: int, n_genes: int) -> None:
    rng = JAXRNG(jax.random.PRNGKey(seed))

    Y1 = jax.random.bernoulli(rng.key, p=0.6, shape=(2, 7, n_genes))
    Y2 = jnp.vstack(Y1)
    assert Y2.shape == (Y1.shape[0] * Y1.shape[1], Y1.shape[2])

    mixture_weights = jnp.asarray([0.2, 0.3, 0.5])
    assert float(jnp.sum(mixture_weights)) == pytest.approx(1.0)

    mixture_components = jnp.asarray(
        [
            jnp.linspace(0.1, 0.3, n_genes),
            jnp.linspace(0.7, 0.9, n_genes),
            jnp.linspace(0.5, 0.6, n_genes),
        ]
    )

    loglike1 = bmm.loglikelihood_bernoulli_mixture(
        Y1, mixture_weights=mixture_weights, mixture_components=mixture_components
    )
    loglike2 = bmm.loglikelihood_bernoulli_mixture(
        Y2, mixture_weights=mixture_weights, mixture_components=mixture_components
    )

    assert float(loglike2) == pytest.approx(float(loglike1))

    def loglike_point(y):
        log_probs = jnp.log(mixture_components)
        log_1probs = jnp.log1p(-mixture_components)

        ls = [
            jnp.sum(y * log_probs[i, :] + (1 - y) * log_1probs[i, :])
            for i in range(mixture_components.shape[0])
        ]
        ls = jnp.asarray(ls)
        loglike = logsumexp(ls + jnp.log(mixture_weights))
        return loglike

    loglike_true = jnp.sum(jax.vmap(loglike_point)(Y2))
    assert float(loglike2) == pytest.approx(float(loglike_true))
