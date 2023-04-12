import time

import jax
import jax.numpy as jnp
from jax import random

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import numpy.testing as nptest

import pytest

import jnotype.logistic._polyagamma as pg


class JAXRNG:
    def __init__(self, key: random.PRNGKeyArray) -> None:
        self._key = key

    @property
    def key(self) -> random.PRNGKeyArray:
        key, subkey = random.split(self._key)
        self._key = key
        return subkey


@pytest.mark.parametrize("n_features", [1, 2])
def test_sample_coefficients(
    save_artifact,
    tmp_path,
    n_features: int,
    seed: int = 30,
    n_points: int = 5_500,
    n_covariates: int = 3,
    n_gibbs_steps: int = 1_000,
) -> None:
    key_dataset, key_sampling = random.split(random.PRNGKey(seed))

    # Prepare the data set
    rng_dataset = JAXRNG(key_dataset)

    design_matrix = random.bernoulli(
        rng_dataset.key, 0.5, shape=(n_points, n_covariates)
    )
    true_coefficients = 0.8 * jnp.power(
        -1, random.bernoulli(rng_dataset.key, 0.5, shape=(n_features, n_covariates))
    )
    #   random.normal(rng_dataset.key, shape=(n_features, n_covariates))
    logits = pg._calculate_logits(
        coefficients=true_coefficients, covariates=design_matrix
    )
    observed = jnp.asarray(
        random.bernoulli(rng_dataset.key, jax.nn.sigmoid(logits)), dtype=int
    )

    current_coefficients = jnp.zeros_like(true_coefficients)
    all_samples = []
    subkeys = random.split(key_sampling, n_gibbs_steps)
    numpy_rng = np.random.default_rng(seed + 21)
    t0 = time.time()

    for subkey in subkeys:
        current_coefficients = pg.sample_coefficients(
            jax_key=subkey,
            numpy_rng=numpy_rng,
            observed=observed,
            design_matrix=design_matrix,
            coefficients_prior_mean=jnp.zeros_like(true_coefficients),
            coefficients_prior_variance=2 * jnp.ones_like(true_coefficients),
            coefficients=current_coefficients,
        )
        assert current_coefficients.shape == true_coefficients.shape
        all_samples.append(current_coefficients)

    delta_t = time.time() - t0

    print(f"Speed: {n_gibbs_steps/delta_t:.2f} steps/second.")

    burnin = n_gibbs_steps // 4
    all_samples = jnp.asarray(all_samples)[burnin:, ...]

    coefficients_mean = all_samples.mean(axis=0)
    coefficients_std = all_samples.std(axis=0)

    if save_artifact:
        directory = tmp_path / "test_sample_coefficients"
        directory.mkdir()
        fig, axs = plt.subplots(1, 3)

        sns.heatmap(true_coefficients, ax=axs[0], cmap="seismic")
        sns.heatmap(coefficients_mean, ax=axs[1], cmap="seismic")
        sns.heatmap(coefficients_std, ax=axs[2], cmap="seismic")
        axs[0].set_title("True")
        axs[1].set_title("Posterior mean")
        axs[2].set_title("Posterior std")

        fig.tight_layout()
        fig.savefig(directory / "coefficient_matrix.pdf")

    nptest.assert_allclose(true_coefficients, coefficients_mean, atol=0.1)
