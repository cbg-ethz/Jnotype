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
def test_sample_coefficients_trivial_structure(
    save_artifact,
    tmp_path,
    n_features: int,
    seed: int = 30,
    n_points: int = 5_500,
    n_covariates: int = 3,
    n_gibbs_steps: int = 1_000,
) -> None:
    """This function tests the `sample_coefficients` function
    assuming that the structure matrix has only 1s
    (no sampling from pseudoprior)."""
    key_dataset, key_sampling = random.split(random.PRNGKey(seed))

    # Prepare the data set
    rng_dataset = JAXRNG(key_dataset)

    design_matrix = random.bernoulli(
        rng_dataset.key, 0.5, shape=(n_points, n_covariates)
    )
    true_coefficients = 0.8 * jnp.power(
        -1, random.bernoulli(rng_dataset.key, 0.5, shape=(n_features, n_covariates))
    )
    logits = pg._calculate_logits(
        coefficients=true_coefficients,
        covariates=design_matrix,
        structure=jnp.ones_like(true_coefficients),
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
            structure=jnp.ones_like(true_coefficients),
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


def test_sample_coefficients_nontrivial_structure(
    n_points: int = 3_000,
    n_gibbs_steps: int = 800,
    seed: int = 21,
) -> None:
    """In this unit test we `sample_coefficients` but have some entries
    in the structure matrix set to 0.
    Hence, they should be sampled from the (pseudo)prior,
    rather than the posterior.

    Let's try 2 covariates and 3 outputs.
    """
    n_covariates = 2
    true_coefficients = jnp.asarray(
        [
            [1.0, 1e9],
            [1e10, -2.0],
            [2e9, 4e9],
        ]
    )
    # These e9 can't be identified
    # and will be sampled just from the prior
    structure = jnp.asarray(
        [
            [1, 0],
            [0, 1],
            [0, 0],
        ]
    )

    key_dataset, key_sampling = random.split(random.PRNGKey(seed))

    # Prepare the data set
    rng_dataset = JAXRNG(key_dataset)
    design_matrix = random.bernoulli(
        rng_dataset.key, 0.5, shape=(n_points, n_covariates)
    )
    logits = pg._calculate_logits(
        coefficients=true_coefficients,
        covariates=design_matrix,
        structure=structure,
    )
    observed = jnp.asarray(
        random.bernoulli(rng_dataset.key, jax.nn.sigmoid(logits)), dtype=int
    )

    prior_mean = jnp.asarray(
        [
            [0.5, 10.0],
            [20.0, -1.5],
            [30.0, 40.0],
        ]
    )

    prior_variance = jnp.asarray(
        [[0.6, 0.02**2], [0.03**2, 0.5], [0.04**2, 0.05**2]]
    )

    current_coefficients = jnp.zeros_like(true_coefficients)

    all_samples = []
    subkeys = random.split(key_sampling, n_gibbs_steps)
    numpy_rng = np.random.default_rng(seed + 152)
    t0 = time.time()

    for subkey in subkeys:
        current_coefficients = pg.sample_coefficients(
            jax_key=subkey,
            numpy_rng=numpy_rng,
            observed=observed,
            design_matrix=design_matrix,
            coefficients_prior_mean=prior_mean,
            coefficients_prior_variance=prior_variance,
            coefficients=current_coefficients,
            structure=structure,
        )
        assert current_coefficients.shape == true_coefficients.shape
        all_samples.append(current_coefficients)

    delta_t = time.time() - t0

    print(f"Speed: {n_gibbs_steps/delta_t:.2f} steps/second.")

    burnin = n_gibbs_steps // 4
    all_samples = jnp.asarray(all_samples)[burnin:, ...]

    coefficients_mean = all_samples.mean(axis=0)
    coefficients_std = all_samples.std(axis=0)

    # What we expect: well-determined terms
    # are close to their true values,
    # non-determined values have means of their priors
    wanted_mean = jnp.asarray(
        [
            [1.0, 10.0],
            [20.0, -2.0],
            [30.0, 40.0],
        ]
    )
    nptest.assert_allclose(coefficients_mean, wanted_mean, atol=0.1)

    # For non-determined values we also know that the standard deviation
    # should be the one of the prior
    assert coefficients_std[0, 1] == pytest.approx(0.02, rel=0.1)
    assert coefficients_std[1, 0] == pytest.approx(0.03, rel=0.1)
    assert coefficients_std[2, 0] == pytest.approx(0.04, rel=0.1)
    assert coefficients_std[2, 1] == pytest.approx(0.05, rel=0.1)

    # assert False, f"\n--- COEFFICIENTS: ---\n{coefficients_mean}\n{coefficients_std}"


def test_augment_matrices() -> None:
    """Tests whether design matrix augmentation works properly.

    We have n=5 points with k=2 covariates each and three outputs f=3.
    """
    intercepts = jnp.asarray([51, 52, 53])
    structure = jnp.asarray(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    coefficients = jnp.asarray(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
    )
    covariates = jnp.asarray(
        [
            [0, 1],
            [10, 11],
            [20, 21],
            [30, 31],
            [40, 41],
        ]
    )
    intercept_prior_mean = 0.8
    intercept_prior_variance = 2.0
    variances = jnp.asarray([5.0, 12.0])
    pseudoprior_variance = 0.01

    matrices = pg._augment_matrices(
        intercepts=intercepts,
        covariates=covariates,
        coefficients=coefficients,
        structure=structure,
        pseudoprior_variance=pseudoprior_variance,
        variances=variances,
        intercept_prior_mean=intercept_prior_mean,
        intercept_prior_variance=intercept_prior_variance,
    )

    nptest.assert_allclose(
        matrices["covariates"],
        jnp.asarray(
            [
                [1, 0, 1],
                [1, 10, 11],
                [1, 20, 21],
                [1, 30, 31],
                [1, 40, 41],
            ]
        ),
    )
    nptest.assert_allclose(
        matrices["structure"],
        jnp.asarray(
            [
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
            ]
        ),
    )
    nptest.assert_allclose(
        matrices["coefficients"],
        jnp.asarray(
            [
                [51, 0.1, 0.2],
                [52, 0.3, 0.4],
                [53, 0.5, 0.6],
            ]
        ),
    )
    np.asarray(
        matrices["prior_mean"],
        jnp.asarray(
            [
                [intercept_prior_mean, 0, 0],
                [intercept_prior_mean, 0, 0],
                [intercept_prior_mean, 0, 0],
            ]
        ),
    )

    np.asarray(
        matrices["prior_variance"],
        jnp.asarray(
            [
                [intercept_prior_variance, 5.0, pseudoprior_variance],
                [intercept_prior_variance, pseudoprior_variance, 12.0],
                [intercept_prior_variance, 5.0, 12.0],
            ]
        ),
    )


@pytest.mark.parametrize("n_points", [10, 30])
@pytest.mark.parametrize("n_outputs", [1, 3])
@pytest.mark.parametrize("n_covariates", [1, 2])
def test_sample_smoke_test(n_points: int, n_covariates: int, n_outputs: int) -> None:
    """This is the high-level sampling function, which is composed
    out of the building blocks above.

    TODO(Pawel): This is just a smoke test, more elaborate tests to follow.
    """
    observed = jnp.ones((n_points, n_outputs), dtype=int)
    intercepts = jnp.zeros(n_outputs)
    coefficients = jnp.zeros((n_outputs, n_covariates))
    structure = jnp.ones_like(coefficients, dtype=int)
    covariates = jnp.ones((n_points, n_covariates))
    variances = jnp.ones(n_covariates)

    new_intercepts, new_coefficients = pg.sample_intercepts_and_coefficients(
        jax_key=random.PRNGKey(21),
        numpy_rng=np.random.default_rng(10),
        observed=observed,
        intercepts=intercepts,
        coefficients=coefficients,
        structure=structure,
        covariates=covariates,
        variances=variances,
        pseudoprior_variance=0.01,
        intercept_prior_mean=0,
        intercept_prior_variance=1.0,
    )

    assert intercepts.shape == new_intercepts.shape
    assert coefficients.shape == new_coefficients.shape
