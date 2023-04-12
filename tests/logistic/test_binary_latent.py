import time

import numpy as np
import numpy.testing as nptest
import jax
from jax import random
import jax.numpy as jnp

import matplotlib.pyplot as plt
import pytest
import seaborn as sns

from jnotype.logistic._binary_latent import sample_binary_codes
from jnotype.logistic.logreg import calculate_logits


@pytest.mark.parametrize("n_points", [30])
@pytest.mark.parametrize("n_additional_covariates", [0, 2])
def test_sample_binary_codes(
    save_artifact,
    tmp_path,
    n_points: int,
    n_additional_covariates: int,
    seed: int = 42,
    n_binary_codes: int = 3,
    n_clusters: int = 2,
    n_features: int = 250,
    intercept: float = -0.2,
    coefficient_scale: float = 4.0,
    n_gibbs_samples: int = 1_000,
    n_anchor_per_covariate: int = 25,
) -> None:
    key = random.PRNGKey(seed)

    key, key_labels, key_mixing, key_codes, key_add_covs = random.split(key, 5)
    cluster_labels = random.categorical(
        key_labels, logits=jnp.zeros(n_clusters), shape=(n_points,)
    )
    mixing = random.beta(key_mixing, 2.0, 2.0, shape=(n_binary_codes, n_clusters))

    true_codes = jnp.asarray(
        random.bernoulli(key_codes, mixing[:, cluster_labels].T), dtype=float
    )
    assert true_codes.shape == (n_points, n_binary_codes)

    additional_covariates = 2 * random.normal(
        key_add_covs, shape=(n_points, n_additional_covariates)
    )

    all_covariates = jnp.hstack((true_codes, additional_covariates))
    assert all_covariates.shape == (n_points, n_binary_codes + n_additional_covariates)

    # Now create a sparse matrix from covariates to features so that each covariate
    # has three anchor features
    n_covariates = all_covariates.shape[1]

    assert n_features >= n_anchor_per_covariate * n_covariates
    structure = np.zeros((n_features, n_covariates), dtype=int)
    for i in range(n_covariates):
        indices = list(range(i, i + n_anchor_per_covariate))
        structure[indices, i] = 1

    # The rest of the structure matrix will be sampled from sparse Bernoulli
    numpy_rng = np.random.default_rng(seed + 5)
    additional_structure = numpy_rng.binomial(
        n=1,
        p=0.1,
        size=(n_features - n_anchor_per_covariate * n_covariates, n_covariates),
    )

    structure[n_anchor_per_covariate * n_covariates :, :] = additional_structure
    structure = jnp.asarray(structure, dtype=int)

    intercepts = jnp.full(n_features, fill_value=intercept, dtype=float)
    key, key_coef, key_observed = jax.random.split(key, 3)
    coefficients = coefficient_scale * random.normal(key_coef, shape=structure.shape)

    logits = calculate_logits(
        intercepts=intercepts,
        coefficients=coefficients,
        structure=structure,
        covariates=all_covariates,
    )
    probs = jax.nn.sigmoid(logits)
    assert probs.shape == (n_points, n_features)
    observed = random.bernoulli(key_observed, probs)

    key, *subkeys = jax.random.split(key, n_gibbs_samples + 1)

    @jax.jit
    def sample(key, covs):
        return sample_binary_codes(
            key=key,
            intercepts=intercepts,
            coefficients=coefficients,
            structure=structure,
            covariates=covs,
            n_binary_codes=n_binary_codes,
            labels_to_codes=mixing,
            labels=cluster_labels,
            observed=observed,
        )

    t0 = time.time()
    covariates_samples = []
    current_covariates = jnp.hstack((jnp.zeros_like(true_codes), additional_covariates))

    for subkey in subkeys:
        current_covariates = sample(key=subkey, covs=current_covariates)
        covariates_samples.append(current_covariates)

    delta_t = time.time() - t0
    burnin = n_gibbs_samples // 2

    covariates_samples = jnp.asarray(covariates_samples, dtype=float)[burnin:, ...]

    inferred_mean = covariates_samples.mean(axis=0)
    inferred_std = covariates_samples.std(axis=0)

    title = f"Time/sample: {delta_t / n_gibbs_samples:.3f}"

    if save_artifact:
        directory = tmp_path / "test_sample_binary_codes"
        directory.mkdir()

        # All covariates heatmap
        fig, axs = plt.subplots(1, 3)
        sns.heatmap(
            inferred_mean - all_covariates,
            ax=axs[0],
            vmin=-0.1,
            vmax=0.1,
            cmap="seismic",
        )
        sns.heatmap(inferred_mean, ax=axs[1], vmin=0, vmax=1, cmap="jet")
        axs[0].set_title("Difference")
        axs[1].set_title("Inferred")
        axs[2].set_title("True")
        sns.heatmap(all_covariates, ax=axs[2], vmin=0, vmax=1, cmap="jet")

        fig.suptitle(title)

        fig.tight_layout()
        fig.savefig(directory / "all_covariates.pdf")

        # Binary codes heatmap
        fig, axs = plt.subplots(1, 3)
        sns.heatmap(
            inferred_mean[:, :n_binary_codes] - true_codes,
            ax=axs[0],
            vmin=-0.1,
            vmax=0.1,
            cmap="seismic",
        )
        sns.heatmap(
            inferred_mean[:, :n_binary_codes], ax=axs[1], vmin=0, vmax=1, cmap="jet"
        )
        axs[0].set_title("Difference")
        axs[1].set_title("Inferred")
        axs[2].set_title("True")
        sns.heatmap(true_codes, ax=axs[2], vmin=0, vmax=1, cmap="jet")

        fig.suptitle(title)

        fig.tight_layout()
        fig.savefig(directory / "binary_codes.pdf")

    # The additional covariates should *not* be resampled at all
    if n_additional_covariates > 0:
        nptest.assert_allclose(
            inferred_mean[:, n_binary_codes:],
            all_covariates[:, n_binary_codes:],
            rtol=0.0001,
        )
        nptest.assert_allclose(
            inferred_std[:, n_binary_codes:],
            np.zeros_like(all_covariates[:, n_binary_codes:]),
            atol=0.0001,
        )
