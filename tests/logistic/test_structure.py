import time

import jax
import jax.numpy as jnp
import pytest
from jax import random

from jnotype.logistic.logreg import calculate_logits
import jnotype.logistic._structure as _str

import seaborn as sns
import matplotlib.pyplot as plt


def test_sample_structure(
    tmp_path,
    save_artifact: bool,
    seed: int = 10,
    n_features: int = 20,
    n_covariates: int = 3,
    n_points: int = 5_000,
    threshold: float = 0.1,
    n_iterations: int = 100,
) -> None:
    """Test whether the sampling can rediscover
    the true structure matrix, up to threshold."""
    key = random.PRNGKey(seed)

    intercepts = jnp.zeros(n_features)

    # True variances
    variances = jnp.linspace(0.5, 3, n_covariates)
    # Pseudoprior variance
    pseudoprior = 0.01

    gamma = 0.3
    key, subkey = random.split(key)
    true_structure = jnp.asarray(
        random.bernoulli(subkey, gamma, shape=(n_features, n_covariates)), dtype=int
    )
    assert true_structure.shape == (n_features, n_covariates)

    # Covariates, let's make these normally distributed
    key, subkey = random.split(key)
    covariates = 5 * random.normal(subkey, shape=(n_points, n_covariates))

    # Sample coefficients
    key, *subkeys = random.split(key, 3)
    coefs_on = jnp.sqrt(variances)[None, :] * random.normal(
        subkeys[0], shape=(n_features, n_covariates)
    )
    coefs_off = jnp.sqrt(pseudoprior) * random.normal(
        subkeys[1], shape=(n_features, n_covariates)
    )

    coefficients = coefs_on * true_structure + coefs_off * (1 - true_structure)

    logits = calculate_logits(
        intercepts=intercepts,
        coefficients=coefficients,
        covariates=covariates,
        structure=true_structure,
    )
    assert logits.shape == (n_points, n_features)

    probs = jax.nn.sigmoid(logits)

    key, subkey = random.split(key)

    observed = jnp.asarray(random.bernoulli(subkey, probs), dtype=int)
    assert observed.shape == (n_points, n_features)

    # We have simulated the data set from the true model.
    # For enough features and data points we should hope that
    # the sampled structures will be somewhat similar to the true ones
    structure = jnp.ones_like(true_structure, dtype=int)

    key, *subkeys = random.split(key, n_iterations + 1)

    sample = jax.jit(_str.sample_structure)

    t0 = time.time()

    samples = []

    for subkey in subkeys:
        structure = sample(
            key=subkey,
            intercepts=intercepts,
            covariates=covariates,
            coefficients=coefficients,
            structure=structure,
            observed=observed,
            variances=variances,
            pseudoprior_variance=pseudoprior,
            gamma=gamma,
        )
        samples.append(structure)

    burnin = n_iterations // 2
    posterior_mean = jnp.asarray(samples[burnin:]).mean(axis=0)
    assert posterior_mean.shape == true_structure.shape

    equal = (jnp.abs(posterior_mean - true_structure) <= threshold).sum()
    larger = (posterior_mean > true_structure + threshold).sum()
    smaller = (posterior_mean < true_structure - threshold).sum()

    assert equal + larger + smaller == posterior_mean.ravel().shape[0]

    title = (
        f"Equal: {equal} "
        f"Larger: {larger} "
        f"Smaller: {smaller}\n"
        f"Time/sample: {(time.time() - t0) / n_iterations:.2f}"
    )

    if save_artifact:
        directory = tmp_path / "test_sample_structure"
        directory.mkdir()

        fig, axs = plt.subplots(1, 3)
        sns.heatmap(
            posterior_mean - true_structure,
            ax=axs[0],
            vmin=-threshold,
            vmax=threshold,
            cmap="seismic",
        )
        sns.heatmap(posterior_mean, ax=axs[1], vmin=0, vmax=1, cmap="jet")
        axs[0].set_title("Difference")
        axs[1].set_title("Inferred")
        axs[2].set_title("True")
        sns.heatmap(true_structure, ax=axs[2], vmin=0, vmax=1, cmap="jet")

        fig.suptitle(title)

        fig.tight_layout()
        fig.savefig(directory / "heatmap.pdf")

    # Assert that the number of wrong entries is small
    assert (larger + smaller) < 0.05 * equal


@pytest.mark.parametrize("n_samples", (1000,))
@pytest.mark.parametrize("n_ones", (3, 10, 95))
@pytest.mark.parametrize("prior", [(1.0, 1.0), (3.0, 10.0)])
def test_sample_gamma(
    n_samples: int,
    n_ones: int,
    prior: tuple[float, float],
) -> None:
    """Basic tests for sampling from beta distribution."""
    alpha, beta = prior

    key = random.PRNGKey(32)
    subkeys = random.split(key, n_samples)

    structure = jnp.zeros((10, 10), dtype=int)

    n_set = 0
    for i in range(10):
        if n_set >= n_ones:
            break
        for j in range(10):
            structure = structure.at[i, j].set(1)
            n_set += 1
            if n_set >= n_ones:
                break

    assert structure.sum() == n_ones

    def aux_sample_gamma(key):
        """Partial application of `sample_gamma`,
        so we can get multiple samples by using `vmap`."""
        return _str.sample_gamma(
            key=key,
            structure=structure,
            prior_a=alpha,
            prior_b=beta,
        )

    samples = jax.vmap(aux_sample_gamma)(subkeys)
    assert len(samples) == n_samples

    # Calculate the analytic variance and mean
    a = alpha + n_ones
    b = beta + 10 * 10 - n_ones
    analytic_mean = a / (a + b)
    analytic_var = a * b / (a + b) ** 2 / (a + b + 1)

    assert jnp.mean(samples) == pytest.approx(analytic_mean, abs=0.01)
    assert jnp.var(samples) == pytest.approx(analytic_var, rel=0.05)
