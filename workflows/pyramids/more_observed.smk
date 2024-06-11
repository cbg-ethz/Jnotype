"""More observed variables."""
import dataclasses

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import seaborn as sns

from jnotype.pyramids import TwoLayerPyramidSampler, TwoLayerPyramidSamplerNonparametric
from jnotype.sampling import ListDataset


@dataclasses.dataclass
class Config:
    # Latent factors
    n_a: int
    probs_a: np.ndarray
    # Observed factors
    n_x: int
    probs_x: np.ndarray
    
    # Number of samples
    n_samples: int
    
    # Covariate matrix
    n_genes_per_covariate: int = 5
    n_additional_genes: int = 8
    effect_size: float = 4.0

    # MCMC setup
    n_warmup: int = 4_000
    n_steps: int = 1_000


CONFIGS = {
    "large_sample": Config(
        n_a=4,
        probs_a=np.array([0.55, 0.4, 0.25, 0.1]),
        n_x=3,
        probs_x=np.array([0.55, 0.4, 0.25]),
        n_samples=1_000,
    ),
    "small_sample": Config(
        n_a=4,
        probs_a=np.array([0.55, 0.4, 0.25, 0.1]),
        n_x=3,
        probs_x=np.array([0.55, 0.4, 0.25]),
        n_samples=200,
    ),
}

N_SEEDS: int = 20

workdir: "generated/more_observed/"

rule all:
    # input: expand("{analysis}/pyramids_fitted.done", analysis=CONFIGS.keys())
    input: expand("{analysis}/similarities/{n_known}-{seed}.npz", analysis=CONFIGS.keys(), seed=range(N_SEEDS), n_known=range(4))


def construct_required_files(analysis):
    config = CONFIGS[analysis]
    return [f"{analysis}/pyramids/{n_known}-{seed}.nc" for seed in range(N_SEEDS) for n_known in range(config.n_x + 1)]


rule fit_all_pyramids:
    output: touch("{analysis}/pyramids_fitted.done")
    input: lambda wildcards: construct_required_files(wildcards.analysis)


def magic_sort(x: np.ndarray) -> np.ndarray:
    """Reorders the samples in a binary matrix (n_samples, n_covariates),
    sorting by the binary features.
    """
    def get_key(a: np.ndarray) -> str:
        return "".join(map(str, a))
    idx = np.argsort(list(map(get_key, x)))
    return x[idx, :]


rule generate_data:
    output:
        arrays = "{analysis}/data/{seed}.npz",
        heatmap = "{analysis}/data/{seed}.pdf",
    run:
        config = CONFIGS[wildcards.analysis]
        rng = np.random.default_rng(int(wildcards.seed))

        probs_all = np.concatenate([config.probs_a, config.probs_x])
        n_all = len(probs_all)

        true_characteristics = rng.binomial(1, probs_all, size=(config.n_samples, n_all))
        # Reorder the samples
        true_characteristics = magic_sort(true_characteristics)        
        
        # Split into latent traits and observed covariates
        latent_traits = true_characteristics[:, :config.n_a]
        observed_covariates = true_characteristics[:, config.n_a:]

        n_genes_per_covariate = config.n_genes_per_covariate
        n_additional_genes = config.n_additional_genes

        n_genes = config.n_genes_per_covariate * n_all + n_additional_genes
        coefs = np.zeros((n_genes, n_all))
        effect_size = config.effect_size

        for i in range(n_all):
            coefs[i*n_genes_per_covariate:(i+1)*n_genes_per_covariate, i] = effect_size

        if n_additional_genes > 0:
            coefs[-n_additional_genes:, :] = effect_size * (-1) ** rng.binomial(1, 0.5, size=(n_additional_genes, n_all))

        offset = -5
        logits = offset + np.einsum("nf,gf->ng", true_characteristics, coefs)
        ps = 1/(1 + np.exp(-logits))
        Y = rng.binomial(1, ps)

        np.savez(
            output.arrays,
            A=latent_traits,
            X=observed_covariates,
            Y=Y,
            coefs=coefs,
        )

        # Save figures
        fig, axs = plt.subplots(1, 3)

        ax = axs[0]
        sns.heatmap(true_characteristics, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=False)
        ax.set_title("Patient traits $(A^*, X^*)$")

        ax = axs[1]
        sns.heatmap(coefs, cmap="bwr", center=0, ax=ax)
        ax.set_title("True coefficients")

        ax = axs[2]
        sns.heatmap(Y, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=False)
        ax.set_title("Gene mutations Y")

        fig.tight_layout()
        fig.savefig(output.heatmap)


rule fit_pyramid:
    input: "{analysis}/data/{seed}.npz"
    output:
        pyramid_samples="{analysis}/pyramids/{n_known}-{seed}.nc", 
    run:
        input_arrays = np.load(input[0])
        X = input_arrays["X"]
        Y = input_arrays["Y"]
        
        # We fit the pyramid using only selected covariates
        n_known = int(wildcards.n_known)
        if n_known == 0:
            covariates = np.zeros((X.shape[0], 1))
        else:
            covariates = X[:, :n_known].astype(np.float32)

        config = CONFIGS[wildcards.analysis]

        # Now, fit the pyramid
        dataset = ListDataset(thinning=5, dimensions=TwoLayerPyramidSamplerNonparametric.dimensions())

        sampler = TwoLayerPyramidSamplerNonparametric(
            datasets=[dataset],
            observed=Y,
            observed_covariates=covariates,
            dirichlet_prior=np.ones(2) / 2,
            max_binary_codes=10,
            expected_binary_codes=5.0,
            n_clusters=2,
            verbose=True,
            warmup=config.n_warmup,
            steps=config.n_steps,
            inactive_latent_variance_theta_inf = 0.1**2,
            mixing_beta_prior=(1.0, 5.0),
        )
        sampler.run()

        dataset.dataset.to_netcdf(output.pyramid_samples)


rule calculate_similarities:
    input:
        pyramid_samples="{analysis}/pyramids/{n_known}-{seed}.nc",
        data="{analysis}/data/{seed}.npz"
    output:
        similarities="{analysis}/similarities/{n_known}-{seed}.npz"
    run:
        samples = xr.open_dataset(input.pyramid_samples)
        data = np.load(input.data)

        # Shape (n_latent_traits,)
        latent_variances = samples["latent_variances"].mean(axis=0).values
        # Shape (n_samples, n_latent_traits)
        latent_traits_probs = samples["latent_traits"].mean(axis=0).values
        
        # Now we need to remove "wrong" latent traits.
        # By "wrong" we will understand the following:
        #   - It appears in too few patients.
        #   - It has very small variance (i.e., uncertainty of it for all patients is almost identical)
        #   - The variance of associated coefficients is too small. (I.e., it's inactive)   
        is_too_rare = (np.mean(latent_traits_probs, axis=0) < 0.01) | (np.mean(latent_traits_probs, axis=0) > 0.99)
        is_constant = np.std(latent_traits_probs, axis=0) < 0.01
        has_zero_variance = latent_variances < 0.05
        is_wrong = is_too_rare | is_constant | has_zero_variance

        latent_traits_probs = latent_traits_probs[:, ~is_wrong]

        # Now we have to calculate the correlations between the latent traits and the observed covariates

        def pearson_rho(a, b):
            return pd.DataFrame({"a": a, "b": b}).corr(method="pearson").iloc[0, 1]

        def correlation_matrix(X, Q):
            assert len(X) == len(Q)
            F = X.shape[1]
            K = Q.shape[1]

            arr = np.zeros((F, K))
            for f in range(F):
                for k in range(K):
                    arr[f, k] = pearson_rho(X[:, f], Q[:, k])
            return arr

        def sort_abs(a):
            return np.sort(np.abs(a), axis=1)[:, ::-1]

        def similarities(X, Q):
            return sort_abs(correlation_matrix(X, Q))

        np.savez(
            output.similarities,
            similarities_latent=similarities(data["A"], latent_traits_probs),
            similarities_observed=similarities(data["X"], latent_traits_probs),
            n_active=np.array(latent_traits_probs.shape[1]),
            n_known=np.array(int(wildcards.n_known)),
        )
