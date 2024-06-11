"""False positives experiment (no latent structure)."""
import dataclasses
from typing import Literal

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
    "small": Config(
        n_x=3,
        probs_x=np.array([0.55, 0.4, 0.25]),
        n_samples=200,
    ),
    "medium": Config(
        n_x=3,
        probs_x=np.array([0.55, 0.4, 0.25]),
        n_samples=1_000,
    ),
    "large": Config(
        n_x=3,
        probs_x=np.array([0.55, 0.4, 0.25]),
        n_samples=5_000,
    )
}

N_SEEDS: int = 20

workdir: "generated/false_positives"

rule all:
    input: expand("{analysis}/variances_summary-{threshold}.npz", analysis=CONFIGS.keys(), threshold=[0, 0.01, 0.03, 0.05])


def construct_required_files(analysis):
    config = CONFIGS[analysis]
    return [f"{analysis}/pyramids/{seed}.nc" for seed in range(N_SEEDS)]


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

        # Generate (discrete) observed covariates
        observed_covariates = rng.binomial(1, config.probs_x, size=(config.n_samples, config.n_x))
        # Reorder the samples
        observed_covariates = magic_sort(observed_covariates)
        
        # Merge latent factors and observed covariates
        true_characteristics = observed_covariates

        n_genes_per_covariate = config.n_genes_per_covariate
        n_additional_genes = config.n_additional_genes

        n_all = config.n_x
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
            X=observed_covariates,
            Y=Y,
            coefs=coefs,
        )

        # Save figures
        fig, axs = plt.subplots(1, 3)

        ax = axs[0]
        sns.heatmap(true_characteristics, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=False)
        ax.set_title("Patient traits $X^*$")

        ax = axs[1]
        sns.heatmap(coefs, cmap="bwr", center=0, ax=ax)
        ax.set_title("True coefficients")

        ax = axs[2]
        sns.heatmap(Y, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=False)
        ax.set_title("Gene mutations $Y$")

        fig.tight_layout()
        fig.savefig(output.heatmap)


rule fit_pyramid:
    input: "{analysis}/data/{seed}.npz"
    output:
        pyramid_samples="{analysis}/pyramids/{seed}.nc", 
    run:
        input_arrays = np.load(input[0])
        X = input_arrays["X"]
        Y = input_arrays["Y"]
        
        config = CONFIGS[wildcards.analysis]

        # Now, fit the pyramid
        dataset = ListDataset(thinning=5, dimensions=TwoLayerPyramidSamplerNonparametric.dimensions())

        sampler = TwoLayerPyramidSamplerNonparametric(
            datasets=[dataset],
            observed=Y,
            observed_covariates=X,
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


rule calculate_variances:
    input:
        lambda wildcards: [f"{wildcards.analysis}/pyramids/{seed}.nc" for seed in range(N_SEEDS)]
    output:
        variances="{analysis}/variances_summary-{threshold}.npz"
    run:
        latent_variances = []
        observed_variances = []
        for inp_path in input:
            samples = xr.open_dataset(inp_path)
            
            latent_traits_probs = samples["latent_traits"].mean(axis=0).values
            # Now we need to remove "wrong" latent traits.
            # By "wrong" we will understand the following:
            #   - It appears in too few patients.
            #   - It has very small variance (i.e., uncertainty of it for all patients is almost identical)
            #   - The variance of associated coefficients is too small. (I.e., it's inactive)   
            threshold = float(wildcards.threshold)

            is_too_rare = (np.mean(latent_traits_probs, axis=0) < threshold) | (np.mean(latent_traits_probs, axis=0) > 1 - threshold)
            is_constant = np.std(latent_traits_probs, axis=0) < threshold
            is_wrong = is_too_rare | is_constant

            n_points = latent_traits_probs.shape[0]

            lat = samples["latent_variances"].mean(axis=0).values
            lat = lat * (~is_wrong)

            # Shape (n_latent_traits,)
            obs = samples["observed_variances"].mean(axis=0).values
            latent_variances.append(lat)
            observed_variances.append(obs)

        np.savez(
            output.variances,
            latent_variances=np.asarray(latent_variances),
            observed_variances=np.asarray(observed_variances),
            threshold=np.asarray(threshold),
            n_points = np.asarray(n_points),
        )
