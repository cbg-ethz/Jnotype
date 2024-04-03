import dataclasses
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xarray as xr
import sys
import io
import joblib
import json
import seaborn as sns
from scipy.stats import chi2

from sklearn.preprocessing import StandardScaler, OneHotEncoder

import formulaic as fm
from lifelines import CoxPHFitter
from lifelines.calibration import survival_probability_calibration

from jnotype.pyramids import TwoLayerPyramidSampler, TwoLayerPyramidSamplerNonparametric
from jnotype.sampling import ListDataset

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive




matplotlib.use("agg")

workdir: "generated/presentation"

DPI: int = 500
FIGSIZE = (3 * 1.25, 2 * 1.25)

ANALYSES = {
  "GBM": ["GBM"],
  "BRCA": ["BRCA"],
  "COAD": ["COAD"],
}

@dataclasses.dataclass
class ModelSettings:
    true_loc: tuple[int, int]

MODELS = {
    "one_parameter_model": ModelSettings(true_loc=(1, 2)),
    "independent": ModelSettings(true_loc=(-1, 1)),
}

rule all:
    input: "analysis/COAD/everything.done"

rule run_all:
    input:
        basic_info = "analysis/{analysis}/basic_info/summary.json",
        one_parameter_model = "analysis/{analysis}/one_parameter_model/done.done",
        independent_model = "analysis/{analysis}/independent/done.done",
    output: touch("analysis/{analysis}/everything.done")


# === Basic statistics ===

rule basic_statistics:
    input: "data/preprocessed/{analysis}/mutation-matrix.csv"
    output:
        summary = "analysis/{analysis}/basic_info/summary.json",
        hist_n_mutations = "analysis/{analysis}/basic_info/hist_n_mutations.pdf",
        hist_mut_frequency = "analysis/{analysis}/basic_info/hist_mutation_frequency.pdf"
    run:
        df = pd.read_csv(str(input), index_col=0)
        mutations = df.values

        # Calculate basic summary
        summary = {
           "n_patients": mutations.shape[0],
           "n_genes": mutations.shape[1],
           "frequency": np.mean(mutations),
        }

        with open(output.summary, "w") as fh:
            json.dump(summary, fp=fh)


        # Plot a histogram of mutation frequencies
        fig, ax = plt.subplots(dpi=DPI, figsize=FIGSIZE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.hist(mutations.mean(axis=0), bins=np.linspace(-1e-5, 1+1e-5, 25), alpha=0.7)
        ax.set_xlabel("Mutation frequency")
        ax.set_ylabel("Number of genes")
        ax.axvline(np.mean(mutations), linestyle="--", color="black")
        fig.tight_layout()
        fig.savefig(output.hist_mut_frequency)

        # Plot a histogram of number of mutations per patient
        fig, ax = plt.subplots(dpi=DPI, figsize=FIGSIZE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.hist(mutations.sum(axis=1), bins=np.arange(-0.5, 1.5 + np.max(mutations.sum(axis=1))), alpha=0.7)
        ax.set_xlabel("Number of mutations")
        ax.set_ylabel("Number of patients")

        fig.tight_layout()
        fig.savefig(output.hist_n_mutations)


# === One-parameter model ===
# All genes share a single theta parameter, with the mutation frequency

rule one_parameter_all:
    input:
        prior_posterior_plot = "analysis/{analysis}/one_parameter_model/prior_posterior.pdf",
        posterior_histogram = "analysis/{analysis}/one_parameter_model/posterior.pdf",
        posterior_predictive_matrices = "analysis/{analysis}/one_parameter_model/posterior_predictive_matrices.pdf",
        posterior_predictive_occurrences = "analysis/{analysis}/one_parameter_model/posterior_predictive_occurrence.pdf",
    output: touch("analysis/{analysis}/one_parameter_model/done.done")


rule one_parameter_prior_posterior:
    input: "data/preprocessed/{analysis}/mutation-matrix.csv"
    output: "analysis/{analysis}/one_parameter_model/prior_posterior_samples.joblib"
    run:
        df = pd.read_csv(str(input), index_col=0)
        mutations = df.values
        rng = np.random.default_rng(42)
        prior_samples = rng.beta(a=1, b=1, size=50_000)
        posterior_samples = rng.beta(a=1 + mutations.sum(), b=1 + (1 - mutations).sum(), size=5_000)
        joblib.dump({"prior": prior_samples, "posterior": posterior_samples, "shape": mutations.shape}, str(output))


rule one_parameter_prior_posterior_histogram:
    input: "analysis/{analysis}/one_parameter_model/prior_posterior_samples.joblib"
    output: "analysis/{analysis}/one_parameter_model/prior_posterior.pdf"
    run:
        data = joblib.load(str(input))
        fig, axs = plt.subplots(1, 2, dpi=DPI, figsize=(6, 2.5), sharex=True)
        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("$\\theta$")

        bins = np.linspace(-1e-5, 1+1e-5, 25)

        ax = axs[0]
        ax.set_ylabel("Prior $P(\\theta)$")
        ax.hist(data["prior"], bins=bins, color="salmon", rasterized=True, density=True)
        
        ax = axs[1]
        ax.set_ylabel("Posterior $P(\\theta \\mid \\mathrm{data})$")
        ax.hist(data["posterior"], bins=bins, color="darkblue", rasterized=True, density=True)

        fig.tight_layout()
        fig.savefig(str(output))


rule plot_one_parameter_posterior_zoomed:
    input: "analysis/{analysis}/one_parameter_model/prior_posterior_samples.joblib"
    output: "analysis/{analysis}/one_parameter_model/posterior.pdf"
    run:
        posterior_samples = joblib.load(str(input))["posterior"]

        fig, ax = plt.subplots(dpi=DPI, figsize=FIGSIZE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("Posterior $P(\\theta \\mid \\mathrm{data})$")
        ax.hist(posterior_samples, bins=np.linspace(0.95 * np.quantile(posterior_samples, 0.02), 1.05 * np.quantile(posterior_samples, 0.98), 30), color="darkblue", rasterized=True, density=True)
        ax.set_xticklabels([f"{100*x:.1f}%" for x in  ax.get_xticks()])

        fig.tight_layout()
        fig.savefig(str(output))


rule one_parameter_sample_posterior_predictive:
    input: "analysis/{analysis}/one_parameter_model/prior_posterior_samples.joblib"
    output: "analysis/{analysis}/one_parameter_model/posterior_predictive.joblib"
    run:
        n_samples: int = 200
        
        data = joblib.load(str(input))
        posterior_samples, shape = data["posterior"], data["shape"]
        rng = np.random.default_rng(101)
        samples = np.asarray([rng.binomial(1, theta, size=shape) for theta in posterior_samples[:n_samples]])
        joblib.dump(samples, str(output))


# === Independent-probabilities model ===
# Each gene has a separate mutation probability theta[g]

rule independent_all:
    input:
        posterior_predictive_matrices_raw = "analysis/{analysis}/independent/posterior_predictive_matrices.pdf",
        posterior_predictive_matrices_ordered = "analysis/{analysis}/independent/posterior_predictive_matrices_ordered.pdf",
        posterior_predictive_occurrences = "analysis/{analysis}/independent/posterior_predictive_occurrence.pdf",
        posterior_predictive_histograms_many_panels = "analysis/{analysis}/independent/histogram_number_of_mutations_many_panels.pdf",
        posterior_predictive_histograms_single_panel = "analysis/{analysis}/independent/histogram_number_of_mutations_single_panel.pdf",
    output: touch("analysis/{analysis}/independent/done.done")


def independent_model(Y = None, N = None, G = None):
    if N is None or G is None:
        N, G = Y.shape
    
    # Priors for the probability vector theta
    alpha = np.ones(G)  # Adjust these based on your knowledge
    beta = np.ones(G)
    with numpyro.plate('features', G):
        theta = numpyro.sample('theta', dist.Beta(alpha, beta))
    
    with numpyro.plate('data', N, dim=-2):
        with numpyro.plate("features", G, dim=-1):
            numpyro.sample('obs', dist.Bernoulli(theta[None, :]), obs=Y)


rule mcmc_independent_model:
    input: "data/preprocessed/{analysis}/mutation-matrix.csv"
    output: "analysis/{analysis}/independent/posterior_samples.joblib"
    run:
        Y = pd.read_csv(str(input), index_col=0).values

        nuts_kernel = NUTS(independent_model)
        mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
        mcmc.run(jax.random.PRNGKey(0), Y=Y)

        samples = mcmc.get_samples()
        joblib.dump(samples, str(output))


rule posterior_predictive_independent_model:
    input:
        samples = "analysis/{analysis}/independent/posterior_samples.joblib",
        mutations = "data/preprocessed/{analysis}/mutation-matrix.csv"
    output: "analysis/{analysis}/independent/posterior_predictive.joblib"
    run:
        Y = pd.read_csv(input.mutations, index_col=0).values
        posterior_samples = joblib.load(input.samples)
        predictive = Predictive(independent_model, posterior_samples=posterior_samples)
        predictive_samples = predictive(jax.random.PRNGKey(12), N=Y.shape[0], G=Y.shape[1])["obs"]
        joblib.dump(predictive_samples, str(output))


# === Model-independent-rules ===

def optionally_order_matrix(data: np.ndarray, ordered: bool = True) -> np.ndarray:
    id0 = np.argsort(np.sum(data, axis=1))
    id1 = np.argsort(np.sum(data, axis=0))
    if ordered:
        return data[id0, :][:, id1]
    else:
        return data


def plot_posterior_predictive_matrices(
    axs,
    samples,
    mutations,
    model: str,
    ordered: bool,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(samples.shape[0],  size=len(axs.ravel()), replace=False)
    
    suffix = "" if not ordered else " (ordered)"

    for ax, index in zip(axs.ravel(), indices):
        data = samples[index, ...]

        sns.heatmap(optionally_order_matrix(data, ordered), ax=ax, cmap="Blues", xticklabels=False, yticklabels=False, cbar=False, square=False)
        ax.set_xlabel("Genes" + suffix)
        ax.set_ylabel("Patients" + suffix)

    x_true, y_true = MODELS[model].true_loc
    ax = axs[x_true, y_true]
    ax.clear()

    sns.heatmap(optionally_order_matrix(mutations, ordered), ax=ax, cmap="Blues", xticklabels=False, yticklabels=False, cbar=False, square=False)
    ax.set_xlabel("Genes" + suffix)
    ax.set_ylabel("Patients" + suffix)


rule plot_posterior_predictive_matrices:
    input:
        mutations = "data/preprocessed/{analysis}/mutation-matrix.csv",
        posterior_predictive = "analysis/{analysis}/{model}/posterior_predictive.joblib"
    output:
        matrices = "analysis/{analysis}/{model}/posterior_predictive_matrices.pdf",
    run:
        mutations = pd.read_csv(input.mutations, index_col=0).values
        samples = joblib.load(input.posterior_predictive)

        fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, dpi=250, figsize=(4*2, 3*2))
        plot_posterior_predictive_matrices(axs, samples, mutations=mutations, model=wildcards.model, ordered=False)
        fig.tight_layout()
        fig.savefig(output.matrices)


rule plot_posterior_predictive_matrices_ordered:
    input:
        mutations = "data/preprocessed/{analysis}/mutation-matrix.csv",
        posterior_predictive = "analysis/{analysis}/{model}/posterior_predictive.joblib"
    output:
        matrices = "analysis/{analysis}/{model}/posterior_predictive_matrices_ordered.pdf",
    run:
        mutations = pd.read_csv(input.mutations, index_col=0).values
        samples = joblib.load(input.posterior_predictive)

        fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, dpi=250, figsize=(4*2, 3*2))
        plot_posterior_predictive_matrices(axs, samples, mutations=mutations, model=wildcards.model, ordered=True)
        fig.tight_layout()
        fig.savefig(output.matrices)


rule plot_posterior_number_of_mutations_histograms_many_panels:
    input:
        mutations = "data/preprocessed/{analysis}/mutation-matrix.csv",
        posterior_predictive = "analysis/{analysis}/{model}/posterior_predictive.joblib"
    output: "analysis/{analysis}/{model}/histogram_number_of_mutations_many_panels.pdf"
    run:
        mutations = pd.read_csv(input.mutations, index_col=0).values
        samples = joblib.load(input.posterior_predictive)

        fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, dpi=250, figsize=(4*2, 3*2))

        seed = 42
        rng = np.random.default_rng(seed)
        indices = rng.choice(samples.shape[0],  size=len(axs.ravel()), replace=False)

        max_mut = max(samples.sum(axis=-1).max(), mutations.sum(axis=-1).max())
        bins = np.arange(-0.5, max_mut + 1.5)

        for ax, index in zip(axs.ravel(), indices):
            data = samples[index, ...]
            n_mutations = data.sum(axis=-1)
            ax.hist(n_mutations, bins=bins, rasterized=True, color="darkblue")
            
            ax.spines[["top", "right"]].set_visible(False)
            
        x_true, y_true = MODELS[wildcards.model].true_loc
        ax = axs[x_true, y_true]
        ax.clear()
        n_mutations = mutations.sum(axis=-1)
        ax.hist(n_mutations, bins=bins, rasterized=True, color="goldenrod")


        for ax in axs.ravel():
            ax.set_xlabel("Num. of mutations")
            ax.set_xticks(np.arange(0, max_mut + 1, 20))

        for ax in axs[:, 0]:
            ax.set_ylabel("Num. of patients")

        fig.tight_layout()
        fig.savefig(str(output))


rule plot_posterior_number_of_mutations_histograms_single_panel:
    input:
        mutations = "data/preprocessed/{analysis}/mutation-matrix.csv",
        posterior_predictive = "analysis/{analysis}/{model}/posterior_predictive.joblib"
    output: "analysis/{analysis}/{model}/histogram_number_of_mutations_single_panel.pdf"
    run:
        mutations = pd.read_csv(input.mutations, index_col=0).values
        samples = joblib.load(input.posterior_predictive)

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Number of mutations")
        ax.set_ylabel("Number of patients")

        seed = 42
        rng = np.random.default_rng(seed)
        indices = rng.choice(samples.shape[0],  size=min(samples.shape[0], 150), replace=False)

        max_mut = max(samples.sum(axis=-1).max(), mutations.sum(axis=-1).max())
        bins = np.arange(-0.5, max_mut + 1.5)

        for index in indices:
            data = samples[index, ...]
            n_mutations = data.sum(axis=-1)
            ax.hist(n_mutations, bins=bins, rasterized=True, color="darkblue", linewidth=0.2, alpha=0.15, histtype="step")
    
        n_mutations = mutations.sum(axis=-1)
        ax.hist(n_mutations, bins=bins, rasterized=True, color="goldenrod", linewidth=1, histtype="step")

        fig.tight_layout()
        fig.savefig(str(output))


rule plot_posterior_predictive_gene_occurrence:
    input:
        mutations = "data/preprocessed/{analysis}/mutation-matrix.csv",
        posterior_predictive = "analysis/{analysis}/{model}/posterior_predictive.joblib"
    output:
        occurrences = "analysis/{analysis}/{model}/posterior_predictive_occurrence.pdf",
    run:
        samples = joblib.load(input.posterior_predictive)

        fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, dpi=250, figsize=(4*2, 3*2))
        for ax, data in zip(axs.ravel(), samples[:len(axs.ravel()), ...]):
            ax.scatter(np.arange(data.shape[1]), np.sort(data.mean(axis=0)), s=2, c="darkblue")

        # Add the ground-truth values
        x_true, y_true = MODELS[wildcards.model].true_loc
        ax = axs[x_true, y_true]
        data = pd.read_csv(input.mutations, index_col=0).values
        ax.clear()

        ax.scatter(np.arange(data.shape[1]), np.sort(data.mean(axis=0)), s=2, c="darkblue")

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("Mutation frequency")
            ax.set_xlabel("Genes (ordered)")
            ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(output.occurrences)



# ==================================
# = Download data and preprocess ===
# ==================================

rule download_clinical:
  output: "data/raw/clinical-information.tsv"
  shell: "wget https://raw.githubusercontent.com/cbg-ethz/graphClust_NeurIPS/main/tcga_analysis/data/tcga-clinical-information.txt -O {output}"


rule download_mutation:
  output: "data/raw/mutation-matrix.tsv"
  shell: "wget https://raw.githubusercontent.com/cbg-ethz/graphClust_NeurIPS/main/tcga_analysis/data/binary-mutationCovariate-matrix.txt -O {output}"



rule preprocess_data:
  input:
    clinical = "data/raw/clinical-information.tsv",
    mutation = "data/raw/mutation-matrix.tsv"
  output:
    clinical = "data/preprocessed/{analysis}/clinical-information.csv",
    mutation = "data/preprocessed/{analysis}/mutation-matrix.csv"
  run:
    # Read mutation matrix and remove binarised covariates
    mutations = pd.read_csv(input.mutation, sep="\t", index_col=0)
    mutations = mutations.drop(
        ["Age", "Gender", "Stage"] + ["BLCA", "BRCA", "CESC", "COAD", "READ", "ESCA", "GBM", "HNSC", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "OV", "PAAD", "PCPG", "PRAD", "SARC", "STAD", "THCA", "UCEC"],
        axis="columns",
    )
    
    # Read clinical information and remove binarised (derived) information
    clinical = pd.read_csv(input.clinical, sep="\t", index_col=0)
    clinical = clinical.drop(["age.bin", "gender.bin", "stage.bin"], axis="columns")

    # Select cancers corresponding to selected tissue types
    allowed_tissue_types = ANALYSES[wildcards.analysis]
    clinical = clinical[clinical["type"].isin(allowed_tissue_types)]

    # Align data frames
    mutations, clinical = mutations.align(clinical, join="inner", axis=0)
    
    # Remove mutations which are constant for all patients
    mutations = mutations.loc[:, mutations.nunique() > 1]
    
    # Add standardized age
    clinical['std_age'] = (clinical['age'] - clinical['age'].mean()) / clinical['age'].std()

    mutations.to_csv(output.mutation, index=True)
    clinical.to_csv(output.clinical, index=True)
