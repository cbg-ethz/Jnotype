import numpy as np
import pandas as pd
import json
import jax
import jax.numpy as jnp

from contextlib import redirect_stdout

import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from subplots_from_axsize import subplots_from_axsize
import seaborn as sns

import numpyro
import numpyro.diagnostics as diagnostics
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


import jnotype.exclusivity._full as full
from jnotype.exclusivity import bmm
from jnotype.checks import calculate_quantiles, calculate_mcc

import jnotype.checks._statistics as st
from jnotype.checks import plot_summary_statistic, rcParams, subsample_pytree
from jnotype._utils import order_genotypes


workdir: "generated/exclusivity/gbm_reproducibility"


# GENESETS = {
#     "muex-0-3B": ['ABCC9', 'PIK3CA', 'RPL5', 'TRAT1'],
# }

rcParams["color_data"] = "black"

def read_archive(pth):
    npzfile = np.load(pth)
    return {key: npzfile[key] for key in npzfile.files}

def plot_dataset(ax, dataset, labels=True, change_color=False, gene_list = None):
    order = order_genotypes(dataset, reverse=True)
    if labels:
        assert gene_list is not None
        ticklabels = gene_list
        xlabel = "Genotypes"
    else:
        ticklabels = []
        xlabel = ""
    if change_color:
        cmap = "Blues"
    else:
        cmap = "Greys"
    sns.heatmap(dataset[order, :].T, ax=ax, vmin=0, vmax=1, cbar=False, cmap=cmap, xticklabels=[], yticklabels=ticklabels)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.set_xlabel(xlabel)


def pytree_at(samples, index):
    def index_leaves(x):
        """Function indexing each leaf"""
        return x[index]

    # Apply the indexing function to all leaves
    subsampled_pytree = jax.tree_util.tree_map(index_leaves, samples)

    return subsampled_pytree

def get_artificial_dataset_factory(N_SAMPLES: int):
    def get_artificial_dataset(key, sample):
        n_artificial = N_SAMPLES
        return bmm.sample_bernoulli_mixture(
            key=key,
            mixture_weights=sample["mixture_weights"],
            mixture_components=sample["components_noisy"],
            n_samples=n_artificial,
        )
    return get_artificial_dataset


def _mcc_stat_fn(y):
    indices = jnp.triu_indices(y.shape[1], k=1)
    return st.calculate_mcc(y)[indices]


SUMMARY_STATS = {
    "mutation_frequency": (st.calculate_mutation_frequencies, "Mutation frequency"),
    "num_mutations": (st.calculate_number_of_mutations_histogram, "Number of mutations"),
    "mcc": (_mcc_stat_fn, "MCC"),
    "atoms": (st.calculate_atoms_occurrence, "Atoms"),
}


GENESETS = {
    # "misspecified": ["TP53", "CDKN2B", "NF1", "SPTA1"],
    "muex-permutation-3A": ["EGFR", "GCSAML", "IDH1", "OTC"],
    "muex-0-3B": ['ABCC9', 'PIK3CA', 'RPL5', 'TRAT1'],
    "muex-1-3C": ['PIK3C2G', 'PIK3CA', 'RPL5', 'TRAT1'],
    "muex-2-3D": ['NF1', 'PIK3C2G', 'PIK3R1', 'TRAT1'],
    # "muex-3": ['ABCC9', 'PIK3C2G', 'PIK3CA', 'TRAT1'],
    # "muex-4": ['ABCC9', 'PIK3C2G', 'PIK3CA', 'SPTA1'],
    # "muex-5": ['ABCC9', 'KEL', 'PIK3C2G', 'PIK3CA'],
    # "muex-5": ['ABCC9', 'PIK3R1', 'RPL5', 'TRAT1'],
    # "muex-6": ['ABCC9', 'PIK3C2G', 'PIK3R1', 'TRAT1'],
    # "muex-7": ['PIK3C2G', 'PIK3R1', 'RPL5', 'TRAT1'],
    # "muex-8": ['ABCC9', 'PIK3C2G', 'PIK3CA', 'RPL5'],
    # "muex-9": ['ABCC9', 'KEL', 'PIK3C2G', 'RPL5'],
}

def get_prior_posterior_flag(name):
    if name == "prior":
        return False
    elif name == "posterior":
        return True
    else:
        raise ValueError(f"Name {name} not known.")

MODEL_KWARGS = {
    "restricted": dict(
        spike_prob=0.01,
        dirichlet_prior_weight=10_000,
        impurity_scale=0.1,
        independent_high=0.02,
        fnr_high=0.01,
        fpr_high=0.01,
    ),
    "restricted_high_noise": dict(
        spike_prob=0.01,
        dirichlet_prior_weight=10_000,
        impurity_scale=0.1,
        independent_high=0.02,
    ),
    "restricted_more_flexible": dict(
        spike_prob=0.01,
        dirichlet_prior_weight=2.0,
        impurity_scale=0.1,
        independent_high=0.02,
    ),
    "independent": dict(
        spike_prob=0.01,
        spike_width=0.01,
        slab_width=0.01,
        fnr_high=0.01,
        fpr_high=0.01,
        impurity_scale=0.01,
        dirichlet_prior_weight=10_000,
        independent_high=0.999,
    ),
    "full": dict(
        spike_prob=0.5,
        dirichlet_prior_weight=2.0,
        impurity_scale=0.1,
    ),
    "full_less_skeptical": dict(
        spike_prob=0.25,
        dirichlet_prior_weight=2.0,
        impurity_scale=0.1,
    ),
    "full_non_skeptical": dict(
        spike_prob=0.02,
        dirichlet_prior_weight=2.0,
        impurity_scale=0.1,
    ),
}

AX_HEIGHT = 0.6


rule all:
    input:
        mcmc_samples = expand("{scenario}/{model}/{prior_posterior}/samples.npz", scenario=GENESETS.keys(), model=MODEL_KWARGS.keys(), prior_posterior=["prior", "posterior"]),
        figure_restricted = expand("{scenario}/figure_restricted.pdf", scenario=GENESETS.keys()),
        figure_comparison = expand("{scenario}/figure_comparison.pdf", scenario=GENESETS.keys()),
        figure_artificial_data = expand("{scenario}/{model}/posterior_artificial_data.pdf", scenario=GENESETS.keys(), model=MODEL_KWARGS.keys()),
        figure_posterior_predictive = expand("{scenario}/{model}/posterior_predictive/{summary_statistic}.pdf", scenario=GENESETS.keys(), model=MODEL_KWARGS.keys(), summary_statistic=SUMMARY_STATS.keys()),
        figure_plot_data = expand("{scenario}/figure_data.pdf", scenario=GENESETS.keys()),
        figure_posterior_coverage = expand("{scenario}/{model}/posterior_figures/coverage.pdf", scenario=GENESETS.keys(), model=MODEL_KWARGS.keys()),
        figure_restricted_flexible = expand("{scenario}/figure_restricted_flexible.pdf", scenario=GENESETS.keys()),
        figure1 = "figure1/a.pdf",
        figure2 = "figure2.pdf",
        figure_full_pred = expand("{scenario}/figure_full_predictions.pdf", scenario=GENESETS.keys()),


rule figure1:
    input:
        a = "muex-permutation-3A/figure_data.pdf",
        b = "muex-0-3B/figure_data.pdf",
        c = "muex-1-3C/figure_data.pdf",
        d = "muex-2-3D/figure_data.pdf",
    output:
        a = "figure1/a.pdf",
        b = "figure1/b.pdf",
        c = "figure1/c.pdf",
        d = "figure1/d.pdf"
    shell:
        """
        cp {input.a} {output.a}
        cp {input.b} {output.b}
        cp {input.c} {output.c}
        cp {input.d} {output.d}
        """

rule figure2:
    input: "muex-permutation-3A/figure_restricted_flexible.pdf",
    output: "figure2.pdf"
    shell:
        """cp {input} {output}"""


rule figure_plot_data:
    input:
        data = "{scenario}/data.csv",
    output: "{scenario}/figure_data.pdf"
    run:
        df = pd.read_csv(input.data, index_col=0)
        gene_names = df.columns
        fig, ax = subplots_from_axsize(axsize=(3, 2), dpi=350)

        plot_dataset(
            ax=ax,
            dataset=df.values,
            labels=True,
            gene_list=df.columns,
        )
        fig.savefig(str(output))


rule plot_posterior_coverage:
    input:
        samples = "{scenario}/{model}/posterior/samples.npz"
    output: "{scenario}/{model}/posterior_figures/coverage.pdf"
    run:
        fig, ax = subplots_from_axsize(axsize=(3, 2), dpi=350)
        samples = read_archive(input.samples)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.hist(samples["coverage"], bins=30, rasterized=True, density=True)
        ax.set_yticks([])
        ax.set_xlabel("Coverage")
        fig.savefig(str(output))



rule plot_artificial_data:
    input:
        data = "{scenario}/data.csv",
        posterior_samples =  "{scenario}/{model}/posterior/samples.npz",
    output: "{scenario}/{model}/posterior_artificial_data.pdf"
    run:
        posterior_samples = read_archive(input.posterior_samples)
        df = pd.read_csv(input.data, index_col=0)
        gene_names = df.columns
        data = df.values
        N_SAMPLES = data.shape[0]

        get_artificial_dataset = get_artificial_dataset_factory(N_SAMPLES)

        fig, axs = plt.subplots(3, 4, figsize=(5, 2), dpi=500)

        i_cool, j_cool = 0, 0
        sk = jax.random.PRNGKey(1042)

        change_color = True

        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                sk, sk_ = jax.random.split(sk)
                ax = axs[i, j]
                if i == i_cool and j == j_cool:
                    plot_dataset(ax, data, labels=False, change_color=change_color)        
                else:
                    smple = pytree_at(posterior_samples, i * 300 + j * 30)
                    ds = get_artificial_dataset(sk_, smple)
                    plot_dataset(ax, ds, labels=False)        
    
        fig.savefig(str(output))


rule plot_model_full:
    input:
        data = "{scenario}/data.csv",
        posterior_full = "{scenario}/full/posterior/samples.npz",
        posterior_full_middle = "{scenario}/full_less_skeptical/posterior/samples.npz", 
        posterior_full_non_skeptical = "{scenario}/full_non_skeptical/posterior/samples.npz",
    output: "{scenario}/figure_full_predictions.pdf"
    run:
        def read_archive(pth):
            npzfile = np.load(pth)
            return {key: npzfile[key] for key in npzfile.files}

        posterior_full = read_archive(input.posterior_full)
        posterior_full_middle = read_archive(input.posterior_full_middle)
        posterior_nonsk = read_archive(input.posterior_full_non_skeptical)

        color_full = "darkblue"
        color_middle = "lightblue"
        color_nonsk = "grey"

        df = pd.read_csv(input.data, index_col=0)
        gene_names = df.columns

        h_size = 1.5
        fig, axs = subplots_from_axsize(axsize=([h_size] * 3, AX_HEIGHT), wspace=[0.3, 0.3], left=0.1, bottom=0.65)

        ax = axs[0]
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_xlabel("Coverage $\\gamma$")
        ax.set_yticks([])
        # ax.set_xlim(0, 0.4)
        ax.hist(posterior_full["coverage"], bins=20, alpha=0.5, color=color_full, density=True)
        ax.hist(posterior_full_middle["coverage"], bins=20, alpha=0.3, color=color_middle, density=True)
        ax.hist(posterior_nonsk["coverage"], bins=20, alpha=0.3, color=color_nonsk, density=True)

        # ax = axs[1]
        # ax.spines[["top", "right"]].set_visible(False)
        # ax.set_xlabel("Probabilities $\\omega$")

        # stats = posterior["exclusive_weights"]
        # color = "blue"
        # quantiles = calculate_quantiles(samples=stats, quantiles=np.array([0.01, 0.25, 0.5, 0.75, 0.99]))
        # x_axis = jnp.arange(1, stats.shape[1] + 1)
        
        # ax.set_ylim(0, quantiles.max() + 0.05)
        # ax.set_xticks(x_axis, gene_names, rotation=30, fontsize="x-small")
        # ax.set_xlim(1-0.2, jnp.max(x_axis) + 0.2)
        
        # ax.plot(x_axis, quantiles[2, :], c=color, marker=".")
        # ax.fill_between(x_axis, quantiles[0, :], quantiles[-1, :], color=color, alpha=0.1, edgecolor=None)
        # ax.fill_between(x_axis, quantiles[1, :], quantiles[-2, :], color=color, alpha=0.2, edgecolor=None)

        # Comparison between logodds ratio
        def get_lambda(locus1, locus2, samples):
            def f(sample):
                return bmm.logodds_ratio(
                    locus1=locus1,
                    locus2=locus2,
                    mixture_weights=jnp.clip(sample["mixture_weights"], 1e-7, 1-1e-7),
                    mixture_components=jnp.clip(sample["components_noiseless"], 1e-7, 1-1e-7),
                )
            return jax.vmap(f)(samples)

        # Comparison between conditional probability differences
        def get_delta(locus1, locus2, samples):
            def f(sample):
                return bmm.conditional_probability_difference(
                    response=locus1,
                    conditioning=locus2,
                    mixture_weights=jnp.clip(sample["mixture_weights"], 1e-7, 1-1e-7),
                    mixture_components=jnp.clip(sample["components_noiseless"], 1e-7, 1-1e-7),
                )
            return jax.vmap(f)(samples)

        # sple = get_lambda(0, 2, posterior)

        ax = axs[1]
        loc1, loc2 = 0, 2
        ax.set_xlabel(f"$\\lambda$({gene_names[loc1]}, {gene_names[loc2]})", fontsize="small")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        ax.hist(get_lambda(loc1, loc2, posterior_full), color=color_full, bins=20, alpha=0.7, rasterized=True)
        ax.hist(get_lambda(loc1, loc2, posterior_full_middle), color=color_middle, bins=20, alpha=0.7, rasterized=True)
        ax.hist(get_lambda(loc1, loc2, posterior_nonsk), color=color_nonsk, bins=20, alpha=0.7, rasterized=True)
        # ax.hist(get_lambda(loc1, loc2, restricted_posterior), color="grey", bins=20, alpha=0.7)

        ax = axs[2]
        loc1, loc2 = 2, 0
        ax.set_xlabel(f"$\\Delta$({gene_names[loc1]} | {gene_names[loc2]})", fontsize="small")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        ax.hist(get_delta(loc1, loc2, posterior_full), color=color_full, bins=20, alpha=0.7, rasterized=True)
        ax.hist(get_delta(loc1, loc2, posterior_full_middle), color=color_middle, bins=20, alpha=0.7, rasterized=True)
        ax.hist(get_delta(loc1, loc2, posterior_nonsk), color=color_nonsk, bins=20, alpha=0.7, rasterized=True)


        fig.savefig(str(output))



rule plot_restricted_flexible:
    input:
        data = "{scenario}/data.csv",
        posterior_samples = "{scenario}/restricted_more_flexible/posterior/samples.npz",
    output: "{scenario}/figure_restricted_flexible.pdf"
    run:
        def read_archive(pth):
            npzfile = np.load(pth)
            return {key: npzfile[key] for key in npzfile.files}

        posterior = read_archive(input.posterior_samples)

        df = pd.read_csv(input.data, index_col=0)
        gene_names = df.columns

        h_size = 1.5
        fig, axs = subplots_from_axsize(axsize=([h_size] * 4, AX_HEIGHT), wspace=[0.4, 0.3, 0.3], left=0.1, bottom=0.65)

        ax = axs[0]
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_xlabel("Coverage $\\gamma$")
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.hist(posterior["coverage"], bins=20, alpha=0.5, color="darkblue", density=True)

        ax = axs[1]
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Probabilities $\\omega$")

        stats = posterior["exclusive_weights"]
        color = "blue"
        quantiles = calculate_quantiles(samples=stats, quantiles=np.array([0.01, 0.25, 0.5, 0.75, 0.99]))
        x_axis = jnp.arange(1, stats.shape[1] + 1)
        
        ax.set_ylim(0, quantiles.max() + 0.05)
        ax.set_xticks(x_axis, gene_names, rotation=30, fontsize="x-small")
        ax.set_xlim(1-0.2, jnp.max(x_axis) + 0.2)
        
        ax.plot(x_axis, quantiles[2, :], c=color, marker=".")
        ax.fill_between(x_axis, quantiles[0, :], quantiles[-1, :], color=color, alpha=0.1, edgecolor=None)
        ax.fill_between(x_axis, quantiles[1, :], quantiles[-2, :], color=color, alpha=0.2, edgecolor=None)

        # Comparison between logodds ratio
        def get_lambda(locus1, locus2, samples):
            def f(sample):
                return bmm.logodds_ratio(
                    locus1=locus1,
                    locus2=locus2,
                    mixture_weights=jnp.clip(sample["mixture_weights"], 1e-7, 1-1e-7),
                    mixture_components=jnp.clip(sample["components_noiseless"], 1e-7, 1-1e-7),
                )
            return jax.vmap(f)(samples)

        # Comparison between conditional probability differences
        def get_delta(locus1, locus2, samples):
            def f(sample):
                return bmm.conditional_probability_difference(
                    response=locus1,
                    conditioning=locus2,
                    mixture_weights=jnp.clip(sample["mixture_weights"], 1e-7, 1-1e-7),
                    mixture_components=jnp.clip(sample["components_noiseless"], 1e-7, 1-1e-7),
                )
            return jax.vmap(f)(samples)

        # sple = get_lambda(0, 2, posterior)

        ax = axs[2]
        loc1, loc2 = 0, 2
        ax.set_xlabel(f"$\\lambda$({gene_names[loc1]}, {gene_names[loc2]})", fontsize="small")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        ax.hist(get_lambda(loc1, loc2, posterior), color="gold", bins=20, alpha=0.7, rasterized=True)
        # ax.hist(get_lambda(loc1, loc2, restricted_posterior), color="grey", bins=20, alpha=0.7)

        ax = axs[3]
        loc1, loc2 = 2, 0
        ax.set_xlabel(f"$\\Delta$({gene_names[loc1]} | {gene_names[loc2]})", fontsize="small")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        ax.hist(get_delta(loc1, loc2, posterior), color="gold", bins=20, alpha=0.7)

        fig.savefig(str(output))




rule plot_posterior_predictive_check:
    input:
        data = "{scenario}/data.csv",
        posterior_samples =  "{scenario}/{model}/posterior/samples.npz",
    output:
        raw = "{scenario}/{model}/posterior_predictive/{summary_statistic}.pdf",
        residuals = "{scenario}/{model}/posterior_predictive/{summary_statistic}-residuals.pdf",
    run:
        key = jax.random.PRNGKey(987 + hash(str(wildcards.summary_statistic)) // 1000 + hash(str(wildcards.scenario)) // 1000 + hash(str(wildcards.model)) // 1000 )
        
        key, subkey = jax.random.split(key)
        posterior_samples = read_archive(input.posterior_samples)
        posterior_samples_subsampled = st.subsample_pytree(subkey, posterior_samples, 2000)

        df = pd.read_csv(input.data, index_col=0)
        gene_names = df.columns
        data = df.values
        N_SAMPLES = data.shape[0]

        get_artificial_dataset = get_artificial_dataset_factory(N_SAMPLES)


        def plot(
            stat_fn,
            stat_title,
            residuals: bool = False,
        ):
            full_stat = st.simulate_summary_statistic(
                key,
                simulator_fn=get_artificial_dataset,
                statistic_fn=stat_fn,
                samples=posterior_samples,
            )
            data_stat = stat_fn(data)
            fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
            suffix = " (residuals)" if residuals else ""
            fig.suptitle(stat_title + suffix)
            plot_summary_statistic(
                ax,
                y_data=data_stat,
                y_simulated=full_stat,
                summary_type="median",
                residuals=residuals,
            )
            return fig

        stat_fn, stat_title = SUMMARY_STATS[wildcards.summary_statistic]

        fig_raw = plot(stat_fn, stat_title, residuals=False)
        fig_raw.savefig(output.raw)
        fig_residuals = plot(stat_fn, stat_title, residuals=True)
        fig_residuals.savefig(output.residuals)



rule plot_figure_comparison:
    input:
        data = "{scenario}/data.csv",
        posterior_restricted_samples = "{scenario}/restricted/posterior/samples.npz",
        prior_full_samples = "{scenario}/full/prior/samples.npz",
        posterior_full_samples = "{scenario}/full/posterior/samples.npz",
    output: "{scenario}/figure_comparison.pdf"
    run:
        df = pd.read_csv(input.data, index_col=0)
        gene_names = df.columns

        def read_archive(pth):
            npzfile = np.load(pth)
            return {key: npzfile[key] for key in npzfile.files}

        prior = read_archive(input.prior_full_samples)
        posterior = read_archive(input.posterior_full_samples)
        restricted_posterior = read_archive(input.posterior_restricted_samples)

        fig, axs = subplots_from_axsize(axsize=([1, 1, 1, 1], AX_HEIGHT), wspace=[0.4, 0.3, 0.3], left=0.1, bottom=0.65)

        ax = axs[0]
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_xlabel("Coverage $\\gamma$")
        ax.set_yticks([])

        ax.hist(prior["coverage"], bins=20, alpha=0.5, color="salmon", density=True)
        ax.hist(posterior["coverage"], bins=20, alpha=0.5, color="darkblue", density=True)

        ax = axs[1]
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Probabilities $\\pi$")

        stats = posterior["_component_independent"]
        color = "blue"
        quantiles = calculate_quantiles(samples=stats, quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95]))
        x_axis = jnp.arange(1, stats.shape[1] + 1)
        
        ax.set_ylim(0, quantiles.max() + 0.05)
        ax.set_xticks(x_axis, gene_names, rotation=30, fontsize="x-small")
        ax.set_xlim(1-0.2, jnp.max(x_axis) + 0.2)
        
        ax.plot(x_axis, quantiles[2, :], c=color, marker=".")
        ax.fill_between(x_axis, quantiles[0, :], quantiles[-1, :], color=color, alpha=0.1, edgecolor=None)
        ax.fill_between(x_axis, quantiles[1, :], quantiles[-2, :], color=color, alpha=0.2, edgecolor=None)

        # Comparison between logodds ratio
        def get_lambda(locus1, locus2, samples):
            def f(sample):
                return bmm.logodds_ratio(
                    locus1=locus1,
                    locus2=locus2,
                    mixture_weights=jnp.clip(sample["mixture_weights"], 1e-7, 1-1e-7),
                    mixture_components=jnp.clip(sample["components_noiseless"], 1e-7, 1-1e-7),
                )
            return jax.vmap(f)(samples)

        # Comparison between conditional probability differences
        def get_delta(locus1, locus2, samples):
            def f(sample):
                return bmm.conditional_probability_difference(
                    response=locus1,
                    conditioning=locus2,
                    mixture_weights=jnp.clip(sample["mixture_weights"], 1e-7, 1-1e-7),
                    mixture_components=jnp.clip(sample["components_noiseless"], 1e-7, 1-1e-7),
                )
            return jax.vmap(f)(samples)

        # sple = get_lambda(0, 2, posterior)

        ax = axs[2]
        loc1, loc2 = 0, 2
        ax.set_xlabel(f"$\\lambda$({gene_names[loc1]}, {gene_names[loc2]})", fontsize="small")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        ax.hist(get_lambda(loc1, loc2, posterior), color="gold", bins=20, alpha=0.7)
        ax.hist(get_lambda(loc1, loc2, restricted_posterior), color="grey", bins=20, alpha=0.7)

        ax = axs[3]
        loc1, loc2 = 1, 3
        ax.set_xlabel(f"$\\lambda$({gene_names[loc1]}, {gene_names[loc2]})", fontsize="small")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        ax.hist(get_lambda(loc1, loc2, posterior), color="gold", bins=20, alpha=0.7)
        ax.hist(get_lambda(loc1, loc2, restricted_posterior), color="grey", bins=20, alpha=0.7)

        fig.savefig(str(output))


rule plot_figure_restricted:
    input:
        data = "{scenario}/data.csv",
        prior_samples = "{scenario}/restricted/prior/samples.npz",
        posterior_samples = "{scenario}/restricted/posterior/samples.npz",
    output: "{scenario}/figure_restricted.pdf"
    run:
        df = pd.read_csv(input.data, index_col=0)
        data = df.values
        gene_names = df.columns

        prior = np.load(input.prior_samples)
        posterior = np.load(input.posterior_samples)

        fig, axs = subplots_from_axsize(axsize=([1, 1, 2], AX_HEIGHT), wspace=[0.2, 0.7], left=0.1)


        ax = axs[0]
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_xlabel("Coverage $\\gamma$")
        ax.set_yticks([])

        ax.hist(prior["coverage"], bins=20, alpha=0.5, color="salmon", density=True)
        ax.hist(posterior["coverage"], bins=20, alpha=0.5, color="darkblue", density=True)


        ax = axs[1]
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_xlabel("Impurity $\\xi$")
        ax.set_yticks([])

        ax.hist(prior["impurity"], bins=20, alpha=0.5, color="salmon", density=True)
        ax.hist(posterior["impurity"], bins=20, alpha=0.5, color="darkblue", density=True)

        ax = axs[2]
        order = order_genotypes(data, reverse=True)
        sns.heatmap(data[order, :].T, ax=ax, vmin=0, vmax=1, cbar=False, cmap="Greys", xticklabels=[], yticklabels=gene_names)
        ax.spines[["bottom", "left"]].set_visible(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Observed data")

        fig.savefig(str(output))

rule prepare_data:
    output: "{scenario}/data.csv"
    run:
        df = pd.read_csv("../data/gbm-muex-2014.csv", index_col=0)
        genes = GENESETS[wildcards.scenario]
        df[genes].to_csv(str(output), index=True)


rule fit_model:
    input: "{scenario}/data.csv"
    output:
        samples = "{scenario}/{model}/{prior_posterior}/samples.npz",
        summary = "{scenario}/{model}/{prior_posterior}/summary.csv",
        summary_readable = "{scenario}/{model}/{prior_posterior}/summary.txt",
    run:
        df = pd.read_csv(str(input), index_col=0)
        data = df.values
        
        model = full.extended_model
        nuts_kernel = NUTS(model, max_tree_depth=15, target_accept_prob=0.99)

        mcmc_kwargs = MODEL_KWARGS[wildcards.model]

        # Run the MCMC sampler
        mcmc = MCMC(nuts_kernel, num_warmup=8_000, num_samples=8_000, num_chains=4)
        rng_key = jax.random.PRNGKey(2014)
        mcmc.run(
            rng_key,
            data=data,
            posterior=get_prior_posterior_flag(wildcards.prior_posterior),
            use_preprocessing=True,
            **mcmc_kwargs,
        )

        # Get the samples
        samples = mcmc.get_samples()
        np.savez(output.samples, **samples)

        summary_dict = diagnostics.summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
        pd.DataFrame(summary_dict).to_csv(output.summary, index=True)

        with open(output.summary_readable, "w") as fh:
            with redirect_stdout(fh):
                mcmc.print_summary(exclude_deterministic=True)

