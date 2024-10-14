import numpy as np
import pandas as pd
import json
import jax
import jax.numpy as jnp

from contextlib import redirect_stdout

import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("Agg")
from subplots_from_axsize import subplots_from_axsize
import seaborn as sns

import numpyro
import numpyro.diagnostics as diagnostics
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


import jnotype.exclusivity._full as full
from jnotype.exclusivity import bmm
from jnotype._utils import order_genotypes
from jnotype.checks import calculate_quantiles, calculate_mcc


workdir: "generated/exclusivity/gbm_reproducibility"


GENESETS = {
    "muex-0-3B": ['ABCC9', 'PIK3CA', 'RPL5', 'TRAT1'],
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
        spike_prob=0.5,
        dirichlet_prior_weight=1_000,
        impurity_scale=0.1,
        independent_high=0.02,
        fnr_high=0.01,
        fpr_high=0.01,
    ),
    "full": dict(
        spike_prob=0.5,
        dirichlet_prior_weight=1_000,
        impurity_scale=0.1,
    ),
}

AX_HEIGHT = 0.6


rule all:
    input:
        mcmc_samples = expand("{scenario}/{model}/{prior_posterior}/samples.npz", scenario=GENESETS.keys(), model=MODEL_KWARGS.keys(), prior_posterior=["prior", "posterior"]),
        figure_restricted = expand("{scenario}/figure_restricted.pdf", scenario=GENESETS.keys()),
        figure_comparison = expand("{scenario}/figure_comparison.pdf", scenario=GENESETS.keys()),


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
        quantiles = calculate_quantiles(samples=stats, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
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

        sple = get_lambda(0, 1, posterior)

        ax = axs[2]
        ax.set_xlabel(f"$\\lambda$({gene_names[0]}, {gene_names[1]})", fontsize="small")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        ax.hist(get_lambda(0, 1, posterior), color="gold", bins=20, alpha=0.7)
        ax.hist(get_lambda(0, 1, restricted_posterior), color="grey", bins=20, alpha=0.7)

        ax = axs[3]
        ax.set_xlabel(f"$\\lambda$({gene_names[2]}, {gene_names[3]})", fontsize="small")
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        ax.hist(get_lambda(2, 3, posterior), color="gold", bins=20, alpha=0.7)
        ax.hist(get_lambda(2, 3, restricted_posterior), color="grey", bins=20, alpha=0.7)

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
        nuts_kernel = NUTS(model, max_tree_depth=15, target_accept_prob=0.97)

        mcmc_kwargs = MODEL_KWARGS[wildcards.model]

        # Run the MCMC sampler
        mcmc = MCMC(nuts_kernel, num_warmup=5000, num_samples=5000, num_chains=4)
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

