
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("Agg")
import seaborn as sns
from subplots_from_axsize import subplots_from_axsize

import json
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

import numpyro
import numpyro.diagnostics as diagnostics
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import jnotype.exclusivity._full as full
from jnotype.exclusivity import bmm
from jnotype._utils import order_genotypes
from jnotype.checks import calculate_quantiles, calculate_mcc

workdir: "generated/exclusivity/cooccurrence"

GROUND_TRUTH_NOISELESS = {
    "weights": jnp.asarray([0.3, 0.7]),
    "components": jnp.asarray([
        [0.99, 0.99, 0.3, 0.3, 0.3],
        [0.01, 0.01, 0.3, 0.3, 0.3],
    ]),
}
GROUND_TRUTH_NOISY = {
    "weights": GROUND_TRUTH_NOISELESS["weights"],
    "components": bmm.adjust_mixture_components_for_noise(
        GROUND_TRUTH_NOISELESS["components"],
        false_negative_rate=0.05,
        false_positive_rate=0.05,
    ),
}
N_SAMPLES: int = 200
AXES_HEIGHT: float = 1.1

rule all:
    input:
        genotypes_plot = "genotypes.pdf", 
        samples = expand("full_model/{model}/samples.npz", model=["prior", "posterior"]),
        figure_summary = "figure_summary.pdf",


rule plot_figure:
    input:
        prior_samples = "full_model/prior/samples.npz",
        posterior_samples = "full_model/posterior/samples.npz",
        data = "data.npy",
    output:
        figure = "figure_summary.pdf"
    run:
        def read_archive(pth):
            npzfile = np.load(pth)
            return {key: npzfile[key] for key in npzfile.files}

        prior_samples = read_archive(input.prior_samples)
        posterior_samples = read_archive(input.posterior_samples)
        data = np.load(input.data)

        fig, axs = subplots_from_axsize(axsize=([1, 1.5, 1.5, 1, 1], AXES_HEIGHT), left=0.1, wspace=0.4)

        ax = axs[0]
        # ax.hist(prior_samples["coverage"], bins=30, density=True, color="salmon")
        ax.hist(posterior_samples["coverage"], bins=30, density=True, color="darkblue")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel("Coverage $\\gamma$")


        ax = axs[1]
        ax.set_xlabel("Probabilities $\\pi$")
        ax.spines[["top", "right"]].set_visible(False)

        stats = posterior_samples["_component_independent"]
        color = "blue"
        quantiles = calculate_quantiles(samples=stats, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        x_axis = jnp.arange(1, stats.shape[1] + 1)
        
        ax.set_ylim(0, 1)
        ax.set_xticks(x_axis, x_axis)
        
        ax.plot(x_axis, quantiles[2, :], c=color, marker=".")
        ax.fill_between(x_axis, quantiles[0, :], quantiles[-1, :], color=color, alpha=0.1, edgecolor=None)
        ax.fill_between(x_axis, quantiles[1, :], quantiles[-2, :], color=color, alpha=0.2, edgecolor=None)

        ax = axs[2]
        ax.set_ylim(0, 1)
        ax.set_xlabel("Weights $\\omega$")
        ax.spines[["top", "right"]].set_visible(False)

        stats = posterior_samples["exclusive_weights"]
        color = "green"
        quantiles = calculate_quantiles(samples=stats, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        x_axis = jnp.arange(1, stats.shape[1] + 1, dtype=int)
        
        ax.set_xticks(x_axis, x_axis)
        
        ax.plot(x_axis, quantiles[2, :], c=color, marker=".")
        ax.fill_between(x_axis, quantiles[0, :], quantiles[-1, :], color=color, alpha=0.1, edgecolor=None)
        ax.fill_between(x_axis, quantiles[1, :], quantiles[-2, :], color=color, alpha=0.2, edgecolor=None)



        def get_artificial_dataset(key, sample):
            n_artificial = N_SAMPLES
            return bmm.sample_bernoulli_mixture(
                key=key,
                mixture_weights=sample["mixture_weights"],
                mixture_components=sample["components_noisy"],
                n_samples=n_artificial,
            )

        def fun_mcc(key, sample):
            return calculate_mcc(get_artificial_dataset(key, sample))

        key = jax.random.PRNGKey(101)
        n_mcmc_samples = posterior_samples["components_noiseless"].shape[0]
        mccs = jax.vmap(fun_mcc)(jax.random.split(key, n_mcmc_samples), posterior_samples)

        ax = axs[3]
        ax.set_xlabel("MCC$(D_1, D_2)$")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks([])
        ax.hist(mccs[:, 0, 1], bins=20, density=True, color="maroon", alpha=0.3)
        ax.axvline(calculate_mcc(data)[0, 1], linestyle=":", color="black")
        ax.set_xlim(0, 1)

        ax = axs[4]
        ax.set_xlabel("MCC$(D_4, D_5)$")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks([])
        ax.hist(mccs[:, 3, 4], bins=20, density=True, color="maroon", alpha=0.3)
        ax.axvline(calculate_mcc(data)[3, 4], linestyle=":", color="black")
        ax.set_xlim(-0.5, 0.5)

        fig.savefig(output.figure)


rule fit_mcmc_posterior:
    input: "data.npy"
    output:
        samples = "full_model/posterior/samples.npz",
        summary = "full_model/posterior/summary.csv",
        summary_readable = "full_model/posterior/summary.txt"
    run:
        data = np.load(str(input))
        
        model = full.extended_model
        nuts_kernel = NUTS(model, max_tree_depth=15, target_accept_prob=0.97)

        # Run the MCMC sampler
        mcmc = MCMC(nuts_kernel, num_warmup=5000, num_samples=5000, num_chains=4)
        rng_key = jax.random.PRNGKey(0)
        mcmc.run(rng_key, data=data, use_preprocessing=True, posterior=True)

        # Get the samples
        samples = mcmc.get_samples()
        np.savez(output.samples, **samples)

        summary_dict = diagnostics.summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
        pd.DataFrame(summary_dict).to_csv(output.summary, index=True)

        with open(output.summary_readable, "w") as fh:
            with redirect_stdout(fh):
                mcmc.print_summary(exclude_deterministic=True)


rule fit_mcmc_prior:
    input: "data.npy"
    output:
        samples = "full_model/prior/samples.npz",
        summary = "full_model/prior/summary.csv",
        summary_readable = "full_model/prior/summary.txt"
    run:
        data = np.load(str(input))
        
        model = full.extended_model
        nuts_kernel = NUTS(model, max_tree_depth=15, target_accept_prob=0.97)

        # Run the MCMC sampler
        mcmc = MCMC(nuts_kernel, num_warmup=5000, num_samples=5000, num_chains=4)
        rng_key = jax.random.PRNGKey(0)
        mcmc.run(rng_key, data=data, posterior=False)

        # Get the samples
        samples = mcmc.get_samples()
        np.savez(output.samples, **samples)

        summary_dict = diagnostics.summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
        pd.DataFrame(summary_dict).to_csv(output.summary, index=True)

        with open(output.summary_readable, "w") as fh:
            with redirect_stdout(fh):
                mcmc.print_summary(exclude_deterministic=True)


rule plot_data:
    input: "data.npy"
    output: "genotypes.pdf"
    run:
        data = np.load(str(input))
        n_genes = data.shape[1]
        order = order_genotypes(data, reverse=True)
        fig, ax = subplots_from_axsize(axsize=(2.0, AXES_HEIGHT))
        sns.heatmap(data[order, :].T, ax=ax, vmin=0, vmax=1, cbar=False, cmap="Greys", xticklabels=[], yticklabels=[f"$G_{g}$" for g in range(1, n_genes + 1)])
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Simulated genotypes")
        fig.savefig(str(output))


rule sample_data:
    output: "data.npy"
    run:
        data = bmm.sample_bernoulli_mixture(
            key=jax.random.PRNGKey(2024),
            n_samples=N_SAMPLES,
            mixture_weights=GROUND_TRUTH_NOISY["weights"],
            mixture_components=GROUND_TRUTH_NOISY["components"],
        )

        data = np.asarray(data)
        np.save(str(output), data, allow_pickle=False)
