import numpy as np
import pandas as pd
import json
import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("Agg")
from subplots_from_axsize import subplots_from_axsize
import seaborn as sns

import numpyro
import numpyro.diagnostics as diagnostics
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from jnotype.checks import calculate_quantiles
from jnotype.exclusivity import muex
from jnotype._utils import order_genotypes
import jnotype.exclusivity._bernoulli_mixtures as bmm


workdir: "generated/exclusivity/gbm"

GENESETS = {
    "misspecified": ["TP53", "CDKN2B", "NF1", "SPTA1"],
    "muex-permutation-3A": ["EGFR", "GCSAML", "IDH1", "OTC"],
    "muex-0-3B": ['ABCC9', 'PIK3CA', 'RPL5', 'TRAT1'],
    "muex-1-3C": ['PIK3C2G', 'PIK3CA', 'RPL5', 'TRAT1'],
    "muex-2-3D": ['NF1', 'PIK3C2G', 'PIK3R1', 'TRAT1'],
    "muex-3": ['ABCC9', 'PIK3C2G', 'PIK3CA', 'TRAT1'],
    "muex-4": ['ABCC9', 'PIK3C2G', 'PIK3CA', 'SPTA1'],
    "muex-5": ['ABCC9', 'KEL', 'PIK3C2G', 'PIK3CA'],
    "muex-5": ['ABCC9', 'PIK3R1', 'RPL5', 'TRAT1'],
    "muex-6": ['ABCC9', 'PIK3C2G', 'PIK3R1', 'TRAT1'],
    "muex-7": ['PIK3C2G', 'PIK3R1', 'RPL5', 'TRAT1'],
    "muex-8": ['ABCC9', 'PIK3C2G', 'PIK3CA', 'RPL5'],
    "muex-9": ['ABCC9', 'KEL', 'PIK3C2G', 'RPL5'],
}

N_SIMULATED: int = 500


def calculate_freq(X):
    return jnp.mean(X, axis=0)

def calculate_bincount(X):
    n_genes = X.shape[1]
    counts = jnp.sum(X, axis=1)
    return jnp.bincount(counts, length=n_genes + 1)

def calculate_mcc(X):
    return jnp.corrcoef(X, rowvar=False)


rule all:
    input: 
        comparisons = expand("{scenario}/summary_statistic_comparison.pdf", scenario=GENESETS.keys()),
        genotype_plots = expand("{scenario}/genotypes.pdf", scenario=GENESETS.keys()),
        posterior_plot = expand("{scenario}/muex_bayes/posterior.pdf", scenario=GENESETS.keys()),

rule prepare_data:
    output: "{scenario}/data.csv"
    run:
        df = pd.read_csv("../data/gbm-muex-2014.csv", index_col=0)
        genes = GENESETS[wildcards.scenario]
        df[genes].to_csv(str(output), index=True)

rule plot_data:
    input: "{scenario}/data.csv" 
    output: "{scenario}/genotypes.pdf"
    run:
        df = pd.read_csv(str(input), index_col=0)
        gene_names = df.columns
        data = df.values

        fig, ax = subplots_from_axsize(axsize=(4, 1), left=0.8, bottom=0.5)
        order = order_genotypes(data, reverse=True)
        sns.heatmap(data[order, :].T, ax=ax, vmin=0, vmax=1, cbar=False, cmap="Greys", xticklabels=[], yticklabels=gene_names)
        ax.spines[["left", "bottom"]].set_visible(True)
        ax.set_xlabel("Samples")
        fig.savefig(str(output))


rule mle_independent:
    input: "{scenario}/data.csv"
    output:
        params = "{scenario}/independence_mle/params.json",
        artificial = "{scenario}/independence_mle/artificial.npy",
    run:
        df = pd.read_csv(str(input), index_col=0)
        gene_names = df.columns
        matrix = df.values

        n_patients, n_genes = matrix.shape

        pi_ = matrix.mean(axis=0)  # Maximum likelihood estimate
        
        # Save the maximum likelihood estimate
        params = {gene_name: float(freq) for gene_name, freq in zip(df.columns, pi_)}
        with open(output.params, "w") as fp:
            json.dump(params, fp=fp)

        # Now generate simulated data sets
        rng = np.random.default_rng(42)
        X = rng.binomial(1, p=pi_, size=(N_SIMULATED, n_patients, n_genes))
        np.save(file=output.artificial, arr=X, allow_pickle=False)


rule visualise_bayes_muex:
    input:
        prior = "{scenario}/muex_bayes/prior.npz",
        posterior = "{scenario}/muex_bayes/posterior.npz",
    output:
        prior_visualisation = "{scenario}/muex_bayes/prior.pdf",
        posterior_visualisation = "{scenario}/muex_bayes/posterior.pdf",
        comparison_visualisation = "{scenario}/muex_bayes/comparison_prior_and_posterior.pdf",
    run:
        prior_samples = np.load(input.prior)
        posterior_samples = np.load(input.posterior)
        
        vars = [
            ("alpha", "FPR $\\alpha$"),
            ("beta", "FNR $\\beta$"),
            ("gamma", "Coverage $\\gamma$"),
            ("delta", "Impurity $\\xi$"),
        ]

        def get_fig_and_axes():
            return subplots_from_axsize(axsize=([1.2, 1.2, 1.2, 1.2], 1), hspace=0.02, bottom=0.5)
        
        def visualise_samples(samples_dict, color, axs=None):
            # Plot the prior and the posterior
            if axs is None:
                fig, axs = get_fig_and_axes()
            else:
                fig = None

            for ax, (var_name, label) in zip(axs, vars):
                samples = samples_dict[var_name]
                ax.hist(samples, bins=30, density=True, alpha=0.7, color=color, edgecolor=None)
                ax.spines[["top", "left", "right"]].set_visible(False)
                ax.set_yticks([])
                ax.set_xlabel(label)
            return fig

        visualise_samples(prior_samples, "salmon").savefig(output.prior_visualisation)
        visualise_samples(posterior_samples, "skyblue").savefig(output.posterior_visualisation)

        fig, axs = get_fig_and_axes()
        visualise_samples(prior_samples, "salmon", axs=axs)
        visualise_samples(posterior_samples, "skyblue", axs=axs)
        fig.savefig(output.comparison_visualisation)


rule bayes_muex:
    input: "{scenario}/data.csv"
    output:
        prior = "{scenario}/muex_bayes/prior.npz",
        posterior = "{scenario}/muex_bayes/posterior.npz",
        artificial = "{scenario}/muex_bayes/artificial.npy",
        prior_summary = "{scenario}/muex_bayes/prior_summary.csv",
        posterior_summary = "{scenario}/muex_bayes/posterior_summary.csv",
    run:
        eps = 1e-3
        def model(Y, posterior: bool):
            alpha = numpyro.sample("alpha", dist.TruncatedNormal(loc=0, scale=0.1, low=eps, high=0.3))
            beta = numpyro.sample("beta", dist.TruncatedNormal(loc=0, scale=0.1, low=eps, high=0.3))
            gamma = numpyro.sample("gamma", dist.Uniform(low=eps, high=1-eps))
            delta = numpyro.sample("delta", dist.TruncatedNormal(loc=0, scale=0.2, low=eps, high=1-eps))
            
            if posterior:
                ll_fn = muex.get_loglikelihood_function(Y, from_params=False)
                numpyro.factor("loglikelihood", ll_fn(alpha, beta, gamma, delta))

        df = pd.read_csv(str(input), index_col=0)
        gene_names = df.columns
        matrix = df.values

        key = jax.random.PRNGKey(42)
        key, subkey_prior, subkey_posterior = jax.random.split(key, 3)

        # Sample from the prior
        
        _mcmc_chains = 4
        _mcmc_warmup = 1500
        _mcmc_samples = 1000

        prior_kernel = NUTS(model)  #, step_size=0.05, max_tree_depth=15)
        mcmc_prior = MCMC(prior_kernel, num_warmup=_mcmc_warmup, num_samples=_mcmc_samples, num_chains=_mcmc_chains)
        mcmc_prior.run(subkey_prior, Y=None, posterior=False)
        prior_samples = mcmc_prior.get_samples()
        np.savez(output.prior, **prior_samples)

        summary_dict = diagnostics.summary(mcmc_prior.get_samples(group_by_chain=True), group_by_chain=True)
        pd.DataFrame(summary_dict).to_csv(output.prior_summary, index=True)


        posterior_kernel = NUTS(model, step_size=0.05, max_tree_depth=15)
        mcmc_posterior = MCMC(posterior_kernel, num_warmup=_mcmc_warmup, num_samples=_mcmc_samples, num_chains=_mcmc_chains)
        mcmc_posterior.run(subkey_posterior, Y=matrix, posterior=True)
        posterior_samples = mcmc_posterior.get_samples()
        np.savez(output.posterior, **posterior_samples)

        summary_dict = diagnostics.summary(mcmc_posterior.get_samples(group_by_chain=True), group_by_chain=True)
        pd.DataFrame(summary_dict).to_csv(output.posterior_summary, index=True)

        # Generate samples from posterior predictive
        key, subkey = jax.random.split(key)
        subsampled_indices = jax.random.choice(subkey, jnp.arange(_mcmc_samples), shape=(N_SIMULATED,), replace=False)

        def generate_sample(k, idx):
            alpha = posterior_samples["alpha"][idx]
            beta = posterior_samples["beta"][idx]
            gamma = posterior_samples["gamma"][idx]
            delta = posterior_samples["delta"][idx]
            weights, components = muex.convert_to_bernoulli_mixture(muex.Parameters(
                false_positive_rate=alpha,
                false_negative_rate=beta,
                coverage=gamma,
                impurity=delta,
            ), n_genes=matrix.shape[1])
            return bmm.sample_bernoulli_mixture(k, n_samples=matrix.shape[0], mixture_weights=weights, mixture_components=components)

        keys = jax.random.split(key, N_SIMULATED)

        samples = jax.vmap(generate_sample)(keys, subsampled_indices)
        np.save(output.artificial, samples)



rule mle_muex:
    input: "{scenario}/data.csv"
    output:
        params = "{scenario}/muex_mle/params.json",
        artificial = "{scenario}/muex_mle/artificial.npy",
    run:
        df = pd.read_csv(str(input), index_col=0)
        gene_names = df.columns
        matrix = df.values

        n_patients, n_genes = matrix.shape

        rng = np.random.default_rng(101)

        # Estimate maximum likelihood
        ll_fn = muex.get_loglikelihood_function(matrix, from_params=True)
        params_no_errors = muex.estimate_no_errors(matrix)
        

        params_runs = []
        n_inits = 30

        for i in range(n_inits):
            alpha = rng.uniform(low=1e-6, high=0.15)
            beta = rng.uniform(low=1e-6, high=0.15)
            noise_gamma = rng.uniform(low=-0.15, high=0.15)            
            noise_delta = rng.uniform(low=-0.15, high=0.15)

            params0 = muex.Parameters(
                false_positive_rate=alpha,
                false_negative_rate=beta,
                coverage=jnp.clip(params_no_errors.coverage + noise_gamma, min=0.01, max=1.0),
                impurity=jnp.clip(params_no_errors.impurity + noise_delta, min=0.01, max=1.0),
            )            

            params_final, trajectory = muex.em_algorithm(Y=matrix, params0=params0, max_iter=1000, threshold=1e-5)

            if len(trajectory) > 990:
                raise ValueError("There are convergence issues.")

            params_runs.append(params_final)

        params_optimum, loglike_optimum = None, -1e9
        for params in params_runs:
            ll = ll_fn(params)
            if ll > loglike_optimum:
                ll = loglike_optimum
                params_optimum = params

        def get_summary(params: muex.Parameters):
            return {
                "false_positive_rate": float(params.false_positive_rate),
                "false_negative_rate": float(params.false_negative_rate),
                "coverage": float(params.coverage),
                "impurity": float(params.impurity),
                "loglikelihood": float(ll_fn(params)),
            }

        # Save the maximum likelihood estimate
        with open(output.params, "w") as fp:
            json.dump({
                "optimum": get_summary(params_optimum),
                "runs": [get_summary(p) for p in params_runs],
            }, fp=fp)


        weight, components = muex.convert_to_bernoulli_mixture(parameters=params_optimum, n_genes=matrix.shape[1])

        def sample(key):
            return bmm.sample_bernoulli_mixture(key, n_samples=matrix.shape[0], mixture_weights=weight, mixture_components=components)

        keys = jax.random.split(jax.random.PRNGKey(10101), N_SIMULATED)
        X = jnp.asarray(jax.vmap(sample)(keys))
        
        np.save(file=output.artificial, arr=X, allow_pickle=False)



rule plot_mle_comparison:
    input:
        data = "{scenario}/data.csv",
        independent = "{scenario}/independence_mle/artificial.npy",
        muex_model = "{scenario}/muex_mle/artificial.npy",
        muex_bayes = "{scenario}/muex_bayes/artificial.npy",
    output: "{scenario}/summary_statistic_comparison.pdf"
    run:
        fig, axs = subplots_from_axsize(axsize=([5/3, 5/3, 2], 1.1), wspace=[0.5, 0.5], left=0.5, bottom=0.55, top=0.3, right=1.0)
        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)
        
        df = pd.read_csv(str(input.data), index_col=0)
        matrix = df.values
        gene_names = df.columns
        n_genes = len(gene_names)

        model_spec = [
            ("Ind.", "maroon", input.independent),
            ("Exc.", "darkblue", input.muex_model),
            ("Bay.", "green", input.muex_bayes),
        ]

        # --- First plot: bincount ---
        ax = axs[0]
        ax.set_title("Mutation counts")
        ax.set_xlabel("Num. of mutations")
        x_axis = jnp.arange(n_genes + 1)

        ax.plot(x_axis, calculate_bincount(matrix), c="k", marker=".", zorder=30, linestyle=":")

        for name, color, filepath in model_spec:
            array = np.load(filepath, allow_pickle=False)
            stats = jax.vmap(calculate_bincount)(array)

            quantiles = calculate_quantiles(samples=stats, quantiles=[0.05, 0.5, 0.95])
            ax.plot(x_axis, quantiles[1, :], c=color)
            ax.fill_between(x_axis, quantiles[0, :], quantiles[-1, :], color=color, alpha=0.3, edgecolor=None)


        # --- Second plot: mutational frequencies ---
        ax = axs[1]
        ax.set_title("Mutation frequencies")
        x_axis = jnp.arange(n_genes)

        ax.set_xticks(x_axis, gene_names, rotation=30, fontsize="small")

        ax.plot(x_axis, calculate_freq(matrix), c="k", marker=".", zorder=30, linestyle=":")

        for name, color, filepath in model_spec:
            array = np.load(filepath, allow_pickle=False)
            stats = jax.vmap(calculate_freq)(array)
            quantiles = calculate_quantiles(samples=stats, quantiles=[0.05, 0.5, 0.95])
            ax.plot(x_axis, quantiles[1, :], c=color)
            ax.fill_between(x_axis, quantiles[0, :], quantiles[-1, :], color=color, alpha=0.3, edgecolor=None)


        # --- Third plot: MCC ---
        index = jnp.triu_indices(n_genes, 1)
        if len(index[0]) != n_genes * (n_genes - 1) // 2:
            raise ValueError("Length mismatch.")

        ax = axs[2]
        ax.set_title("Correlation")
        x_axis = jnp.arange(len(index[0]))
        xticks = []

        for i1, i2 in zip(index[0], index[1]):
            gene1, gene2 = gene_names[int(i1)], gene_names[int(i2)]
            name = f"{gene1}-{gene2}"
            xticks.append(name)

        ax.set_xticks(x_axis, xticks, rotation=30, fontsize="x-small")

        def mcc_fn(X):
            return calculate_mcc(X)[index]

        ax.plot(x_axis, mcc_fn(matrix), c="k", marker=".", zorder=30, linestyle=":")

        for name, color, filepath in model_spec:
            array = np.load(filepath, allow_pickle=False)
            stats = jax.vmap(mcc_fn)(array)

            stats = np.asarray(stats)
            stats = np.nan_to_num(stats)

            quantiles = calculate_quantiles(samples=stats, quantiles=[0.05, 0.5, 0.95])
            ax.plot(x_axis, quantiles[1, :], c=color)
            ax.fill_between(x_axis, quantiles[0, :], quantiles[-1, :], color=color, alpha=0.3, edgecolor=None)

        legend_patches = [mpatches.Patch(color=color, label=label) for label, color, _ in model_spec]
        legend_patches.append(mpatches.Patch(color="black", label="Data"))

        ax.legend(handles=legend_patches, frameon=False, bbox_to_anchor=(0.99, 1.0))

        fig.savefig(str(output))