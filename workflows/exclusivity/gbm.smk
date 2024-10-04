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