# ====================================================================
# = Experiment showing the behaviour of maximum likelihood estimate  =
# = when the sample size is too small                                =
# ====================================================================
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("Agg")
import seaborn as sns
from subplots_from_axsize import subplots_from_axsize

import json

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

import numpyro
import numpyro.diagnostics as diagnostics
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from jnotype.exclusivity import muex
from jnotype.exclusivity import bmm
from jnotype._utils import order_genotypes

workdir: "generated/exclusivity/small_sample"


N_INITS: int = 20
GROUND_TRUTH_PARAMETERS = muex.Parameters(false_positive_rate=0.05, false_negative_rate=0.05, coverage=0.6, impurity=0.05)
N_GENES = 5
N_SAMPLES = 200

def make_grid(fn, xrange, yrange, steps: int = 101):
    xs = jnp.linspace(*xrange, steps)
    ys = jnp.linspace(*yrange, steps)
    mxs, mys = jnp.meshgrid(xs, ys, indexing="ij")
    return jax.vmap(fn)(mxs, mys)

  
def plot_grid(ax, grid, xrange, yrange, **kwargs):
    extent = (*xrange, *yrange) 

    return ax.imshow(
        grid.T,  # The first array dimension should correspond to the x axis
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Blues",
        **kwargs,
    )

def plot_trajectory(ax, xs, ys, chain_num):
    cmap = plt.get_cmap('Wistia')

    # Normalize chain_num to be between 0 and 1
    norm = plt.Normalize(vmin=-1, vmax=N_INITS)
    normalized_chain_num = norm(chain_num)

    # Get color from the colormap
    color = cmap(normalized_chain_num)
    
    # ax.plot(xs, ys, c=color, marker=".", markersize=1, linewidth=0.4, alpha=0.8)

    ax.scatter(xs[-1:], ys[-1:], c=color, marker=".", s=1)


def muex_model(
    Y,
    posterior: bool,
):
    eps = 1e-3
    alpha = numpyro.sample("alpha", dist.TruncatedNormal(loc=0.0, scale=0.08, low=eps, high=0.2))
    beta = numpyro.sample("beta", dist.TruncatedNormal(loc=0.0, scale=0.08, low=eps, high=0.2))
    coverage = numpyro.sample("coverage", dist.Uniform(low=eps, high=1-eps))
    impurity = numpyro.sample("impurity", dist.TruncatedNormal(loc=0.0, scale=0.2, low=eps, high=1-eps))

    params = muex.Parameters(
        false_positive_rate=alpha,
        false_negative_rate=beta,
        coverage=coverage,
        impurity=impurity,
    )
    
    if posterior:
        ll = muex.get_loglikelihood_function(Y, from_params=True)
        numpyro.factor("loglikelihood", ll(params))


rule all:
    input: ["loglikelihood.pdf", "em_summary.json", "em_table.txt", "prior_posterior.pdf"]


rule sample_data_mutual_exclusivity:
    output: "data.npy"
    run:
        weights, components = muex.convert_to_bernoulli_mixture(GROUND_TRUTH_PARAMETERS, n_genes=N_GENES)
        data = bmm.sample_bernoulli_mixture(
            key=jax.random.PRNGKey(2024),
            n_samples=N_SAMPLES,
            mixture_weights=weights,
            mixture_components=components,
        )

        data = np.asarray(data)
        np.save(str(output), data, allow_pickle=False)


rule fit_muex:
    input: "data.npy"
    output:
        inits = [f"em/init{i}.npy" for i in range(N_INITS)]
    run:
        data = np.load(str(input), allow_pickle=False)
        
        initial_params_list = []

        no_errors_estimate = muex.estimate_no_errors(data)
        initial_params = muex.Parameters(
            false_positive_rate=0.02,
            false_negative_rate=0.02,
            coverage=no_errors_estimate.coverage,
            impurity=no_errors_estimate.impurity,
        )
        initial_params_list.append(initial_params)

        additional_params = [
            [0.05, 0.05, 0.7, 0.1],
            [0.1, 0.2, 0.5, 0.3],
            [0.1, 0.1, 0.3, 0.05],
            [0.01, 0.01, 0.8, 0.01],
            [0.08, 0.08, 0.5, 0.15],
            [0.2, 0.05, 0.5, 0.1],
        ]

        for p in additional_params:
            initial_params = muex.Parameters(
                false_positive_rate=p[0],
                false_negative_rate=p[1],
                coverage=p[2],
                impurity=p[3],
            )
            initial_params_list.append(initial_params)

        if len(initial_params_list) < N_INITS:
            additional_trajectories = N_INITS - len(initial_params_list)
            rng = np.random.default_rng(10101)
            for _ in range(additional_trajectories):
                initial_params = muex.Parameters(
                    false_positive_rate=rng.uniform(low=0.005, high=0.2),
                    false_negative_rate=rng.uniform(low=0.005, high=0.2),
                    coverage=rng.uniform(
                        low=max(0.02, float(no_errors_estimate.coverage) - 0.2),
                        high=min(0.98, float(no_errors_estimate.coverage) + 0.2)),
                    impurity=rng.uniform(
                        low=max(0.02, float(no_errors_estimate.impurity) - 0.2),
                        high=min(0.98, float(no_errors_estimate.impurity) + 0.2)),
                )
                initial_params_list.append(initial_params)


        trajectories = []
        for i, initial_params in enumerate(initial_params_list):
            _, trajectory = muex.em_algorithm(Y=data, params0=initial_params, max_iter=1000, threshold=1e-5)
            
            def wrap_params(p):
                return jnp.asarray([
                    p.false_positive_rate,
                    p.false_negative_rate,
                    p.coverage,
                    p.impurity,
                ])
            trajectory = jnp.stack([wrap_params(p.new_params) for p in trajectory])
            np.save(str(output.inits[i]), np.asarray(trajectory))


rule plot_loglikelihood:
    input:
        data = "data.npy",
        inits = [f"em/init{i}.npy" for i in range(N_INITS)]
    output: "loglikelihood.pdf"
    run:
        data = np.load(input.data, allow_pickle=False)
        ll_fn = muex.get_loglikelihood_function(data, from_params=True)

        @jax.jit
        def ll_rates(alpha: float, beta: float):
            params = muex.Parameters(
                false_positive_rate=alpha,
                false_negative_rate=beta,
                coverage=GROUND_TRUTH_PARAMETERS.coverage,
                impurity=GROUND_TRUTH_PARAMETERS.impurity,
            )
            return ll_fn(params)

        @jax.jit
        def ll_params(gamma: float, delta: float):
            params = muex.Parameters(
                false_positive_rate=GROUND_TRUTH_PARAMETERS.false_positive_rate,
                false_negative_rate=GROUND_TRUTH_PARAMETERS.false_negative_rate,
                coverage=gamma,
                impurity=delta,
            )
            return ll_fn(params)

        ll_perfect = ll_fn(GROUND_TRUTH_PARAMETERS)

        range_alpha = (0.001, 0.4)
        range_beta = (0.001, 0.6)
        range_gamma = (0.2, 0.9)
        range_delta = (0.002, 0.4)

        grid_rates = make_grid(jax.vmap(ll_rates), range_alpha, range_beta)
        grid_params = make_grid(jax.vmap(ll_params), range_gamma, range_delta)

        vmax = max(grid_rates.max(), grid_params.max(), ll_perfect) + 0.01
        vmin = min(ll_perfect, vmax - 5) - 0.01

        fig, axs = subplots_from_axsize(axsize=([2, 4/3, 4/3, 0.1], 1.1), wspace=[0.3, 0.7, 0.1], left=0.3, bottom=0.5, top=0.3, right=0.7)

        # gridspec = {'width_ratios': [2, 1, 1, 0.2]}
        # fig, axs = plt.subplots(1, 4, figsize=(13, 4), gridspec_kw=gridspec)
        
        # ---- Visualisation of the genotypes ----
        ax = axs[0]
        order = order_genotypes(data, reverse=True)
        sns.heatmap(data[order, :].T, ax=ax, vmin=0, vmax=1, cbar=False, cmap="Greys", xticklabels=[], yticklabels=[f"$G_{g}$" for g in range(1, N_GENES+1)])
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.set_xlabel("Simulated genotypes")

        # ---- First plot ----
        for ax in axs[0:3]:
            ax.spines[["top", "right"]].set_visible(False)
        
        ax = axs[1]
        plot_grid(ax, grid_rates, range_alpha, range_beta, vmin=vmin, vmax=vmax)
        ax.set_xlabel("FPR $\\alpha$")
        ax.set_ylabel("FNR $\\beta$")

        input_files = list(input.inits)

        for i, fp in enumerate(input_files):
            trajectory = np.load(fp)
            plot_trajectory(ax, trajectory[:, 0], trajectory[:, 1], chain_num=i)

        ax.scatter(
            [GROUND_TRUTH_PARAMETERS.false_positive_rate], [GROUND_TRUTH_PARAMETERS.false_negative_rate],
            marker="x",
            c="r",
            s=15,
            zorder=100,
        )

        # ---- Second plot ----

        ax = axs[2]
        cbar_data = plot_grid(ax, grid_params, range_gamma, range_delta, vmin=vmin, vmax=vmax)
        ax.set_xlabel("Coverage $\\gamma$")
        ax.set_ylabel("Impurity $\\xi$")

        for i, fp in enumerate(input_files):
            trajectory = np.load(fp)
            plot_trajectory(ax, trajectory[:, 2], trajectory[:, 3], chain_num=i)

        ax.scatter(
            [GROUND_TRUTH_PARAMETERS.coverage], [GROUND_TRUTH_PARAMETERS.impurity],
            marker="x",
            c="r",
            s=15,
            zorder=100,
        )

        # ---- Color bar ----
        ax = axs[-1]
        fig.colorbar(cbar_data, cax=ax, shrink=0.05) #, location='right')

        fig.savefig(str(output))


rule get_estimates:
    input:
        data = "data.npy",
        inits = [f"em/init{i}.npy" for i in range(N_INITS)]
    output: "em_summary.json"
    run:
        data = np.load(input.data, allow_pickle=False)
        ll_fn = muex.get_loglikelihood_function(data, from_params=True)

        def get_summary(params: muex.Parameters, trajectory_length = None):
            return {
                "false_positive_rate": float(params.false_positive_rate),
                "false_negative_rate": float(params.false_negative_rate),
                "coverage": float(params.coverage),
                "impurity": float(params.impurity),
                "loglikelihood": float(ll_fn(params)),
                "trajectory_length": trajectory_length,
            }

        ground_truth = get_summary(GROUND_TRUTH_PARAMETERS, None)

        max_trajectory_length = -1
        values = []
        for fp in input.inits:
            trajectory = np.load(fp, allow_pickle=False)
            p = trajectory[-1]
            params = muex.Parameters(
                false_positive_rate=p[0],
                false_negative_rate=p[1],
                coverage=p[2],
                impurity=p[3],
            )
            values.append(get_summary(params, len(trajectory)))
            max_trajectory_length = max(max_trajectory_length, len(trajectory))

        i_opt = None
        loglike_opt = -1e9
        for i, v in enumerate(values):
            if v["loglikelihood"] > loglike_opt:
                i_opt = i
                loglike_opt = loglike_opt

        summary = {
            "ground_truth": ground_truth,
            "optimum": values[i_opt],
            "max_trajectory_length": max_trajectory_length,
            "runs": values,
        }

        with open(str(output), "w") as fh:
            json.dump(summary, fh)


rule create_table:
    input: "em_summary.json"
    output: "em_table.txt"
    run:
        with open(str(input)) as fp:
            summary = json.load(fp)
        
        data_rows = []

        # Add ground truth data
        gt = summary.get('ground_truth', {})
        data_rows.append({
            'name': 'Ground truth',
            'false_positive_rate': gt.get('false_positive_rate', None),
            'false_negative_rate': gt.get('false_negative_rate', None),
            'coverage': gt.get('coverage', None),
            'impurity': gt.get('impurity', None),
            'loglikelihood': gt.get('loglikelihood', None)
        })

        runs_sorted = sorted(summary["runs"], key=lambda run: run["loglikelihood"], reverse=True)

        # Add individual runs
        for idx, run in enumerate(runs_sorted):
            data_rows.append({
                'name': f'Run {idx + 1}',
                'false_positive_rate': run.get('false_positive_rate', None),
                'false_negative_rate': run.get('false_negative_rate', None),
                'coverage': run.get('coverage', None),
                'impurity': run.get('impurity', None),
                'loglikelihood': run.get('loglikelihood', None),
            })

        # Create LaTeX table components
        header = (
            "\\begin{tabular}{lrrrrr}\n"
            "\\hline\n"
            " & False Positive Rate & False Negative Rate & Coverage & Impurity & Loglikelihood\\\\\n"
            "\\hline\n"
        )

        footer = "\\hline\n\\end{tabular}\n"

        # Construct the table rows
        rows = ''
        for row in data_rows:
            # Format numerical values and handle None
            fpr = f"{row['false_positive_rate']:.3f}" if row['false_positive_rate'] is not None else ''
            fnr = f"{row['false_negative_rate']:.3f}" if row['false_negative_rate'] is not None else ''
            coverage = f"{row['coverage']:.3f}" if row['coverage'] is not None else ''
            impurity = f"{row['impurity']:.3f}" if row['impurity'] is not None else ''
            loglikelihood = f"${row['loglikelihood']:.3f}$" if row['loglikelihood'] is not None else ''
            rows += f"{row['name']} & {fpr} & {fnr} & {coverage} & {impurity} & {loglikelihood} \\\\\n"

        # Combine header, rows, and footer to form the complete table
        table = header + rows + footer

        # Write the LaTeX table to the output file
        with open(str(output), 'w') as f:
            f.write(table)


rule compare_prior_posterior:
    input:
        prior_samples = "prior/samples.npz",
        posterior_samples = "posterior/samples.npz"
    output:
        plot = "prior_posterior.pdf"
    run:
        fig, axs = subplots_from_axsize(axsize=([1, 1, 1, 1], 0.6), wspace=[0.25, 0.25, 0.25], left=0.1, bottom=0.5, top=0.02, right=1.0)

        names = [
            ("alpha", GROUND_TRUTH_PARAMETERS.false_positive_rate, "FPR $\\alpha$"),
            ("beta", GROUND_TRUTH_PARAMETERS.false_negative_rate, "FNR $\\beta$"),
            ("coverage", GROUND_TRUTH_PARAMETERS.coverage, "Coverage $\\gamma$"),
            ("impurity", GROUND_TRUTH_PARAMETERS.impurity, "Impurity $\\xi$"),
        ]

        prior = np.load(input.prior_samples)
        posterior = np.load(input.posterior_samples)

        for (key, val, name), ax in zip(names, axs):
            ax.spines[["top", "left", "right"]].set_visible(False)
            ax.set_xlabel(name)
            ax.set_yticks([])
            
            ax.axvline(val, color="k", linestyle=":", linewidth=1.0)

            ax.hist(prior[key], bins=20, alpha=0.5, color="salmon", density=True)
            ax.hist(posterior[key], bins=20, alpha=0.5, color="darkblue", density=True)

        model_spec = [
            ("Prior", "salmon"),
            ("Posterior", "darkblue"),
        ]

        ax = axs[-1]
        legend_patches = [mpatches.Patch(color=color, label=label) for label, color in model_spec]
        ax.legend(handles=legend_patches, frameon=False, bbox_to_anchor=(0.7, 0.9))

        fig.savefig(output.plot)


rule sample_prior:
    output:
        samples = "prior/samples.npz",
        summary = "prior/summary.csv"
    run:
        num_samples = 8_000
        num_chains = 5

        nuts_kernel = NUTS(muex_model, step_size=0.05, max_tree_depth=15)
        mcmc = MCMC(nuts_kernel, num_warmup=num_samples, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(jax.random.PRNGKey(121), Y=None, posterior=False)
        samples = mcmc.get_samples()
        np.savez(output.samples, **samples)

        summary_dict = diagnostics.summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
        pd.DataFrame(summary_dict).to_csv(output.summary, index=True)


rule sample_posterior:
    input:
        data = "data.npy"
    output:
        samples = "posterior/samples.npz",
        summary = "posterior/summary.csv"
    run:
        data = np.load(input.data, allow_pickle=False)

        num_samples = 8_000
        num_chains = 5

        nuts_kernel = NUTS(muex_model, step_size=0.05, max_tree_depth=15)
        mcmc = MCMC(nuts_kernel, num_warmup=num_samples, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(jax.random.PRNGKey(122), Y=data, posterior=True)
        samples = mcmc.get_samples()
        np.savez(output.samples, **samples)

        summary_dict = diagnostics.summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
        pd.DataFrame(summary_dict).to_csv(output.summary, index=True)
