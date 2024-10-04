# ====================================================================
# = Experiment showing the behaviour of maximum likelihood estimate  =
# = when the sample size is too small                                =
# ====================================================================
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from subplots_from_axsize import subplots_from_axsize

import json

import numpy as np
import jax
import jax.numpy as jnp

from jnotype.exclusivity import muex
from jnotype.exclusivity import bmm
from jnotype._utils import order_genotypes

workdir: "generated/exclusivity/small_sample"


rule all:
    input: ["loglikelihood.pdf", "em_summary.json"]

GROUND_TRUTH_PARAMETERS = muex.Parameters(false_positive_rate=0.05, false_negative_rate=0.05, coverage=0.6, impurity=0.05)
N_GENES = 5
N_SAMPLES = 200

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

N_INITS = 7

rule fit_muex:
    input: "data.npy"
    output:
        inits = [f"em/init{i}.npy" for i in range(N_INITS)]
    run:
        data = np.load(str(input), allow_pickle=False)
        
        initial_params_list = []

        initial_params = muex.estimate_no_errors(data)
        initial_params = muex.Parameters(
            false_positive_rate=0.02,
            false_negative_rate=0.02,
            coverage=initial_params.coverage,
            impurity=initial_params.impurity,
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
    
    ax.plot(xs, ys, c=color, marker=".", markersize=1, linewidth=0.4, alpha=0.8)


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

        i_opt = None
        loglike_opt = -1e9
        for i, v in enumerate(values):
            if v["loglikelihood"] > loglike_opt:
                i_opt = i
                loglike_opt = loglike_opt

        summary = {
            "ground_truth": ground_truth,
            "optimum": values[i_opt],
            "runs": values,
        }

        with open(str(output), "w") as fh:
            json.dump(summary, fh)
