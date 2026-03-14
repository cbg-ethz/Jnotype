import re

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import comparison

from contextlib import contextmanager, redirect_stdout

matplotlib.use("agg")


@contextmanager
def redirect_print(filename, mode="w", encoding=None):
    """
    Redirect sys.stdout to the given filename.

    :param filename: path of the file to write stdout into
    :param mode: open mode, e.g. 'w' (default) or 'a'
    :param encoding: file encoding (if None, uses system default)
    """
    f = open(filename, mode, encoding=encoding) if encoding else open(filename, mode)
    try:
        with redirect_stdout(f):
            yield
    finally:
        f.close()


def generate_observations(key, epsilon, n_samples, p_true):
    key1, key2 = jax.random.split(key)
    data = jax.random.bernoulli(key1, p=p_true, shape=(n_samples, len(p_true)))

    mask = jax.random.bernoulli(key2, p=1 - epsilon, shape=(n_samples, 1))
    return data * mask


DEFAULT_COLORS = {
    "bootstrap": "purple",
    "bootstrap-1000": "indigo",
    "dirichlet-tiny": "salmon",
    "dirichlet1": "red",
    "dirichlet-10": "maroon",
    "parametric": "green",
    "adjustable": "blue",
}

DEFAULT_MAIN_NAMES = {
    "dirichlet-tiny": "Dirichlet ($\\alpha\\to 0$)",
    "dirichlet-10": "Dirichlet ($\\alpha=10$)",
    "parametric": "Parametric",
    "adjustable": "MDP",
}

BASE_METHOD_SPECS = {
    "bootstrap": {
        "family": "bootstrap",
        "predictive_samples": "n_datapoints",
        "seed_base": 11,
        "seed_scale": 1_152_441,
    },
    "bootstrap-1000": {
        "family": "bootstrap",
        "predictive_samples": 1000,
        "seed_base": 51,
        "seed_scale": 1_524_415,
    },
    "parametric": {
        "family": "parametric",
        "predictive_samples": 1000,
        "seed_base": 10,
        "seed_scale": 115,
        "needs_mcmc_summary": True,
    },
    "adjustable": {
        "family": "adjustable",
        "predictive_samples": 1000,
        "seed_base": 10,
        "seed_scale": 115,
        "needs_mcmc_summary": True,
    },
    "dirichlet1": {
        "family": "dirichlet",
        "alpha": 1.0,
        "predictive_samples": 1000,
        "seed_base": 10,
        "seed_scale": 215_051,
    },
    "dirichlet-tiny": {
        "family": "dirichlet",
        "alpha": 1e-6,
        "predictive_samples": 1000,
        "seed_base": 10,
        "seed_scale": 21_235_051,
    },
    "dirichlet-10": {
        "family": "dirichlet",
        "alpha": 10.0,
        "predictive_samples": 1000,
        "seed_base": 10,
        "seed_scale": 21_235_051,
    },
}

SCENARIOS = {
    "comparison": {
        "workdir": "generated/nonparametric/comparison",
        "p_true": {"start": 5e-2, "stop": 0.1, "num": 10},
        "epsilons": [1e-6, 0.5, 0.8],
        "datapoints": [10, 30, 100, 1000],
        "n_bootstrap": 200,
        "population_seed": 101,
        "data_seed_base": 101,
        "data_epsilon_scale": 105_241,
        "methods": [
            "bootstrap",
            "dirichlet-tiny",
            "dirichlet1",
            "dirichlet-10",
            "parametric",
            "adjustable",
        ],
        "methods_main": ["dirichlet-tiny", "dirichlet-10", "parametric", "adjustable"],
        "enable_main_figure": True,
        "emr_bins": {"start": 0.0, "stop": 0.4, "num": 30},
        "emr_xlim": [0.0, 0.4],
        "emr_hide_xticks": True,
        "final_plot_links": {
            ("well-specified", "small"): "plots/1e-06/10/main.pdf",
            ("well-specified", "large"): "plots/1e-06/1000/main.pdf",
            ("misspecified", "small"): "plots/0.8/10/main.pdf",
            ("misspecified", "large"): "plots/0.8/1000/main.pdf",
        },
    },
    "comparison-5-sites": {
        "workdir": "generated/nonparametric/comparison-5-sites",
        "p_true": {"start": 0.1, "stop": 0.25, "num": 5},
        "epsilons": [1e-6, 0.5, 0.8],
        "datapoints": [5, 10, 30, 100, 1000],
        "n_bootstrap": 500,
        "population_seed": 101,
        "data_seed_base": 101,
        "data_epsilon_scale": 105_241,
        "methods": ["bootstrap", "dirichlet1", "parametric", "adjustable"],
        "colors": {"bootstrap": "maroon"},
        "method_overrides": {
            "adjustable": {"seed_scale": 1105},
            "dirichlet1": {"seed_scale": 21_235_051},
        },
        "emr_bins": {"start": 0.0, "stop": 0.2, "num": 20},
        "emr_xlim": [0.0, 0.2],
        "emr_hide_xticks": False,
        "enable_main_figure": False,
    },
    "comparison-5-sites-new": {
        "workdir": "generated/nonparametric/comparison-5-sites-new",
        "p_true": {"start": 0.1, "stop": 0.25, "num": 5},
        "epsilons": [1e-6, 0.5, 0.8],
        "datapoints": [5, 10, 30, 100, 1000, 5000],
        "n_bootstrap": 500,
        "population_seed": 201,
        "data_seed_base": 201,
        "data_epsilon_scale": 15_241,
        "methods": [
            "bootstrap",
            "bootstrap-1000",
            "dirichlet-tiny",
            "dirichlet1",
            "dirichlet-10",
            "parametric",
            "adjustable",
        ],
        "method_overrides": {
            "adjustable": {"seed_scale": 1105},
            "dirichlet1": {"seed_scale": 21_235_051},
        },
        "emr_bins": {"start": 0.0, "stop": 0.4, "num": 30},
        "emr_xlim": [0.0, 0.4],
        "emr_hide_xticks": True,
        "enable_main_figure": False,
    },
    "comparison-20-sites": {
        "workdir": "generated/nonparametric/comparison-20-sites",
        "p_true": {"start": 0.1, "stop": 0.25, "num": 20},
        "epsilons": [1e-6, 0.5, 0.8],
        "datapoints": [10, 30, 100, 1000],
        "n_bootstrap": 500,
        "population_seed": 101,
        "data_seed_base": 101,
        "data_epsilon_scale": 105_241,
        "methods": ["bootstrap", "dirichlet1", "parametric", "adjustable"],
        "colors": {"bootstrap": "maroon"},
        "method_overrides": {
            "adjustable": {"seed_scale": 1105},
            "dirichlet1": {"seed_scale": 21_235_051},
        },
        "emr_bins": {"start": 0.0, "stop": 0.2, "num": 20},
        "emr_xlim": [0.0, 0.2],
        "emr_hide_xticks": False,
        "enable_main_figure": False,
    },
}


# Select a predefined experiment profile (e.g. via --config scenario=comparison-5-sites).
SCENARIO_NAME = config.get("scenario", "comparison")
if SCENARIO_NAME not in SCENARIOS:
    available = ", ".join(sorted(SCENARIOS))
    raise ValueError(f"Unknown scenario '{SCENARIO_NAME}'. Available: {available}")

SCENARIO = SCENARIOS[SCENARIO_NAME]
METHOD_OVERRIDES = SCENARIO.get("method_overrides", {})
METHODS = list(SCENARIO["methods"])
METHOD_SPECS = {}
for method in METHODS:
    if method not in BASE_METHOD_SPECS:
        raise ValueError(f"Method '{method}' does not have a base specification")
    METHOD_SPECS[method] = {**BASE_METHOD_SPECS[method], **METHOD_OVERRIDES.get(method, {})}

P_CFG = SCENARIO["p_true"]
P_TRUE = jnp.linspace(P_CFG["start"], P_CFG["stop"], P_CFG["num"])

EPSILONS = list(SCENARIO["epsilons"])
DATAPOINTS = list(SCENARIO["datapoints"])
N_BOOTSTRAP = int(SCENARIO["n_bootstrap"])

METHODS_EMR = list(SCENARIO.get("methods_emr", METHODS))
METHODS_ZERO_MASS = list(SCENARIO.get("methods_zero_mass", METHODS))
ENABLE_MAIN_FIGURE = bool(SCENARIO.get("enable_main_figure", False))
METHODS_MAIN = list(SCENARIO.get("methods_main", []))
if ENABLE_MAIN_FIGURE and not METHODS_MAIN:
    raise ValueError("'enable_main_figure' is true but 'methods_main' is empty")

if FINAL_PLOT_LINKS := dict(SCENARIO.get("final_plot_links", {})):
    if not ENABLE_MAIN_FIGURE:
        raise ValueError("'final_plot_links' require 'enable_main_figure' to be true")

ALL_METHOD_REFERENCES = set(METHODS) | set(METHODS_EMR) | set(METHODS_ZERO_MASS) | set(METHODS_MAIN)
missing_methods = sorted(m for m in ALL_METHOD_REFERENCES if m not in METHOD_SPECS)
if missing_methods:
    raise ValueError(f"Methods missing from scenario method list: {missing_methods}")

COLORS = dict(DEFAULT_COLORS)
COLORS.update(SCENARIO.get("colors", {}))

EMR_BINS = dict(SCENARIO.get("emr_bins", {"start": 0.0, "stop": 0.2, "num": 20}))
EMR_XLIM = tuple(SCENARIO.get("emr_xlim", [0.0, 0.2]))
EMR_HIDE_XTICKS = bool(SCENARIO.get("emr_hide_xticks", False))

MAIN_NAMES = dict(DEFAULT_MAIN_NAMES)
MAIN_NAMES.update(SCENARIO.get("main_names", {}))

POPULATION_SEED = int(SCENARIO["population_seed"])
DATA_SEED_BASE = int(SCENARIO["data_seed_base"])
DATA_EPSILON_SCALE = int(SCENARIO["data_epsilon_scale"])

MCMC_METHODS = [m for m in METHODS if METHOD_SPECS[m].get("needs_mcmc_summary", False)]
SIMPLE_METHODS = [m for m in METHODS if m not in MCMC_METHODS]


workdir: SCENARIO["workdir"]


def _regex_union(items):
    if not items:
        return "$^"
    return "|".join(re.escape(item) for item in items)


def _predictive_paths(epsilon, n_datapoints, methods):
    return [f"predictive/{epsilon}/{method}/{n_datapoints}.npy" for method in methods]


def _sample_size(spec, n_datapoints):
    requested = spec["predictive_samples"]
    if requested == "n_datapoints":
        return n_datapoints
    return int(requested)


def _method_seed(spec, n_datapoints, epsilon):
    return int(spec["seed_base"] + n_datapoints + int(spec["seed_scale"] * epsilon))


def _emr_predictive_inputs(wildcards):
    return _predictive_paths(wildcards.epsilon, wildcards.n_datapoints, METHODS_EMR)


def _zero_mass_predictive_inputs(wildcards):
    return _predictive_paths(wildcards.epsilon, wildcards.n_datapoints, METHODS_ZERO_MASS)


def _main_predictive_inputs(wildcards):
    return _predictive_paths(wildcards.epsilon, wildcards.n_datapoints, METHODS_MAIN)


population_targets = expand("population/{epsilon}.npy", epsilon=EPSILONS)
data_targets = expand("data/{epsilon}/{n_datapoints}.npy", epsilon=EPSILONS, n_datapoints=DATAPOINTS)
predictive_targets = expand(
    "predictive/{epsilon}/{method}/{n_datapoints}.npy",
    epsilon=EPSILONS,
    method=METHODS,
    n_datapoints=DATAPOINTS,
)
mcmc_targets = expand(
    "mcmc_summary/{epsilon}/{method}/{n_datapoints}.txt",
    epsilon=EPSILONS,
    method=MCMC_METHODS,
    n_datapoints=DATAPOINTS,
)
plot_emr_targets = expand(
    "plots/{epsilon}/{n_datapoints}/expected-mutation-rate.pdf",
    epsilon=EPSILONS,
    n_datapoints=DATAPOINTS,
)
plot_zero_mass_targets = expand(
    "plots/{epsilon}/{n_datapoints}/zero_mass.pdf",
    epsilon=EPSILONS,
    n_datapoints=DATAPOINTS,
)
plot_main_targets = []
if ENABLE_MAIN_FIGURE:
    plot_main_targets = expand(
        "plots/{epsilon}/{n_datapoints}/main.pdf",
        epsilon=EPSILONS,
        n_datapoints=DATAPOINTS,
    )
final_plot_targets = [
    f"plots/experiment-results/{specification}/{sample_size}.pdf"
    for specification, sample_size in FINAL_PLOT_LINKS
]


rule all:
    input:
        population_targets,
        data_targets,
        predictive_targets,
        mcmc_targets,
        plot_emr_targets,
        plot_zero_mass_targets,
        plot_main_targets,
        final_plot_targets,


rule generate_population:
    output: "population/{epsilon}.npy"
    run:
        epsilon = float(wildcards.epsilon)
        key = jax.random.PRNGKey(POPULATION_SEED)
        population = generate_observations(key, epsilon, 200_000, P_TRUE)
        np.save(str(output), population)


rule generate_data:
    output: "data/{epsilon}/{n_datapoints}.npy"
    run:
        epsilon = float(wildcards.epsilon)
        n_datapoints = int(wildcards.n_datapoints)

        seed = DATA_SEED_BASE + n_datapoints + int(DATA_EPSILON_SCALE * epsilon)
        key = jax.random.PRNGKey(seed)
        data = generate_observations(key, epsilon, n_datapoints, P_TRUE)
        np.save(str(output), data)


if SIMPLE_METHODS:

    rule get_predictive_simple:
        output: "predictive/{epsilon}/{method}/{n_datapoints}.npy"
        input: "data/{epsilon}/{n_datapoints}.npy"
        wildcard_constraints:
            method=_regex_union(SIMPLE_METHODS)
        run:
            epsilon = float(wildcards.epsilon)
            n_datapoints = int(wildcards.n_datapoints)
            method = wildcards.method

            data = jnp.asarray(np.load(input[0]))
            spec = METHOD_SPECS[method]
            n_samples = _sample_size(spec, n_datapoints)
            seed = _method_seed(spec, n_datapoints, epsilon)

            if spec["family"] == "bootstrap":
                key = jax.random.PRNGKey(seed)
                model = comparison.SimpleBootstrap()
                model.train(data, key=None)
                predictive = np.stack(
                    [
                        np.array(model.sample_predictive(jax.random.fold_in(key, i), n_samples=n_samples))
                        for i in range(N_BOOTSTRAP)
                    ],
                    axis=0,
                )
            elif spec["family"] == "dirichlet":
                key = jax.random.PRNGKey(seed)
                key_train, key_sample = jax.random.split(key)
                model = comparison.DirichletPriorModel(alpha=spec["alpha"])
                model.train(data, key=key_train)
                predictive = np.stack(
                    [
                        model.sample_predictive(jax.random.fold_in(key_sample, i), n_samples=n_samples)
                        for i in range(N_BOOTSTRAP)
                    ],
                    axis=0,
                )
            else:
                raise ValueError(f"Unsupported simple method family: {spec['family']}")

            np.save(str(output), predictive)


if MCMC_METHODS:

    rule get_predictive_mcmc:
        output:
            predictive="predictive/{epsilon}/{method}/{n_datapoints}.npy",
            mcmc_summary="mcmc_summary/{epsilon}/{method}/{n_datapoints}.txt"
        input: "data/{epsilon}/{n_datapoints}.npy"
        wildcard_constraints:
            method=_regex_union(MCMC_METHODS)
        run:
            epsilon = float(wildcards.epsilon)
            n_datapoints = int(wildcards.n_datapoints)
            method = wildcards.method

            data = jnp.asarray(np.load(input[0]))
            spec = METHOD_SPECS[method]
            n_samples = _sample_size(spec, n_datapoints)
            seed = _method_seed(spec, n_datapoints, epsilon)
            key = jax.random.PRNGKey(seed)
            key_train, key_sample = jax.random.split(key)

            if spec["family"] == "parametric":
                model = comparison.ParametricModel()
            elif spec["family"] == "adjustable":
                model = comparison.AdjustableModel()
            else:
                raise ValueError(f"Unsupported MCMC method family: {spec['family']}")

            model.train(data, key=key_train)
            with redirect_print(output.mcmc_summary):
                model.mcmc.print_summary()

            predictive = np.stack(
                [
                    model.sample_predictive(jax.random.fold_in(key_sample, i), n_samples=n_samples)
                    for i in range(N_BOOTSTRAP)
                ],
                axis=0,
            )
            np.save(output.predictive, predictive)


rule plot_performance_expected_mutation_rate:
    output: "plots/{epsilon}/{n_datapoints}/expected-mutation-rate.pdf"
    input:
        population="population/{epsilon}.npy",
        predictive=_emr_predictive_inputs,
    run:
        def emr_fn(i):
            def f(y):
                return y[:, i].mean()

            return f

        population = jnp.asarray(np.load(input.population))
        g = population.shape[1]

        files = dict(zip(METHODS_EMR, input.predictive))

        fig, axs = plt.subplots(g, len(files), sharex=True, sharey=False)
        axs = np.asarray(axs).reshape(g, len(files))

        bins = np.linspace(EMR_BINS["start"], EMR_BINS["stop"], EMR_BINS["num"])

        for i in range(g):
            summary = emr_fn(i)
            for j, (method, filename) in enumerate(files.items()):
                ax = axs[i, j]
                ax.axvline(summary(population), linewidth=3, linestyle=":", color="black")

                predictive = jnp.asarray(np.load(filename))
                ax.hist(
                    jax.vmap(summary)(predictive),
                    bins=bins,
                    density=True,
                    color=COLORS[method],
                    alpha=0.2,
                )

        for ax in axs.ravel():
            ax.set_xlim(*EMR_XLIM)
            ax.spines[["top", "left", "right"]].set_visible(False)
            ax.set_yticks([])
            if EMR_HIDE_XTICKS:
                ax.set_xticks([])

        fig.savefig(output[0])


rule plot_performance_zero_mass:
    output: "plots/{epsilon}/{n_datapoints}/zero_mass.pdf"
    input:
        population="population/{epsilon}.npy",
        predictive=_zero_mass_predictive_inputs,
    run:
        def summary(y):
            return jnp.mean(y.sum(axis=-1) < 1)

        population = jnp.asarray(np.load(input.population))
        files = dict(zip(METHODS_ZERO_MASS, input.predictive))

        fig, axs = plt.subplots(1, len(files), sharex=True, sharey=False)
        axs = np.atleast_1d(axs)

        for ax in axs.ravel():
            ax.axvline(summary(population), linewidth=3, linestyle=":", color="black")

        for i, (method, filename) in enumerate(files.items()):
            ax = axs[i]
            predictive = jnp.asarray(np.load(filename))
            ax.hist(
                jax.vmap(summary)(predictive),
                bins=20,
                density=True,
                color=COLORS[method],
                alpha=0.2,
            )

        for ax in axs.ravel():
            ax.spines[["top", "left", "right"]].set_visible(False)
            ax.set_yticks([])

        fig.savefig(output[0])


if ENABLE_MAIN_FIGURE:

    rule plot_main_figure:
        output: "plots/{epsilon}/{n_datapoints}/main.pdf"
        input:
            population="population/{epsilon}.npy",
            predictive=_main_predictive_inputs,
        run:
            from subplots_from_axsize import subplots_from_axsize

            def zero_atom_fn(y):
                return jnp.mean(y.sum(axis=-1) < 1)

            def mut_rate_fn(y):
                return y[:, 0].mean()

            summary_stats = {
                "Mutation rate": {
                    "fn": mut_rate_fn,
                    "bounds": (0, 0.3),
                },
                "Zero atom": {
                    "fn": zero_atom_fn,
                    "bounds": (0, 1),
                },
            }

            population = jnp.asarray(np.load(input.population))
            files = dict(zip(METHODS_MAIN, input.predictive))

            fig, axs = subplots_from_axsize(
                nrows=len(summary_stats),
                ncols=len(files),
                axsize=(1.5, 1.2),
                sharex="row",
                sharey=False,
                top=0.4,
            )

            for i, (stat_name, stat_cfg) in enumerate(summary_stats.items()):
                for j, (method, filename) in enumerate(files.items()):
                    ax = axs[i, j]

                    if i == 0:
                        ax.set_title(MAIN_NAMES.get(method, method))
                    if j == 0:
                        ax.set_ylabel(stat_name)

                    stat_fn = stat_cfg["fn"]
                    ax.axvline(stat_fn(population), linewidth=3, linestyle=":", color="black")

                    predictive = jnp.asarray(np.load(filename))
                    ax.hist(
                        jax.vmap(stat_fn)(predictive),
                        bins=30,
                        density=True,
                        color=COLORS[method],
                    )

            for ax in axs.ravel():
                ax.spines[["top", "left", "right"]].set_visible(False)
                ax.set_yticks([])

            fig.savefig(output[0])


if FINAL_PLOT_LINKS:

    def _final_plot_source(wildcards):
        key = (wildcards.specification, wildcards.sample_size)
        if key not in FINAL_PLOT_LINKS:
            raise ValueError(f"No source configured for final plot key {key}")
        return FINAL_PLOT_LINKS[key]


    rule link_final_plots:
        output: "plots/experiment-results/{specification}/{sample_size}.pdf"
        input: _final_plot_source
        wildcard_constraints:
            specification=_regex_union(sorted({k[0] for k in FINAL_PLOT_LINKS})),
            sample_size=_regex_union(sorted({k[1] for k in FINAL_PLOT_LINKS})),
        shell:
            "cp {input} {output}"
