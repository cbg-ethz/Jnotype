import jax.numpy as jnp
import jax
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

import comparison

import sys
from contextlib import contextmanager, redirect_stdout

@contextmanager
def redirect_print(filename, mode='w', encoding=None):
    """
    Redirect sys.stdout to the given filename.

    :param filename: path of the file to write stdout into
    :param mode: open mode, e.g. 'w' (default) or 'a'
    :param encoding: file encoding (if None, uses system default)
    """
    # Open the file
    f = open(filename, mode, encoding=encoding) if encoding else open(filename, mode)
    try:
        # Delegate to the stdlib context‐manager
        with redirect_stdout(f):
            yield
    finally:
        f.close()


workdir: "generated/nonparametric/comparison-5-sites"

def generate_data(key, epsilon, n_samples, p_true):
    key1, key2 = jax.random.split(key)
    data = jax.random.bernoulli(key1, p=p_true, shape=(n_samples, len(p_true)))

    mask = jax.random.bernoulli(key2, p=1 - epsilon, shape=(n_samples, 1))
    return data * mask

P_TRUE = jnp.linspace(0.1, 0.25, 5)

EPSILONS = [1e-6, 0.5, 0.8]
DATAPOINTS = [5, 10, 30, 100, 1000]
METHODS = ["bootstrap", "parametric", "dirichlet1", "adjustable"]
N_BOOTSTRAP = 500

COLORS = {
    "bootstrap": "maroon",
    "dirichlet1": "red",
    "parametric": "green",
    "adjustable": "blue",
}


rule all:
    input:
        population=expand("population/{epsilon}.npy", epsilon=EPSILONS),
        data=expand("data/{epsilon}/{n_datapoints}.npy", epsilon=EPSILONS, n_datapoints=DATAPOINTS),
        predictive=expand("predictive/{epsilon}/{method}/{n_datapoints}.npy", epsilon=EPSILONS, n_datapoints=DATAPOINTS, method=METHODS),
        plot_emr=expand("plots/{epsilon}/{n_datapoints}/expected-mutation-rate.pdf", epsilon=EPSILONS, n_datapoints=DATAPOINTS),
        plot_zero_mass=expand("plots/{epsilon}/{n_datapoints}/zero_mass.pdf", epsilon=EPSILONS, n_datapoints=DATAPOINTS),

rule generate_population:
    output: "population/{epsilon}.npy"
    run:
        epsilon = float(wildcards.epsilon)
        key = jax.random.PRNGKey(101)
        population = generate_data(key, epsilon, 200_000, P_TRUE)
        np.save(str(output), population)


rule generate_data:
    output: "data/{epsilon}/{n_datapoints}.npy"
    run:
        epsilon = float(wildcards.epsilon)
        n_datapoints = int(wildcards.n_datapoints)

        key = jax.random.PRNGKey(101 + n_datapoints + int(105241 * epsilon))
        data = generate_data(key, epsilon, n_datapoints, P_TRUE)
        np.save(str(output), data)


rule get_predictive_bootstrap:
    output: "predictive/{epsilon}/bootstrap/{n_datapoints}.npy"
    input: "data/{epsilon}/{n_datapoints}.npy"
    run:
        epsilon = float(wildcards.epsilon)
        n_datapoints = int(wildcards.n_datapoints)
        data = np.load(input[0])
        data = jnp.asarray(data)

        subkey = jax.random.PRNGKey(11 + n_datapoints + int(1152441 * epsilon))
        model = comparison.SimpleBootstrap()
        model.train(data, key=None)

        predictive = np.stack([
            np.array(model.sample_predictive(jax.random.fold_in(subkey, i), n_samples=n_datapoints)) for i in range(N_BOOTSTRAP)
        ], axis=0)
        np.save(str(output), predictive)


rule get_predictive_parametric:
    output:
        predictive="predictive/{epsilon}/parametric/{n_datapoints}.npy",
        mcmc_summary="mcmc_summary/{epsilon}/parametric/{n_datapoints}.txt"
    input: "data/{epsilon}/{n_datapoints}.npy"
    run:
        epsilon = float(wildcards.epsilon)
        n_datapoints = int(wildcards.n_datapoints)
        data = np.load(input[0])
        data = jnp.asarray(data)

        key = jax.random.PRNGKey(10 + n_datapoints + int(115 * epsilon))
        key, key1, key2 = jax.random.split(key, 3)

        model = comparison.ParametricModel()
        model.train(data, key=key1)

        with redirect_print(output.mcmc_summary):
            model.mcmc.print_summary()

        predictive = np.stack([
            model.sample_predictive(jax.random.fold_in(key2, i), n_samples=1000) for i in range(N_BOOTSTRAP)
        ], axis=0)
        np.save(output.predictive, predictive)


rule get_predictive_adjustable:
    output:
        predictive="predictive/{epsilon}/adjustable/{n_datapoints}.npy",
        mcmc_summary="mcmc_summary/{epsilon}/adjustable/{n_datapoints}.txt"
    input: "data/{epsilon}/{n_datapoints}.npy"
    run:
        epsilon = float(wildcards.epsilon)
        n_datapoints = int(wildcards.n_datapoints)
        data = np.load(input[0])
        data = jnp.asarray(data)

        key = jax.random.PRNGKey(10 + n_datapoints + int(1105 * epsilon))
        key, key1, key2 = jax.random.split(key, 3)

        model = comparison.AdjustableModel()
        model.train(data, key=key1)

        with redirect_print(output.mcmc_summary):
            model.mcmc.print_summary()

        predictive = np.stack([
            model.sample_predictive(jax.random.fold_in(key2, i), n_samples=1000) for i in range(N_BOOTSTRAP)
        ], axis=0)
        np.save(output.predictive, predictive)



rule get_predictive_dirichlet1:
    output:
        predictive="predictive/{epsilon}/dirichlet1/{n_datapoints}.npy",
    input: "data/{epsilon}/{n_datapoints}.npy"
    run:
        epsilon = float(wildcards.epsilon)
        n_datapoints = int(wildcards.n_datapoints)
        data = np.load(input[0])
        data = jnp.asarray(data)

        key = jax.random.PRNGKey(10 + n_datapoints + int(21235051 * epsilon))
        key, key1, key2 = jax.random.split(key, 3)

        model = comparison.DirichletPriorModel(alpha=1.0)
        model.train(data, key=key1)

        predictive = np.stack([
            model.sample_predictive(jax.random.fold_in(key2, i), n_samples=1000) for i in range(N_BOOTSTRAP)
        ], axis=0)
        np.save(output.predictive, predictive)



rule plot_performance_expected_mutation_rate:
    output: "plots/{epsilon}/{n_datapoints}/expected-mutation-rate.pdf"
    input:
        population="population/{epsilon}.npy",
        parametric="predictive/{epsilon}/parametric/{n_datapoints}.npy",
        bootstrap="predictive/{epsilon}/bootstrap/{n_datapoints}.npy",
        dirichlet1="predictive/{epsilon}/dirichlet1/{n_datapoints}.npy",
        adjustable="predictive/{epsilon}/adjustable/{n_datapoints}.npy",
    run:

        def emr_fn(i: int):
            def f(y):
                return y[:, i].mean()
            return f


        population = jnp.asarray(np.load(input.population))
        G = population.shape[1]

        FILES = {
            "bootstrap": input.bootstrap,
            "dirichlet1": input.dirichlet1,
            "parametric": input.parametric,
            "adjustable": input.adjustable,
        }

        fig, axs = plt.subplots(G, len(FILES), sharex=True, sharey=False)
        
        for i in range(G):
            summary = emr_fn(i)
            bins = np.linspace(0, 0.2, 20)
            
            for j, (method, file) in enumerate(FILES.items()):
                ax = axs[i, j]

                ax.axvline(summary(population), linewidth=3, linestyle=":", color="black")
                
                color =  COLORS[method]
                predictive = jnp.asarray(np.load(file))
                ax.hist(
                    jax.vmap(summary)(predictive),
                    bins=bins,
                    density=True,
                    color=color,
                    alpha=0.2,
                    # histtype="step",
                )

        for ax in axs.ravel():
            ax.set_xlim(0, 0.2)
            ax.spines[["top", "left", "right"]].set_visible(False)
            ax.set_yticks([])

        fig.savefig(output[0])


rule plot_performance_zero_mass:
    output: "plots/{epsilon}/{n_datapoints}/zero_mass.pdf"
    input:
        population="population/{epsilon}.npy",
        parametric="predictive/{epsilon}/parametric/{n_datapoints}.npy",
        bootstrap="predictive/{epsilon}/bootstrap/{n_datapoints}.npy",
        dirichlet1="predictive/{epsilon}/dirichlet1/{n_datapoints}.npy",
        adjustable="predictive/{epsilon}/adjustable/{n_datapoints}.npy",
    run:
        def summary(y):
            return jnp.mean(y.sum(axis=-1) < 1)

        population = jnp.asarray(np.load(input.population))
        G = population.shape[1]

        
        FILES = {
            "bootstrap": input.bootstrap,
            "dirichlet1": input.dirichlet1,
            "parametric": input.parametric,
            "adjustable": input.adjustable,
        }

        fig, axs = plt.subplots(1, len(FILES), sharex=True, sharey=False)

        for ax in axs.ravel():
            ax.axvline(summary(population), linewidth=3, linestyle=":", color="black")

        bins = 20  # np.linspace(0, 0.6, 15)
            
        for i, (method, file) in enumerate(FILES.items()):
            ax = axs[i]
            color =  COLORS[method]
            predictive = jnp.asarray(np.load(file))
            ax.hist(
                jax.vmap(summary)(predictive),
                bins=bins,
                density=True,
                color=color,
                alpha=0.2,
                # histtype="step",
            )

        # ax.set_xlim(0, 0.6)
        for ax in axs.ravel():
            ax.spines[["top", "left", "right"]].set_visible(False)
            ax.set_yticks([])

        fig.savefig(output[0])