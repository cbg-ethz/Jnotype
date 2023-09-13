import nimfa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jnotype.bmm import BernoulliMixtureGibbsSampler
from jnotype.pyramids import TwoLayerPyramidSampler
from jnotype.sampling import ListDataset

import _benchmark_utils as utils

METHODS = [
    "bmf",
    "pyramids-unsupervised",
    "pyramids-labeled",
    "bmm",
]

DATASETS = {
    "250": 250,
    "500": 500,
    "1000": 1000,
    "2000": 2000,
}
N_SEEDS: int = 10

COMPONENTS = [2, 4, 8]

N_WARMUP: int = 3000
N_STEPS: int = 1500


workdir: "generated/benchmark/"

rule all:
    input: expand("{method}-{components}/{dataset}/{seed}.npz", method=METHODS, components=COMPONENTS, dataset=DATASETS, seed=range(N_SEEDS))

rule assemble_results_mi:
    input: expand("{method}-{components}/{dataset}/{seed}.npz", method=METHODS, components=COMPONENTS, dataset=DATASETS, seed=range(N_SEEDS))
    output: "results_mi.csv"
    run:
        results = []
        for method in METHODS:
            for components in COMPONENTS:
                for dataset in DATASETS:
                    for seed in range(N_SEEDS):
                        arrays = np.load(f"{method}-{components}/{dataset}/{seed}.npz")
                        results.append(
                            {
                                "method": method,
                                "components": components,
                                "dataset": dataset,
                                "seed": seed,
                                "mutual_information_gap": arrays["mutual_information_gap"],
                            }
                        )
        pd.DataFrame(results).to_csv(str(output), index=False)

rule assemble_results_mse:
    input: expand("pyramids-labeled-{components}/{dataset}/{seed}.npz", components=COMPONENTS, dataset=DATASETS, seed=range(N_SEEDS))
    output: "results_mse.csv"
    run:
        results = []
        for components in COMPONENTS:
            for dataset in DATASETS:
                for seed in range(N_SEEDS):
                    arrays = np.load(f"pyramids-labeled-{components}/{dataset}/{seed}.npz")
                    results.append(
                        {
                            "components": components,
                            "dataset": dataset,
                            "seed": seed,
                            "coefficients_X_mse": arrays["coefficients_X_mse"],
                        }
                    )
        pd.DataFrame(results).to_csv(str(output), index=False)

rule plot_results_mse:
    input: "results_mse.csv"
    output: "plot_mse.pdf"
    run:
        df = pd.read_csv(str(input))
        df["Latent traits"] = df["components"]

        palette = {
            2: "dodgerblue",    
            4: "blue",
            8: "navy",
        }

        fig, ax = plt.subplots(figsize=(3, 3), dpi=250)
        sns.boxplot(data=df, hue="Latent traits", x="dataset", y="coefficients_X_mse", ax=ax, palette=palette)
        ax.set_xlabel("Number of patients")
        ax.set_ylabel("Mean squared error")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(str(output))


rule plot_results_mi:
    input: "results_mi.csv"
    output: "plot_mi.pdf"
    run:
        def rename(method: str, components: int) -> str:
            if method == "bmf":
                return f"BMF ({components})"
            elif method == "pyramids-unsupervised":
                return f"Pyramids (unsupervised, {components})"
            elif method == "pyramids-labeled":
                return f"Pyramids (labeled, {components})"
            elif method == "bmm":
                return f"BMM ({components})"
            else:
                raise ValueError(f"{method} not recognized")

        df = pd.read_csv(str(input))
        df["algorithm"] = df.apply(lambda row: rename(row["method"], row["components"]), axis=1)

        color_list = [
            "lightgray",
            "darkgray",
            "dimgray",
            #
            "salmon",
            "red",
            "firebrick",
            #
            "dodgerblue",    
            "blue",
            "navy",
            #
            "lawngreen",
            "limegreen",
            "forestgreen",
        ]
        palette = dict(zip(df["algorithm"].unique(), color_list))

        fig, ax = plt.subplots(figsize=(7, 3), dpi=250)
        sns.boxplot(data=df, x="dataset", y="mutual_information_gap", hue="algorithm", palette = palette, ax=ax)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        ax.set_ylabel("Mutual information gap")
        ax.set_xlabel("Number of patients")

        fig.tight_layout()
        fig.savefig(str(output))


rule generate_ground_truth:
    output: "ground_truth/{dataset}/{seed}.npz"
    run:
        n_points = DATASETS[wildcards.dataset]
        seed = int(wildcards.seed)
        dataset = utils.generate_dataset(n_points=n_points, seed=seed)
        np.savez(str(output), **dataset)


rule run_bmf:
    input: "ground_truth/{dataset}/{seed}.npz"
    output: "bmf-{components}/{dataset}/{seed}.npz"
    run:
        arrays = np.load(str(input))
        Y = arrays["Y"]
        n_components = int(wildcards.components)
        bmf = nimfa.Bmf(
            Y.T,
            seed="nndsvd",
            rank=n_components,
            max_iter=100,
            lambda_w=1.1,
            lambda_h=1.1,
        )
        bmf = bmf()

        features_continuous = np.asarray(bmf.coef()).T
        features_discrete = features_continuous > 0.5

        mi = utils.mutual_information(features_discrete, arrays["A"])
        np.savez(
            str(output),
            features_continuous=features_continuous,
            features_discrete=features_discrete,
            n_components=np.array(n_components),
            mutual_information=np.array(mi),
            mutual_information_gap=arrays["mutual_information"] - mi,
        )


rule run_unsupervised_pyramids:
    input: "ground_truth/{dataset}/{seed}.npz"
    output: "pyramids-unsupervised-{components}/{dataset}/{seed}.npz"
    run:
        arrays = np.load(str(input))
        Y = arrays["Y"]
        n_components = int(wildcards.components)
        
        dataset = ListDataset(thinning=5, dimensions=TwoLayerPyramidSampler.dimensions())
        sampler = TwoLayerPyramidSampler(
            datasets=[dataset],
            observed=Y,
            n_binary_codes=n_components,
            n_clusters=2,
            dirichlet_prior=np.ones(2) / 2,
            warmup=N_WARMUP,
            steps=N_STEPS,
            verbose=False,
        )
        sampler.run()
        
        latent_traits = (dataset.dataset["latent_traits"].mean(axis=0) > 0.5).values
        mi = utils.mutual_information(latent_traits, arrays["A"])
        
        np.savez(
            str(output),
            latent_traits=latent_traits,
            n_components=np.array(n_components),
            mutual_information=np.array(mi),
            mutual_information_gap=arrays["mutual_information"] - mi,
        )


rule run_labeled_pyramids:
    input: "ground_truth/{dataset}/{seed}.npz"
    output: "pyramids-labeled-{components}/{dataset}/{seed}.npz"
    run:
        arrays = np.load(str(input))
        Y = arrays["Y"]
        X = arrays["X"]
        n_components = int(wildcards.components)

        dataset = ListDataset(thinning=5, dimensions=TwoLayerPyramidSampler.dimensions())

        sampler = TwoLayerPyramidSampler(
            datasets=[dataset],
            observed=Y,
            observed_covariates=X,
            n_binary_codes=n_components,
            n_clusters=2,
            dirichlet_prior=np.ones(2) / 2,
            warmup=N_WARMUP,
            steps=N_STEPS,
            verbose=False,
        )
        sampler.run()
        
        latent_traits = (dataset.dataset["latent_traits"].mean(axis=0) > 0.5).values
        mi = utils.mutual_information(latent_traits, arrays["A"])

        coefs_inferred = dataset.dataset["coefficients_observed"].mean(axis=0).values
        coefs_true = arrays["coefficients_X"]
        coefs_mse = np.mean((coefs_inferred - coefs_true) ** 2)

        np.savez(
            str(output),
            latent_traits=latent_traits,
            n_components=np.array(n_components),
            mutual_information=np.array(mi),
            mutual_information_gap=arrays["mutual_information"] - mi,
            coefficients_X=coefs_inferred,
            coefficients_X_mse=coefs_mse,
        )


rule run_bernoulli_mixture_model:
    input: "ground_truth/{dataset}/{seed}.npz"
    output: "bmm-{components}/{dataset}/{seed}.npz"
    run:
        arrays = np.load(str(input))
        Y = arrays["Y"]
        n_components = int(wildcards.components)
        
        dataset = ListDataset(dimensions=BernoulliMixtureGibbsSampler.dimensions(), thinning=5)
        sampler = BernoulliMixtureGibbsSampler(
            datasets=[dataset],
            observed_data=Y,
            dirichlet_prior=np.ones(n_components),
            warmup=N_WARMUP,
            steps=N_STEPS,
        )
        sampler.run()
        
        posterior_samples = dataset.dataset["labels"].values
        labels = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=posterior_samples)

        mi = utils.mutual_information(labels[:, None], arrays["A"])
        
        np.savez(
            str(output),
            posterior_samples=posterior_samples,
            labels=labels,
            n_components=np.array(n_components),
            mutual_information=np.array(mi),
            mutual_information_gap=arrays["mutual_information"] - mi,
        )
