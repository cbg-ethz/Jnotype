import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from jax import random

from jnotype.pyramids import TwoLayerPyramidSampler
from jnotype.sampling import ListDataset, XArrayChunkedDataset


SEEDS = range(2)
BINARY_CODES: int = 10
CLUSTERS: int = 3
THINNING: int = 2
WARMUP: int = 5_000

N_SAMPLES: int = 1_000

N_STEPS = N_SAMPLES * THINNING


rule all:
    input: expand("generated/TCGA-test/chain-{seed}", seed=SEEDS)


rule run_chain:
    input: "data/TCGA-only-genes-from-Fritz.csv"
    output: directory("generated/TCGA-test/chain-{seed}")
    run:
        seed = int(wildcards.seed)

        observed = pd.read_csv(str(input), index_col=0).values

        dataset = XArrayChunkedDataset(
            directory=str(output),
            basic_dimensions=TwoLayerPyramidSampler.dimensions(),
            attrs={"description": f"Run with seed {seed}."},
            buffer_size=500,
            thinning=THINNING,
        )

        gibbs_sampler = TwoLayerPyramidSampler(
            datasets=[dataset],
            observed=observed,
            dirichlet_prior=jnp.ones(shape=(CLUSTERS)),
            n_binary_codes=BINARY_CODES,
            n_clusters=CLUSTERS,
            warmup=WARMUP,
            steps=N_STEPS,
            verbose=True,
            seed=seed,
            gamma_prior=(100.0, 2000.0),
        )
        gibbs_sampler.run()
