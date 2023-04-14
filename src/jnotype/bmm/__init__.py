"""Bernoulli Mixture Model."""
from jnotype.bmm._em import expectation_maximization
from jnotype.bmm._gibbs import (
    gibbs_sampler,
    sample_mixing,
    sample_cluster_labels,
    sample_cluster_proportions,
)
from jnotype.bmm._gibbs import single_sampling_step as sample_bmm

__all__ = [
    "expectation_maximization",
    "gibbs_sampler",
    "sample_mixing",
    "sample_cluster_labels",
    "sample_cluster_proportions",
    "sample_bmm",
]
