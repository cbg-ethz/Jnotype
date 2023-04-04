"""Bernoulli Mixture Model."""
from jnotype.bmm._em import expectation_maximization
from jnotype.bmm._gibbs import gibbs_sampler

__all__ = [
    "expectation_maximization",
    "gibbs_sampler",
]
