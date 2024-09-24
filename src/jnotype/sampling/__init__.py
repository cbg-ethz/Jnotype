"""Generic utilities for sampling."""

from jnotype.sampling._chunker import (
    DatasetInterface,
    ListDataset,
    XArrayChunkedDataset,
    AbstractChunkedDataset,
)
from jnotype.sampling._sampler import AbstractGibbsSampler

from jnotype.sampling._jax_loop import (
    sampling_loop,
    compose_kernels,
    iterated_kernel_thinning,
)

__all__ = [
    "DatasetInterface",
    "ListDataset",
    "XArrayChunkedDataset",
    "AbstractChunkedDataset",
    "AbstractGibbsSampler",
    "sampling_loop",
    "compose_kernels",
    "iterated_kernel_thinning",
]
