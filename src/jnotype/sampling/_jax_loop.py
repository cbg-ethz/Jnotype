"""JAX Markov chain Monte Carlo sampling loop."""

from typing import Sequence, Callable, NewType, Any

import jax
import jax.numpy as jnp

_Sample = NewType("_Sample", Any)  # sample can be an arbitrary JAX PyTree
_KernelFn = Callable[
    [jax.Array, _Sample], _Sample
]  # A kernel has signature kernel(key, state) -> new_state


def compose_kernels(kernels: Sequence[_KernelFn]) -> _KernelFn:
    """Composes kernels.

    Note:
        If you compose the same kernel mutiple times (effectively doing thinning),
        a more efficient solution is `iterated_kernel_thinning`.
    """

    def new_kernel(key: jax.Array, sample: _Sample) -> _Sample:
        """Iterated kernel."""
        for i, kernel in enumerate(kernels):
            sample = kernel(jax.random.fold_in(key, i), sample)
        return sample

    return new_kernel


def iterated_kernel_thinning(kernel: _KernelFn, thinning: int) -> _KernelFn:
    """Composes the kernel with itself, essentially doing thinning.

    Args:
        kernel: kernel to be composed
        thinning: number of times the `kernel` should be composed with itself.
            For `thinning=1` original `kernel` is returned.

    Raises:
        ValueError: when `thinning` is strictly less than 1.
    """
    if thinning <= 0:
        raise ValueError("Thinning should be at least 1.")
    elif thinning == 1:
        return kernel

    def new_kernel(key: jax.Array, sample: _Sample) -> _Sample:
        """Iterated kernel."""

        def f(s: _Sample, i: int) -> _Sample:
            """Folds-in the iteration number to the key and returns new sample."""
            new_key = jax.random.fold_in(key, i)
            new_sample = kernel(new_key, s)
            return new_sample, None

        new_sample, _ = jax.lax.scan(f, sample, jnp.arange(thinning))
        return new_sample

    return new_kernel


def sampling_loop(
    key: jax.Array,
    kernel: _KernelFn,
    x0: _Sample,
    samples: int = 1_000,
    thinning: int = 1,
    warmup: int = 0,
):
    """Sampling loop.

    Args:
        key: JAX key array
        kernel: kernel to be used
        x0: starting point
        samples: number of samples to be obtained
        thinning: whether thinning should be applied
        warmup: number of warmup steps

    Returns:
        PyTree with a structure same as `x0`, but each
        array has a 0th axis of length `samples`
    """
    # Run warmup
    if warmup > 0:
        kernel_warmup = jax.jit(
            iterated_kernel_thinning(kernel=kernel, thinning=warmup),
        )
        key, subkey = jax.random.split(key)
        x0 = kernel_warmup(subkey, x0)

    # Run sampling loop with thinned kernel
    kernel_thinned = iterated_kernel_thinning(kernel=kernel, thinning=thinning)

    @jax.jit
    def f(s: _Sample, i: int) -> _Sample:
        """Folds-in the iteration number to the key and returns new sample."""
        new_key = jax.random.fold_in(key, i)
        new_sample = kernel_thinned(new_key, s)
        return new_sample, new_sample

    _, samples = jax.lax.scan(f, x0, jnp.arange(samples))
    return samples
