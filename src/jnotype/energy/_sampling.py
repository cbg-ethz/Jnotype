"""Markov chain Monte Carlo sampling of binary vectors from generalised Ising models."""

from typing import Callable

import jax
import jax.random as jrandom
import jax.numpy as jnp

from jaxtyping import Array, Int


def generate_all_binary_vectors(G: int) -> Int[Array, "2**G G"]:
    """Generates an array of shape (2**G, G) with all binary vectors of length G."""
    return jnp.array(jnp.indices((2,) * G).reshape(G, -1).T)


_DataPoint = Int[Array, " G"]
_UnnormLogProb = Callable[
    [_DataPoint], float
]  # Unnormalised log-probability function maps a binary vector of shape (G,) to a float


def categorical_exact_sampling(
    key: jax.Array,
    n_samples: int,
    G: int,
    log_prob_fn: _UnnormLogProb,
) -> Int[Array, "n_samples G"]:
    """Samples from an energy-based model by constructing all
    `2**G` atoms of the categorical distribution.

    Note:
        G has to be small (e.g., 10), so that a memory overflow does not happen
    """
    binary_vectors = generate_all_binary_vectors(G)

    log_probs = jax.vmap(log_prob_fn)(binary_vectors)  # Unnormalized log-probabilities

    categorical_dist = jrandom.categorical(key, log_probs, shape=(n_samples,))
    return binary_vectors[categorical_dist]


def _gibbs_bitflip(
    key: jax.Array,
    log_prob_fn: _UnnormLogProb,
    y: _DataPoint,
    idx: int,
) -> _DataPoint:
    """Applies a Gibbs sampling step for bit at `idx`."""
    # Calculate the probabilities for both choices of this bit
    y0 = y.at[idx].set(0)
    y1 = y.at[idx].set(1)

    ys = jnp.vstack((y0, y1))
    logits = jax.vmap(log_prob_fn)(ys)

    idx = jrandom.categorical(key, logits=logits)
    return ys[idx]


def _random_site_bitflip(
    key: jax.Array,
    log_prob_fn: _UnnormLogProb,
    y: _DataPoint,
) -> _DataPoint:
    """Samples a single bit in the Ising model."""
    G = y.shape[0]
    # Pick a random index to update
    key, subkey = jrandom.split(key)
    idx = jrandom.randint(subkey, shape=(), minval=0, maxval=G)

    return _gibbs_bitflip(key=key, log_prob_fn=log_prob_fn, y=y, idx=idx)


def construct_random_bitfip_kernel(log_prob_fn: _UnnormLogProb):
    """Constructs a kernel resampling a random site."""

    def kernel(key, y):
        """Kernel flipping a random bit."""
        return _random_site_bitflip(log_prob_fn=log_prob_fn, key=key, y=y)

    return jax.jit(kernel)


def construct_systematic_bitflip_kernel(log_prob_fn: _UnnormLogProb):
    """Constructs a kernel systematically resampling bits one-after-another."""

    def kernel(key, y):
        """Kernel systematically flipping all bits one-after-another."""

        def f(state, idx: int):
            """Auxiliary function performing Gibbs bitflip at a specified site with
            folding-in the site into the key."""
            subkey = jrandom.fold_in(key, idx)
            new_state = _gibbs_bitflip(
                key=subkey, log_prob_fn=log_prob_fn, y=state, idx=idx
            )
            return new_state, None

        new_state, _ = jax.lax.scan(f, y, jnp.arange(y.shape[0]))
        return new_state

    return kernel
