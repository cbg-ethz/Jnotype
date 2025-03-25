"""Markov chain Monte Carlo sampling of binary vectors from energy-based models."""

from typing import Callable

import jax
import jax.random as jrandom
import jax.numpy as jnp

from jaxtyping import Array, Int, Float


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
    """Applies a Gibbs sampling step for bit at location `idx`."""
    # Calculate the probabilities for both choices of this bit
    y0 = y.at[idx].set(0)
    y1 = y.at[idx].set(1)

    ys = jnp.vstack((y0, y1))
    logits = jax.vmap(log_prob_fn)(ys)

    chosen_index = jrandom.categorical(key, logits=logits)
    return ys[chosen_index]


def _random_site_bitflip(
    key: jax.Array,
    log_prob_fn: _UnnormLogProb,
    y: _DataPoint,
) -> _DataPoint:
    """Samples a single bit in the Ising model."""
    G = y.shape[0]
    # Pick a random index to update
    key1, key2 = jrandom.split(key)
    idx = jrandom.randint(key1, shape=(), minval=0, maxval=G)

    return _gibbs_bitflip(key=key2, log_prob_fn=log_prob_fn, y=y, idx=idx)


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


def _gibbs_blockflip(
    key: jax.Array,
    log_prob_fn: _UnnormLogProb,
    y: _DataPoint,
    sites: Int[Array, " size"],
) -> _DataPoint:
    """Performs a blocked Gibbs update, jointly
    resampling all bits at `sites`.

    Note:
        This requires `2**len(sites)` evaluations
        of the log-probability (and similar memory),
        so that not too many sites should be jointly
        updated.
    """
    # Generate all possible configurations for the block
    block_size = sites.shape[0]
    all_configs = generate_all_binary_vectors(block_size)

    # Generate (unnormalized) log-probs for all possible configurations:
    def logp(config):
        y_candidate = y.at[sites].set(config)
        return log_prob_fn(y_candidate)

    logits = jax.vmap(logp)(all_configs)

    # Select the new configuration
    new_block_idx = jax.random.categorical(key, logits=logits)
    new_config = all_configs[new_block_idx]

    return y.at[sites].set(new_config)


def construct_random_blockflip_kernel(log_prob_fn: _UnnormLogProb, block_size: int):
    """Constructs a kernel resampling a random block of size `block_size`.

    Note:
        One requires `2**block_size` evaluations of the log-probability function
        (and similar memory complexity), so that only relatively small blocks
        can be used.
    """

    def kernel(key, y):
        """Kernel flipping a random bit."""
        G = y.shape[0]

        key1, key2 = jrandom.split(key)
        sites = jrandom.choice(
            key1,
            G,
            shape=(block_size,),
            replace=False,
        )
        return _gibbs_blockflip(
            key=key2,  # type: ignore
            log_prob_fn=log_prob_fn,
            y=y,
            sites=sites,
        )

    return jax.jit(kernel)
