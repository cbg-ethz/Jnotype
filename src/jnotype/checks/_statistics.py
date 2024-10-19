"""Convenient summary statistics,
useful e.g. for posterior predictive checking."""

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Float

_DataSet = Int[Array, "n_samples n_genes"]


def calculate_mutation_frequencies(X: _DataSet) -> Float[Array, " n_genes"]:
    """Calculates the mutation frequency for each loci.

    Returns:
        freq, of shape (n_genes,) with the observed mutation
        frequency at each loci
    """
    return jnp.mean(X, axis=0)


def calculate_number_of_mutations_histogram(X: _DataSet) -> Int[Array, " n_genes+1"]:
    """Creates an array counting the samples with a specific
    number of mutations.

    Returns:
        array of shape (n_genes+1,) with `arr[n]` equal to the number
          of samples with exactly `n` mutations
    """
    n_genes = X.shape[1]
    counts = jnp.sum(X, axis=1)
    return jnp.bincount(counts, length=n_genes + 1)


def calculate_mcc(X: _DataSet) -> Float[Array, "n_genes n_genes"]:
    """Calculates the Matthews correlation coefficient
    of the observed sample.

    Returns:
        array of shape (n_genes, n_genes) with `arr[i, j]` being
          the MCC between loci `i` and `j`.

    Note:
        1. The returned matrix is symmetric and
           should have `1.0` on the diagonal
        2. The MCC is not defined when the column is constant
           (i.e., the mutation frequency is 0 or 1)
    """
    return jnp.corrcoef(X, rowvar=False)


def _convert_binary_code_to_integer(x: Int[Array, " n_genes"]) -> Int[Array, " "]:
    """A binary genotype represented as an array
    is converted into the integer with corresponding
    binary representation.
    """
    n_genes = x.shape[0]
    bit_positions = jnp.arange(n_genes - 1, -1, -1)
    powers = jnp.power(2, bit_positions)
    return jnp.sum(powers * x)


def convert_genotypes_to_integers(X: _DataSet) -> Int[Array, " n_samples"]:
    """Each binary genotype is converted into the
    integer with corresponding representation in
    the binary numeral system.
    """
    return jax.vmap(_convert_binary_code_to_integer)(X)


def convert_integers_to_genotypes(
    integers: Int[Array, " n_samples"], n_genes: int
) -> _DataSet:
    """Maps each integer to a binary genotype.

    Args:
        Y (jnp.ndarray): Array of integers with shape (n_samples,).
        n_genes (int): Number of genes (bits) in the genotype.

    Returns:
        jnp.ndarray: Binary genotype array with shape (n_samples, n_genes).
    """
    # Create an array of bit positions
    bit_positions = jnp.arange(n_genes - 1, -1, -1)
    # Right-shift and mask to get the bits
    genotypes = (integers[:, None] >> bit_positions) & 1
    return genotypes


def calculate_atoms_occurrence(X: _DataSet) -> Int[Array, " 2**n_genes"]:
    """For each unique genotype counts the number of matching samples.

    Note:
        The returned array has exponentially large length, so that this
        function should be avoided for large gene sets.
    """
    indices = convert_genotypes_to_integers(X)

    n_genes = X.shape[1]
    length = jnp.power(2, n_genes)
    return jnp.bincount(indices, length=length)  # type: ignore


def _get_leading_axis_size(pytree):
    # Extract all leaf nodes from the PyTree
    leaves = jax.tree_util.tree_leaves(pytree)

    if not leaves:
        raise ValueError("The PyTree has no leaves.")

    # Assume the first leaf contains the leading axis
    first_leaf = leaves[0]

    # Ensure the leaf has a shape attribute
    if hasattr(first_leaf, "shape") and len(first_leaf.shape) > 0:
        return first_leaf.shape[0]
    else:
        raise ValueError("The first leaf does not have a valid shape.")


def subsample_pytree(
    key: jax.Array,
    samples,
    n_samples: Optional[int] = None,
):
    """Subsamples a PyTree along the leading axis."""
    leading_size = _get_leading_axis_size(samples)

    if n_samples > leading_size:
        raise ValueError("n_samples cannot be larger than the leading axis size.")

    # Generate a permutation of indices and select the first n_samples
    perm = jax.random.permutation(key, leading_size)
    selected_indices = perm[:n_samples]

    def index_leaves(x):
        """Function indexing each leaf"""
        return x[selected_indices]

    # Apply the indexing function to all leaves
    subsampled_pytree = jax.tree_util.tree_map(index_leaves, samples)

    return subsampled_pytree


def simulate_summary_statistic(
    key: jax.Array,
    simulator_fn,
    statistic_fn,
    samples,
):
    """Simulates the summary statistics.

    Args:
        key: JAX random key
        simulator_fn: function with the signature
            (RandomKey, Sample) -> DataSet
        statistic_fn: function with the signature
            Sample -> Statistic
        samples: a PyTree with structure `Sample`,
            which has a leading (0th) axis in each leaf
            corresponding to the samples from the distribution
    """
    n_samples = _get_leading_axis_size(samples)
    keys = jax.random.split(key, n_samples)

    def f(subkey, sample):
        """Simulates a data set and calculates summary statistic."""
        y_sim = simulator_fn(subkey, sample)
        return statistic_fn(y_sim)

    return jax.vmap(f)(keys, samples)
