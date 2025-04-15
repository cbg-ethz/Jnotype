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


def _calculate_mcc(X: _DataSet) -> Float[Array, "n_genes n_genes"]:
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


def calculate_mcc(X):
    """
    Computes the Pearson correlation coefficient matrix for the input array X.

    Parameters:
    X (jnp.ndarray): Input data of shape (n_samples, n_features)

    Returns:
    jnp.ndarray: Correlation matrix of shape (n_features, n_features)
    """
    # Step 1: Center the data (subtract the mean)
    X_mean = jnp.mean(X, axis=0)
    X_centered = X - X_mean

    # Step 2: Compute the covariance matrix
    # Note: Using (n_samples - 1) for an unbiased estimator
    covariance_matrix = jnp.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

    # Step 3: Compute standard deviations
    std_devs = jnp.sqrt(jnp.diag(covariance_matrix))

    # Step 4: Handle zero standard deviations to avoid division by zero
    # Create a mask of non-zero standard deviations
    non_zero_mask = std_devs > 0

    # Compute the outer product of standard deviations
    std_outer = jnp.outer(std_devs, std_devs)

    # To avoid division by zero, set zero std to 1 temporarily (will mask later)
    std_outer_safe = jnp.where(std_outer < 1e-17, 1, std_outer)

    # Step 5: Compute the correlation matrix
    corr_matrix = covariance_matrix / std_outer_safe

    # Set correlations involving constant columns to zero
    # Expand the mask to a 2D mask
    mask_2d = jnp.outer(non_zero_mask, non_zero_mask)
    corr_matrix = jnp.where(mask_2d, corr_matrix, 0.0)

    return corr_matrix


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


def get_leading_axis_size(pytree) -> int:
    """Infers the number of samples in a PyTree."""
    # Extract all leaf nodes from the PyTree
    leaves = jax.tree_util.tree_leaves(pytree)

    if not leaves:
        raise ValueError("The PyTree has no leaves.")

    # TODO(Pawel): Go through all the leaves and check
    #  if shapes agree

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
    leading_size = get_leading_axis_size(samples)

    if n_samples is None:
        n_samples = leading_size

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
    n_samples = get_leading_axis_size(samples)
    keys = jax.random.split(key, n_samples)

    def f(subkey, sample):
        """Simulates a data set and calculates summary statistic."""
        y_sim = simulator_fn(subkey, sample)
        return statistic_fn(y_sim)

    return jax.vmap(f)(keys, samples)
