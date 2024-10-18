"""Convenient summary statistics,
useful e.g. for posterior predictive checking."""

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
