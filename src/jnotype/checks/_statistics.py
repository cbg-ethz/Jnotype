"""Convenient summary statistics,
useful e.g. for posterior predictive checking."""

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


def calculate_number_of_mutations_histogram(X: _DataSet) -> Float[Array, " n_genes+1"]:
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
