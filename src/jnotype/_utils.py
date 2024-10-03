"""Generic utilities.

This file should be as small as possible.
Appearing themes should be refactored and placed
into separate modules."""

import jax
import numpy as np


class JAXRNG:
    """JAX stateful random number generator.

    Example:
      key = jax.random.PRNGKey(5)
      rng = JAXRNG(key)
      a = jax.random.bernoulli(rng.key, shape=(10,))
      b = jax.random.bernoulli(rng.key, shape=(10,))
    """

    def __init__(self, key: jax.Array) -> None:
        """
        Args:
            key: initialization key
        """
        self._key = key

    @property
    def key(self) -> jax.Array:
        """Generates a new key."""
        key, subkey = jax.random.split(self._key)
        self._key = key
        return subkey

    def __repr__(self) -> str:
        """Used by the repr() method."""
        return f"{type(self).__name__}(key={self._key})"

    def __str__(self) -> str:
        """Used by the str() method."""
        return repr(self)


def order_genotypes(mutations, reverse: bool = False):
    """Finds lexicographic order on the provided genotypes,
    e.g., for plotting purposes.

    Args:
        mutations: genotypes, (n_samples, n_genes)
        reverse: whether to swap the order or not

    Returns:
        array (n_samples,) representing the indices of the order
    """
    ord = np.argsort(list(map(lambda x: "".join(map(str, x)), mutations)))
    if reverse:
        ord = ord[::-1]
    return ord
