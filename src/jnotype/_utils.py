"""Generic utilities.

This file should be as small as possible.
Appearing themes should be refactored and placed
into separate modules."""

import jax


class JAXRNG:
    """JAX stateful random number generator.

    Example:
      key = jax.random.PRNGKey(5)
      rng = JAXRNG(key)
      a = jax.random.bernoulli(rng.key, shape=(10,))
      b = jax.random.bernoulli(rng.key, shape=(10,))
    """

    def __init__(self, key: jax.random.PRNGKeyArray) -> None:
        """
        Args:
            key: initialization key
        """
        self._key = key

    @property
    def key(self) -> jax.random.PRNGKeyArray:
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
