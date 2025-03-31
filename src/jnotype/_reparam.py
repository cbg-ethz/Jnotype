"""Reparameterizations of models defined on binary arrays."""

from typing import Callable, NamedTuple, TypeVar, Sequence
from collections import Counter
import jax
import jax.numpy as jnp
from jaxtyping import Int, Array, Float

PosNegArray = Int[Array, "N G"]
BinaryArray = Int[Array, "N G"]


def array_to_01(a: PosNegArray, dtype=int) -> BinaryArray:
    """Converts an array from the +/- format to the 0/1 format.
    Parameterization: 0 corresponds to +1. 1 corresponds to -1

    Note:
        0 corresponds to +1
        1 corresponds to -1

    Note:
        This is the inverse to `array_to_pos_neg`.
    """
    return jnp.array(jnp.where(a < 0, 1, 0), dtype=dtype)


def array_to_pos_neg(a: BinaryArray) -> PosNegArray:
    """Converts an array from the 0/1 format to the +/- format.
    Parameterization: 0 corresponds to +1. 1 corresponds to -1

    Note:
        This is the inverse to `array_to_01`.
    """

    return jnp.where(a, -1, 1)


_T = TypeVar("_T")


def func_to_01(f: Callable[[PosNegArray], _T]) -> Callable[[BinaryArray], _T]:
    """Transforms a function taking the inputs arrays in +/- format
    into an equivalent function taking as the inputs the arrays in 0/1 format.

    Note:
        The inverse to `func_to_pos_neg`
    """

    def g(x):
        """Composed function."""
        return f(array_to_01(x))

    return g


def func_to_pos_neg(f: Callable[[BinaryArray], _T]) -> Callable[[PosNegArray], _T]:
    """Transforms a function taking the inputs arrays in 0/1 format
    into an equivalent function taking as the inputs the arrays in +/- format.

    Note:
        The inverse to `func_to_01`
    """

    def g(y):
        """Composed function."""
        return f(array_to_pos_neg(y))

    return g


_Function = Callable[[Int[Array, " G"]], Float[Array, " "]]


class EmpiricalBinaryVectorDistribution(NamedTuple):
    """Stores an empirical distribution on binary vectors,
    making it more efficient to calculate sums and means,
    such as the loglikelihood.

    Note:
        This object should not be created directly.
        Use `empirical_binary_vector_distribution`.
    """

    atoms: Int[Array, "K G"]
    weights: Float[Array, " K"]
    counts: Int[Array, " K"]
    n_atoms: int
    n_datapoints: int

    def calculate_function_mean(self, f: _Function) -> Float[Array, " "]:
        """Calculates the expected value E[f] over the empirical
        distribution."""
        values = jax.vmap(f)(self.atoms)
        return jnp.dot(values, self.weights)

    def calculate_function_sum(self, f: _Function) -> Float[Array, " "]:
        """Calculates the weighted sum of `f` over all points in the distribution,
        taking into account the multiplicities."""
        values = jax.vmap(f)(self.atoms)
        return jnp.dot(values, self.counts)


def empirical_binary_vector_distribution(
    vectors: Sequence[BinaryArray],
) -> EmpiricalBinaryVectorDistribution:
    """The factory method for `EmpiricalBinaryVectorDistribution`,
    which wraps a collection of vectors into a distribution.

    Raises:
        ValueError: if any of the vectors is not 0/1 vector.
    """
    n_total = len(vectors)

    for vector in vectors:
        if jnp.min(vector) < 0 or jnp.max(vector) > 1:
            raise ValueError("All entries must be from the set {0, 1}.")

    cnt = Counter([tuple(vector.tolist()) for vector in vectors])

    atoms = []
    counts = []
    for atom, n_occur in cnt.most_common(None):
        atoms.append(atom)
        counts.append(n_occur)

    atoms = jnp.array(atoms, dtype=int)
    counts = jnp.array(counts, dtype=int)

    return EmpiricalBinaryVectorDistribution(
        atoms=atoms,
        counts=counts,
        weights=jnp.array(counts, dtype=float) / n_total,
        n_atoms=len(cnt),
        n_datapoints=n_total,
    )
