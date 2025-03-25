import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, Int

from _prior import (
    create_symmetric_interaction_matrix,
    number_of_interactions_quadratic,
)

from _sampling import (
    _DataPoint,
    _UnnormLogProb,
)


def unnorm_log_prob_ising_model(
    interaction_matrix: Int[Array, "G G"],
) -> _UnnormLogProb:
    """Returns the negative energy function for an Ising model.

    Returns:
        Callable:
            A function mapping a binary vector of shape (G,) to a float.
    """

    def energy(y: _DataPoint) -> Float:
        """Calculates the energy of a binary vector `y`."""
        return jnp.dot(y, jnp.dot(interaction_matrix, y))

    return lambda y: -energy(y)


# -------------------------
# Helper function
# -------------------------
def sample_ising_model_matrix(rng_key, G, p_zero, scale):
    """Samples an interaction matrix for an Ising model with a spike-and-slab prior.

    Args:
      rng_key: jrandom.PRNGKey
        Key for random number generation in JAX.
      G: int
        Length of the input sequence.
      p_zero: float
        Probability that off-diagonal entries are zero ("spike").
      scale: float
        Standard deviation for the slab normal distribution.

    Returns:
      jnp.ndarray:
        A (G x G) matrix where the diagonal is drawn from Normal(0, scale^2),
        and the off-diagonal entries are zero with probability p_zero; otherwise
        drawn from Normal(0, scale^2).
    """
    key_off_diag_mask, key_off_diag_norm, key_diag = jrandom.split(rng_key, 3)

    diag_vals = scale * jrandom.normal(key_diag, shape=(G,))

    # Number of off-diagonal interactions
    num_offdiag = number_of_interactions_quadratic(G)
    mask = jrandom.bernoulli(key_off_diag_mask, p=1.0 - p_zero, shape=(num_offdiag,))
    normal_values = scale * jrandom.normal(key_off_diag_norm, shape=(num_offdiag,))
    off_diag_vals = jnp.array(jnp.where(mask, normal_values, 0.0))

    mat = create_symmetric_interaction_matrix(diag_vals, off_diag_vals)

    return mat


def gibbs_sample(
    key: jax.Array,
    kernel,
    initial_sample: jnp.ndarray,
    num_steps: int = 1000,
    warmup: int = 500,
) -> jnp.ndarray:
    """
    Perform Gibbs sampling for multiple steps using a provided kernel.

    Args:
        key: JAX random key
        kernel: MCMC transition kernel function (from construct_*_kernel)
        initial_sample: Starting point for sampling (defaults to zeros if None)
        num_steps: Number of Gibbs steps to return after warmup
        warmup: Number of initial samples to discard

    Returns:
        Array of samples after warmup with shape (num_steps, G)
    """
    total_steps = num_steps + warmup
    step_keys = jrandom.split(key, total_steps)

    def scan_fn(carry, key_i):
        current_sample = carry
        next_sample = kernel(key_i, current_sample)
        return next_sample, next_sample

    _, all_samples = jax.lax.scan(scan_fn, initial_sample, step_keys)

    return all_samples[warmup:]
