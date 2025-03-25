import jax
import jax.numpy as jnp
import jax.random as jrandom

from _prior import (
    create_symmetric_interaction_matrix,
    number_of_interactions_quadratic,
)


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


# def (rng_key, interaction_matrix, N=300):
#     """Generate samples from an Ising model via a simple Metropolis approach.

#     Args:
#       rng_key: jrandom.PRNGKey
#         Random key for reproducible sampling.
#       interaction_matrix: jnp.ndarray
#         A (G x G) matrix of interaction weights. The diagonal is effectively local field
#         terms, and the off-diagonal are pairwise interactions.
#       N: int
#         Number of samples to generate.

#     Returns:
#       jnp.ndarray:
#         Binary array of shape (N, G) containing Ising configurations.
#     """

#     # We interpret the diagonal as local fields, i.e., mat[i,i] is bias for spin i
#     # The off-diagonal mat[i,j] is the interaction between spin i and spin j.
#     # We'll do a straightforward single-site Metropolis.

#     G = interaction_matrix.shape[0]
#     # initialize state
#     key_init, key_sample = jrandom.split(rng_key)
#     state = jrandom.bernoulli(key_init, p=0.5, shape=(G,)).astype(jnp.int32)

#     def energy(config):
#         # E(config) = - sum_i mat[i,i]*config[i] - sum_{i<j} mat[i,j]*config[i]*config[j]
#         # We'll just do full sum_i,j but we can fix the double counting if needed
#         return -0.5 * jnp.sum(interaction_matrix * config * config[:, None])

#     @jax.jit
#     def metropolis_step(key, current):
#         # choose spin to flip
#         key_idx, key_flip = jrandom.split(key)
#         idx = jrandom.randint(key_idx, shape=(), minval=0, maxval=G)

#         # propose a flip
#         proposal = current.at[idx].set(1 - current[idx])
#         dE = energy(proposal) - energy(current)
#         accept_prob = jnp.exp(-dE)

#         # accept?
#         do_accept = jrandom.uniform(key_flip) < accept_prob
#         new_state = jnp.where(do_accept, proposal, current)
#         return new_state

#     # We'll do a certain number of burnin steps, then sample N times
#     burnin = 500
#     total_steps = burnin + N
#     keys = jrandom.split(key_sample, total_steps)

#     def scan_fn(carry, this_key):
#         current_state = carry
#         new_state = metropolis_step(this_key, current_state)
#         return new_state, new_state

#     final_state, chain = jax.lax.scan(scan_fn, state, keys)
#     # The last N states of chain are the samples, ignoring the burnin
#     return chain[burnin:]
