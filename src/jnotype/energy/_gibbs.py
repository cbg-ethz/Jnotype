def make_asymmetric_kernel(
    log_prob_fn, dim_count, m=2, L=0, U=jnp.inf, block_size=1, scan_type="random"
):
    """
    Create an asymmetric proposal kernel for Gibbs sampling

    Args:
        log_prob_fn: Function that computes log probability of a state
        dim_count: Number of dimensions in state space
        m: Window size for proposal (default: 2)
        L: Lower bound for values (default: 0)
        U: Upper bound for values (default: infinity)
        block_size: Size of block of random entries to be resampled at once (default: 1)

    Returns:
        A kernel function that takes (key, x) and returns (new_x, key, val)
    """

    @jax.jit
    def log_acceptance_rate(x, y, idx) -> jnp.ndarray:
        log_prob_term = log_prob_fn(y) - log_prob_fn(x)
        range_y = jnp.minimum(U, y[idx] + m) - jnp.maximum(L, y[idx] - m) + 1
        range_x = jnp.minimum(U, x[idx] + m) - jnp.maximum(L, x[idx] - m) + 1
        log_qyx = -jnp.log(range_y)
        log_qxy = -jnp.log(range_x)
        return log_prob_term + jnp.sum(log_qyx) - jnp.sum(log_qxy)

    @jax.jit
    def sample_idx(key):
        return jax.random.permutation(key, jnp.arange(dim_count), independent=True)[
            :block_size
        ]

    @jax.jit
    def kernel(key, x, iter_idx):
        key, key_idx, key_val, key_accept = jax.random.split(key, 4)

        if scan_type == "ordered":
            idx = jnp.array([iter_idx % dim_count])
        else:
            idx = sample_idx(key_idx)

        min_val = jnp.maximum(L, jnp.maximum(0, x[idx] - m))
        max_val = jnp.minimum(U, x[idx] + m)
        val = jax.random.randint(
            key_val, shape=(idx.shape[0],), minval=min_val, maxval=max_val + 1
        )
        y = x.at[idx].set(val)

        accept = jnp.log(jax.random.uniform(key_accept)) < log_acceptance_rate(
            x, y, idx
        )
        return jnp.where(accept, y, x), key

    return jax.jit(kernel)


def gibbs_sample(
    key,
    initial_sample: jnp.array,
    num_steps: int,
    warmup: int,
    kernel,
) -> jnp.ndarray:
    """
    Perform Gibbs sampling for multiple steps using a provided kernel

    Args:
        key: JAX random key
        initial_sample: Starting point for sampling
        num_steps: Number of Gibbs steps to perform
        warmup: Number of warmup steps to discard
        kernel: MCMC transition kernel function
        scan_type: Type of scan ("random", "ordered", or "ordered_blocks") (default: "random")

    Returns:
        All samples and proposed values
    """
    total_steps = num_steps + warmup
    all_keys = jax.random.split(key, total_steps)
    indices = jnp.arange(total_steps)  # shape: (total_steps,)

    def scan_fn(carry, inputs):
        current_sample = carry
        idx, key_i = inputs
        next_state, _ = kernel(key_i, current_sample, idx)
        return next_state, next_state

    _, all_samples = jax.lax.scan(
        scan_fn,
        initial_sample,
        (indices, all_keys),  # <- tuple of arrays, shape-matched
    )
    return all_samples


def run_gibbs_sampling(
    key, G, model: InferenceModel, warmup: int, num_steps: int, kernel
):
    """Run a single experiment with specified parameters"""
    key, sample_key, gibbs_key = jrandom.split(key, 3)
    # a sequence of {0,1}^G
    initial_sample = jrandom.randint(
        sample_key, shape=(G,), minval=0, maxval=2
    )  # type: ignore

    return gibbs_sample(
        gibbs_key,
        initial_sample,
        warmup=warmup,
        num_steps=num_steps,
        kernel=kernel,
    )
