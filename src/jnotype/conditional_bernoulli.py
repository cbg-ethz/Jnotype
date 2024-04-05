"""Conditional Bernoulli distribution."""

import jax
import jax.numpy as jnp
import numpy as np


from jaxtyping import Float, Int, Array


def _calculate_logZ_vector(
    logZ_prev: Float[Array, " G"],
    log_theta: Float[Array, " G"],
    n: int,
) -> Float[Array, " G"]:
    neg_inf = jnp.finfo(log_theta.dtype).min
    G = logZ_prev.shape[0]

    prev = jnp.where(jnp.arange(G) < n - 1, neg_inf, jnp.roll(logZ_prev, shift=1))
    return jax.lax.cumlogsumexp(prev + log_theta)


@jax.jit
def _calculate_logZ(
    log_theta: Float[Array, " G"],
    n_max_shape: Float[Array, " n_max"],
) -> Float[Array, " n_max+1"]:
    """Calculates (log-)normalizing constants for all requested sample sizes.

    Args:
        log_theta: vector with log-probabilities of the Bernoulli model
        n_max_shape: to calculate the log-normalizing constants `log Z(k, theta)`
          for `k = 0, ..., n_max`, use `jnp.arange(n_max)`

    Returns:
        Vector of log-normalizing constants, shape (n_max+1,)
    """
    G = log_theta.shape[0]
    n_max = n_max_shape.shape[0]

    def f(logZprev: Float[Array, " G"], n: int) -> Float[Array, " G"]:
        new_vec = _calculate_logZ_vector(logZprev, log_theta=log_theta, n=n)
        return new_vec, new_vec[-1]

    logZ0 = jnp.zeros(G, dtype=log_theta.dtype)
    _, logZ = jax.lax.scan(f, logZ0, xs=jnp.arange(1, n_max + 1))
    return jnp.concatenate([logZ0[:1], logZ])


def generate_loglikelihood(ys: Int[Array, "N G"], n_max: int = None):
    """Generates the loglikelihood function.

    Args:
        ys: observed binary data, shape (n_samples, n_features)

    Returns:
        function mapping the `log_theta` vector of shape (G,) to the loglikelihood
    """
    ns = jnp.asarray(ys.sum(axis=-1), dtype=int)

    if n_max is None:
        n_max = ns.max()

    weights = jnp.bincount(ns, length=n_max + 1)

    @jax.jit
    def loglikelihood(log_theta: Float[Array, " G"]) -> float:
        """The loglikelihood function."""
        # Loglikelihood without normalizing constants
        ll = jnp.sum(ys * log_theta[None, ...])
        # Normalizing constants
        logZ = _calculate_logZ(log_theta, n_max_shape=jnp.arange(n_max))
        # Calculate the total loglikelihood
        return ll - jnp.sum(logZ * weights)

    return loglikelihood


@jax.jit
def _calculate_logZ_all(
    log_theta: Float[Array, " G"],
    n_max_shape: Float[Array, " n_max"],
) -> Float[Array, "n_max+1 G"]:
    """Calculates all the normalizing constants.

    Note that for likelihood computation `_calculate_logZ`
    is the preferred choice, while this one is used for
    sampling purposes.
    """
    G = log_theta.shape[0]
    n_max = n_max_shape.shape[0]

    def f(logZprev: Float[Array, " G"], n: int) -> Float[Array, " G"]:
        new_vec = _calculate_logZ_vector(logZprev, log_theta=log_theta, n=n)
        return new_vec, new_vec

    logZ0 = jnp.zeros(G, dtype=log_theta.dtype)
    _, logZ = jax.lax.scan(f, logZ0, xs=jnp.arange(1, n_max + 1))
    return jnp.concatenate([logZ0[None, :], logZ])


def _logR(n: int, y: Int[Array, " G"], log_theta: Float[Array, " G"]):
    """Calculates `log R(n, Y)`
    where subset `Y` is represented by binary indicators.

    TODO(Pawel): Move this function to unit tests.
    """
    size = y.sum()
    if n == 0:
        return 0  # = log 1
    elif n > size:
        return -jnp.finfo(log_theta.dtype).min
    else:
        log_theta_ = log_theta[y == 1]
        return _calculate_logZ(log_theta_, n_max_shape=jnp.arange(n))[-1]


def sample_conditional_bernoulli(
    key: jax.Array,
    ns: list[int],
    log_theta: Float[Array, " G"],
) -> Int[Array, "N G"]:
    """Samples from the conditional Bernoulli
    distribution using the ID-checking sampling (Procedure 3)
    in
      S.X. Chen and J.S. Liu,
      "Statistical applications of the Poisson-Binomial
      and conditional Bernoulli distributions",
      Statistica Sinica (1997)

    Args:
        key: JAX random key
        ns: sample sizes, length `N`
        log_theta: log-probabilities

    Returns:
        binary data of shape `(N, G)`
    """
    G = log_theta.shape[0]
    ns = jnp.asarray(ns, dtype=int)
    if jnp.max(ns) > G:
        raise ValueError(f"We require n={jnp.max(ns)} less or equal to G={G}.")

    # Now the trick: we can cache the normalizing constants for all the samples.
    # Note that we reverse `log_theta`, as the algorithm works using complements
    # Hence, we have
    # log_Rs[n, k] = log R(n, {G-k, ..., G}).
    # For example, k = 0 corresponds to {G} and k = G-1 corresponds to {1, ..., G}
    log_Rs = _calculate_logZ_all(
        log_theta=log_theta[::-1], n_max_shape=jnp.arange(jnp.max(ns))
    )

    def get_logR(n: int, size: int) -> float:
        """Wrapper around `log_Rs` matrix."""
        if n == 0:
            return 0  # = log 1
        elif n > size:
            return jnp.finfo(log_theta.dtype).min
        else:
            k = size - 1
            if k < 0 or k >= G:
                raise KeyError(f"Requested k={k} for n={n} and size={size}.")
            else:
                return log_Rs[n, k]

    # TODO(Pawel): This can be refactored into pure JAX
    samples = []
    for key, n in zip(jax.random.split(key, len(ns)), ns):
        sample = np.zeros(G, dtype=int)
        r = 0

        if n == 0:
            samples.append(np.zeros(G, dtype=int))
            continue
        elif n == G:
            samples.append(np.ones(G, dtype=int))

        for k, subkey in enumerate(jax.random.split(key, G)):
            # We want to have G - (k+1) numbers in the set
            logRnum = get_logR(n - r - 1, G - (k + 1))

            # Now we want G - k numbers in the set
            logRden = get_logR(n - r, G - k)

            log_p = log_theta[k] + logRnum - logRden

            u = jax.random.uniform(subkey)
            if jnp.log(u) < log_p:
                sample[k] = 1
                r += 1

            if r == n:
                break

        if sample.sum() != n:
            raise AssertionError(
                f"This should not happen. "
                f"Had sample with {sample.sum()}, rather than {n} ones."
            )
        samples.append(sample)

    return jnp.asarray(samples, dtype=int)
