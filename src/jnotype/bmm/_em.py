"""The Expectation-Maximization algorithm for Bernoulli Mixture Model."""
import dataclasses
import time
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


def compute_responsibilities(
    observed: Int[Array, "N K"],
    mixing: Float[Array, "K B"],
    proportions: Float[Array, " B"],
) -> Float[Array, "N B"]:
    """Updates responsibility P(Z | X, params) for each data point.

    Args:
        observed: binary matrix with observations
        mixing: P(observed | cluster).
          Each entry should be in the (0, 1) interval.
        proportions: vector encoding the prevalence of each class.
          Entries should sum up to 1.

    Returns:
        responsibilities for each point. Each row sums up to 1.
    """
    # We'll calculate in log space:
    success = jnp.einsum("NK,KB->NB", observed, jnp.log(mixing))
    fails = jnp.einsum("NK,KB->NB", 1 - observed, jnp.log1p(-mixing))

    logits = success + fails + jnp.log(proportions)[None, :]
    return jax.nn.softmax(logits, axis=1)


def compute_proportions(responsibilities: Float[Array, "N B"], /) -> Float[Array, " B"]:
    """Estimates the proportions (prevalence of each class).

    Args:
        responsibilities: P(Z  | X, params) for each data point.
          Each row should sum up to 1

    Returns:
        vector of prevalence, sums up to 1
    """
    return jnp.mean(responsibilities, axis=0)


def compute_mixing(
    observed: Int[Array, "N K"], responsibilities: Float[Array, "N B"]
) -> Float[Array, "K B"]:
    """Computes mixing matrix P(observed | cluster)

    Args:
        observed: binary observations
        responsibilities: responsibility P(Z | X, params) for each data point
    """
    weights = jnp.sum(responsibilities, axis=0)  # Shape (B,)
    averaged_obs = jnp.einsum("NK,NB->KB", observed, responsibilities)

    return averaged_obs / weights[None, :]


@jax.jit
def em_step(
    observed: Int[Array, "N K"],
    mixing: Float[Array, "K B"],
    proportions: Float[Array, " B"],
) -> tuple[Float[Array, "N B"], Float[Array, "K B"], Float[Array, " B"],]:
    """The E and M step combined, for better JIT compiler optimisation.

    Args:
        observed: binary observations
        mixing: estimate for the P(observations | cluster)
          matrix
        proportions: estimate for the P(Z) vector

    Returns:
        responsibilities P(Z | X, params) for each data point
        new mixing matrix P(observations | cluster)
        new proportions vector P(Z)
    """
    # The E step
    responsibilities = compute_responsibilities(
        observed=observed, mixing=mixing, proportions=proportions
    )

    # The M step
    new_mixing = compute_mixing(observed=observed, responsibilities=responsibilities)
    new_proportions = compute_proportions(responsibilities)

    return responsibilities, new_mixing, new_proportions


@dataclasses.dataclass
class EMHistoryEntry:
    """A single entry of the training history.

    Attributes:
        step: training step
        time: elapsed time since the start of the algorithm
        proportions: prevalence vector P(Z), sums up to 1
        responsibilities: P(Z | X, params) for each data point.
            Each row sums up to 1
        mixing: P(observed | cluster), entries between (0, 1)
    """

    step: int
    time: float
    proportions: Float[Array, " B"]
    responsibilities: Float[Array, "N B"]
    mixing: Float[Array, "K B"]


@dataclasses.dataclass
class EMOutput:
    """Output from the Expectation-Maximization algorithm
    for the Bernoulli mixture model.

    Attributes:
        proportions: prevalence vector P(Z), sums up to 1
        responsibilities: P(Z | X, params) for each data point.
          Each row sums up to 1
        mixing: P(observed | cluster), entries between (0, 1)
        early_stopping_threshold: threshold used for early stopping
        n_steps: number of steps until convergence (or reaching
          the maximum allowed number)
        max_n_steps: maximum number of steps allowed
        history: training history
    """

    proportions: Float[Array, " B"]
    responsibilities: Float[Array, "N B"]
    mixing: Float[Array, "K B"]

    n_steps: int
    max_n_steps: int
    early_stopping_threshold: float
    history: list[EMHistoryEntry]


def _n_clusters(
    n_clusters: Optional[int],
    mixing_init: Optional[Float[Array, "K B"]],
    proportions_init: Optional[Float[Array, " B"]],
) -> int:
    """Tries to infer the number of clusters from problem specification."""
    n_clusters_mixing = None if mixing_init is None else mixing_init.shape[-1]
    n_clusters_prop = None if proportions_init is None else proportions_init.shape[0]

    n_clusters = n_clusters or n_clusters_mixing or n_clusters_prop
    if n_clusters is None:
        raise ValueError("It was not possible to infer the number of clusters.")
    return n_clusters


def _init(
    *,
    n_features: int,
    n_clusters: Optional[int],  # type: ignore
    key: Optional[jax.random.PRNGKeyArray],
    mixing_init: Optional[Float[Array, "K B"]],
    proportions_init: Optional[Float[Array, " B"]],
) -> tuple[Float[Array, "K B"], Float[Array, " B"]]:
    # If `mixing_init` and `proportions_init` are set,
    # we can just return them
    if mixing_init is not None and proportions_init is not None:
        return mixing_init, proportions_init
    # Now we know that we need to sample. To sample, we need
    # random key and cluster labels
    if key is None:
        raise ValueError(
            "You need to provide `key` or " "both `mixing_init` and `proportions_init`."
        )
    key_mixing, key_proportions = jax.random.split(key)

    # Now we know we have keys.
    # Let's infer the number of clusters
    n_clusters: int = _n_clusters(
        n_clusters=n_clusters,
        mixing_init=mixing_init,
        proportions_init=proportions_init,
    )

    # Now we know how to sample
    if mixing_init is None:
        mixing_init = jax.random.beta(key_mixing, 1, 1, shape=(n_features, n_clusters))
    if proportions_init is None:
        proportions_init = jax.random.dirichlet(key_proportions, jnp.ones(n_clusters))

    return mixing_init, proportions_init


def _log(msg: str, /) -> None:
    """Logs the message."""
    print(msg)  # TODO(Pawel): Replace with a logger.


def _maxabs(a: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a)))


def _should_stop(
    *,
    threshold: float,
    old_mixing: Float[Array, "K B"],
    old_proportions: Float[Array, " B"],
    new_mixing: Float[Array, "K B"],
    new_proportions: Float[Array, " B"],
) -> bool:
    """Returns True iff the stopping criterion is satisfied."""
    diff_proportions = _maxabs(old_proportions - new_proportions)
    diff_mixing = _maxabs(old_mixing - new_mixing)
    return max(diff_mixing, diff_proportions) < threshold


def expectation_maximization(
    observed: Int[Array, "K B"],
    *,
    n_clusters: Optional[int] = None,
    key: Optional[jax.random.PRNGKeyArray] = None,
    mixing_init: Optional[Float[Array, "K B"]] = None,
    proportions_init: Optional[Float[Array, " B"]] = None,
    max_n_steps: int = 10_000,
    verbose: bool = False,
    record_history: Optional[int] = None,  # type: ignore
    early_stopping_threshold: float = 1e-3,
) -> EMOutput:
    """Expectation-Maximization algorithm for the Bernoulli mixture model.

    Args:
        observed: observed data
        proportions_init: starting point
          for the class prevalence vector.
          If None, it will be randomly sampled
        mixing_init: starting point for the P(observations | cluster) matrix.
          If None, it will be randomly sampled
        n_clusters: number of clusters to be fit
        key: JAX random key. Needs to be specified
          if any of the `proportions_init` or `mixing_init` is None
        max_n_steps: maximum number of steps
        verbose: whether to log progress
        record_history: interval at which the points will be recorded
        early_stopping_threshold: controls early stopping
    """
    mixing, proportions = _init(
        n_features=observed.shape[1],
        n_clusters=n_clusters,
        mixing_init=mixing_init,
        proportions_init=proportions_init,
        key=key,
    )
    t0 = time.time()

    # If `record_history` is None, we will set it to a large number,
    # which will never be reached
    record_history: int = record_history or max_n_steps + 111

    history = []

    if max_n_steps < 1:  # Ensures the loop will run at least for one iteration
        raise ValueError("At least one optimization step is required.")

    # Declare these variables outside the loop, although they will be overridden
    step: int = 0
    responsibilities: jax.Array = jnp.full(
        (observed.shape[0], len(proportions)), 1 / len(proportions)
    )

    for step in range(1, max_n_steps + 1):
        # Run the update
        responsibilities, mixing_, proportions_ = em_step(
            observed=observed,
            mixing=mixing,
            proportions=proportions,
        )
        # Evaluate the stopping criterion
        should_stop = _should_stop(
            threshold=early_stopping_threshold,
            new_mixing=mixing_,
            new_proportions=proportions_,
            old_mixing=mixing,
            old_proportions=proportions,
        )

        # Update the variables
        mixing = mixing_
        proportions = proportions_

        # Now check if we need to update the history
        if step % record_history == 0:
            delta_t = time.time() - t0
            entry = EMHistoryEntry(
                responsibilities=responsibilities,
                mixing=mixing,
                proportions=proportions,
                step=step,
                time=delta_t,
            )
            history.append(entry)
            if verbose:
                _log(
                    f"At time {delta_t} step {step}/{max_n_steps} was reached "
                    f"({step / delta_t:.1f} steps/second)."
                )

        # Now stop if needed
        if should_stop:
            _log(f"Early stopping at step {step}/{max_n_steps}.")
            break

    if verbose:
        delta_t = time.time() - t0
        _log(
            f"At time {delta_t} optimization finished at step {step}/{max_n_steps} "
            f"({step / delta_t:.1f} steps/second)."
        )

    return EMOutput(
        proportions=proportions,
        responsibilities=responsibilities,
        mixing=mixing,
        n_steps=step,
        max_n_steps=max_n_steps,
        early_stopping_threshold=early_stopping_threshold,
        history=history,
    )
