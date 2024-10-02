"""Utilities for building (contrained) Bernoulli mixture models."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float, Int, Array


def add_bernoulli_noise(
    key: jax.Array,
    Y: Int[Array, " *dimensions"],
    false_positive_rate: float,
    false_negative_rate: float,
) -> Int[Array, " *dimensions"]:
    """Adds false positives and false negatives to the data."""
    # Generate random numbers uniformly between 0 and 1
    R = jax.random.uniform(key, shape=Y.shape)

    # Determine flip probabilities based on the current values in Y
    flip_probs = jnp.where(Y == 0, false_positive_rate, false_negative_rate)

    # Create a mask where flips should occur
    flip_mask = R < flip_probs

    # Apply the flips to Y
    Y_noisy = jnp.where(flip_mask, 1 - Y, Y)

    return Y_noisy


def sample_bernoulli_mixture(
    key: jax.Array,
    n_samples: int,
    mixture_weights: Float[Array, " n_components"],
    mixture_components: Float[Array, "n_components n_features"],
) -> Int[Array, "n_samples n_features"]:
    """Samples from a given Bernoulli mixture model."""
    key1, key2 = jax.random.split(key)

    # Convert mixture weights to logits for categorical sampling
    component_logits = jnp.log(mixture_weights)
    component_indices = jax.random.categorical(
        key1, logits=component_logits, shape=(n_samples,)
    )

    # Select the component probabilities for each sample
    component_probs = mixture_components[
        component_indices, :
    ]  # Shape: (n_samples, n_features)

    # Sample from Bernoulli distributions with these probabilities
    samples = jax.random.bernoulli(key2, p=component_probs).astype(jnp.int32)

    return samples


def adjust_mixture_components_for_noise(
    mixture_components: Float[Array, " *dimensions"],
    false_positive_rate: float,
    false_negative_rate: float,
) -> Float[Array, " *dimensions"]:
    """Adjusts a Bernoulli mixture model by false positive and negative rates."""
    compl = 1.0 - (false_positive_rate + false_negative_rate)
    return false_positive_rate + compl * mixture_components


def loglikelihood_bernoulli_mixture(
    Y: Int[Array, "*dimensions n_features"],
    mixture_weights: Float[Array, " n_components"],
    mixture_components: Float[Array, "n_components n_features"],
) -> float:
    """Calculates the loglikelihood in a Bernoulli mixture model."""
    # Compute log probabilities of the mixture components
    log_p = jnp.log(mixture_components)  # Shape: (n_components, n_features)
    log_1_minus_p = jnp.log1p(-mixture_components)  # Shape: (n_components, n_features)

    # Expand dimensions for broadcasting
    Y_expanded = jnp.expand_dims(Y, axis=-2)  # Shape: (*dimensions, 1, n_features)
    log_p = jnp.expand_dims(log_p, axis=0)  # Shape: (1, n_components, n_features)
    log_1_minus_p = jnp.expand_dims(
        log_1_minus_p, axis=0
    )  # Shape: (1, n_components, n_features)

    # Compute the log-likelihood for each component
    component_log_likelihood = jnp.sum(
        Y_expanded * log_p + (1 - Y_expanded) * log_1_minus_p, axis=-1
    )  # Shape: (*dimensions, n_components)

    # Add log mixture weights
    log_mixture_weights = jnp.log(mixture_weights)  # Shape: (n_components,)
    total_log_prob = (
        component_log_likelihood + log_mixture_weights
    )  # Broadcasting over n_components

    # Compute the total log-likelihood using logsumexp for numerical stability
    log_likelihood_per_sample = logsumexp(
        total_log_prob, axis=-1
    )  # Shape: (*dimensions,)

    # Sum over all samples
    total_log_likelihood = jnp.sum(log_likelihood_per_sample)

    return total_log_likelihood
