"""Utilities for building (contrained) Bernoulli mixture models."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Float, Int, Array
from typing import Union

_FloatLike = Union[float, Float[Array, " "]]


def add_bernoulli_noise(
    key: jax.Array,
    Y: Int[Array, " *dimensions"],
    false_positive_rate: _FloatLike,
    false_negative_rate: _FloatLike,
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
    false_positive_rate: _FloatLike,
    false_negative_rate: _FloatLike,
) -> Float[Array, " *dimensions"]:
    """Adjusts a noiseless Bernoulli mixture model
    by adding false positive and negative rates."""
    compl = 1.0 - (false_positive_rate + false_negative_rate)
    return false_positive_rate + compl * mixture_components


def adjust_mixture_components_removing_noise(
    mixture_components: Float[Array, " *dimensions"],
    false_positive_rate: _FloatLike,
    false_negative_rate: _FloatLike,
) -> Float[Array, " *dimensions"]:
    """Recovers noiseless model components from a noisy
    Bernoulli mixture model."""
    compl = 1.0 - (false_positive_rate + false_negative_rate)
    return (mixture_components - false_positive_rate) / compl


def loglikelihood_bernoulli_mixture(
    Y: Int[Array, "*dimensions n_features"],
    mixture_weights: Float[Array, " n_components"],
    mixture_components: Float[Array, "n_components n_features"],
) -> _FloatLike:
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


def calculate_logmarginal(
    y: Int[Array, " n_loci"],
    indices: Int[Array, " n_loci"],
    mixture_weights: Float[Array, " n_components"],
    mixture_components: Float[Array, "n_components n_genes"],
) -> _FloatLike:
    """Calculates

    log P(Y_indices = y | weights, components).

    Args:
        y: values taken at the loci
        indices: specified loci. All the other loci are marginalized out
    """
    return loglikelihood_bernoulli_mixture(
        Y=y[None, :],
        mixture_weights=mixture_weights,
        mixture_components=mixture_components[:, indices],
    )


def log_p_cond(
    *,
    response_value: int,
    condition_value: int,
    response_index: int,
    condition_index: int,
    mixture_weights: Float[Array, " n_components"],
    mixture_components: Float[Array, "n_components n_genes"],
) -> _FloatLike:
    """Evaluates the conditional log-probability
    log P(Y[response_index] = response_value | Y[condition_index] = condition_value)
    """
    # log P(Y[condition_index] = condition_value)
    log_pcond = calculate_logmarginal(
        y=jnp.array([condition_value], dtype=int),
        indices=jnp.array([condition_index], dtype=int),
        mixture_weights=mixture_weights,
        mixture_components=mixture_components,
    )
    # log P(Y[both indices] = both values)
    log_pjoint = calculate_logmarginal(
        y=jnp.array([response_value, condition_value], dtype=int),
        indices=jnp.array([response_index, condition_index], dtype=int),
        mixture_weights=mixture_weights,
        mixture_components=mixture_components,
    )
    # Conditional log-probability
    return log_pjoint - log_pcond


def _conditional_probability_difference(
    response: int,
    conditioning: int,
    mixture_weights: Float[Array, " n_components"],
    mixture_components: Float[Array, "n_components n_genes"],
) -> _FloatLike:
    """Estimates the difference between conditional probabilities:

    P(response=1 | conditioning = 1) - P(response=1 | conditioning = 0)
    """
    log_p11 = log_p_cond(
        response_value=1,
        condition_value=1,
        response_index=response,
        condition_index=conditioning,
        mixture_weights=mixture_weights,
        mixture_components=mixture_components,
    )
    log_p10 = log_p_cond(
        response_value=1,
        condition_value=0,
        response_index=response,
        condition_index=conditioning,
        mixture_weights=mixture_weights,
        mixture_components=mixture_components,
    )
    return jnp.exp(log_p11) - jnp.exp(log_p10)


def logodds_ratio(
    locus1: int,
    locus2: int,
    mixture_weights: Float[Array, " n_components"],
    mixture_components: Float[Array, "n_components n_genes"],
) -> _FloatLike:
    """Calculates the logodds ratio at the specified loci,
    which measures exclusivity and co-occurence."""

    def log_tau(value1: int, value2: int) -> _FloatLike:
        values = jnp.asarray([value1, value2], dtype=int)
        indices = jnp.asarray([locus1, locus2], dtype=int)
        return calculate_logmarginal(
            y=values,
            indices=indices,
            mixture_weights=mixture_weights,
            mixture_components=mixture_components,
        )

    log_numerator = log_tau(1, 0) + log_tau(0, 1)
    log_denominator = log_tau(0, 0) + log_tau(1, 1)
    return log_numerator - log_denominator
