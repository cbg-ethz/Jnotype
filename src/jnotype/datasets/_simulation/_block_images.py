"""Simulation of binary images using Bernoulli mixture model."""
from typing import Optional

from jaxtyping import Array, Float, Int
import jax.numpy as jnp
from jax import random


class BlockImagesSampler:
    """Data set of 6 x 6 binary images
    composed of 4 block features.

    This is based (although not identical)
    on the data set described in

    F. Doshi-Velez, Z. Ghahramani,
    Correlated Non-Parametric Latent Feature Models
    UAI 2009
    """

    _feature_plus = jnp.asarray(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )

    _feature_circle = jnp.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=int,
    )

    _feature_t = jnp.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=int,
    )

    _feature_stairs = jnp.asarray(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )

    _cluster_matrices: list[Int[Array, "6 6"]] = [
        _feature_t,
        _feature_stairs,
        _feature_circle,
        _feature_plus,
        _feature_circle + _feature_stairs,
        _feature_circle + _feature_plus,
        _feature_plus + _feature_t,
        _feature_stairs + _feature_t,
        _feature_circle + _feature_plus + _feature_stairs,
        _feature_circle + _feature_stairs + _feature_t,
        _feature_circle + _feature_plus + _feature_stairs + _feature_t,
    ]

    def __init__(
        self,
        false_positive: float = 0.1,
        false_negative: float = 0.1,
        n_classes: int = 6,
    ) -> None:
        """
        Args:
            false_positive: probability that the 0 pixel will be rendered as 1
            false_negative: probability that the 1 pixel will be rendered as 0
            n_classes: number of classes, between 1 and 11
        """
        if (
            min(false_negative, false_positive) < 0
            or max(false_negative, false_positive) > 1
        ):
            raise ValueError(
                "False positive and false negative rate must be in the interval [0, 1]."
                f"False negative: {false_negative:.3f}, "
                f"false positive: {false_positive:.3f}."
            )

        if n_classes < 1 or n_classes > 11:
            raise ValueError("The number of classes should be between 1 and 11.")

        self._false_positive = false_positive
        self._false_negative = false_negative
        self._n_classes = n_classes

    @property
    def cluster_templates(self) -> Int[Array, "n_classes 6 6"]:
        """Returns binary matrices with cluster templates.

        See also:
            `mixing`: noisy version of cluster templates
        """
        return jnp.asarray(self._cluster_matrices[: self._n_classes])

    @property
    def mixing(self) -> Float[Array, "n_classes 6 6"]:
        """Mixing matrices used to sample the images.

        Returns:
            array of shape (n_classes, 6, 6), where entry (n, i, j)
              represents P(Y_{ij} = 1 | class = n)

        See also:
            `cluster_templates`: noiseless version of mixing matrices,
              in which the mixing is deterministic
        """
        templates = self.cluster_templates
        return jnp.asarray(
            templates * (1 - self._false_negative)
            + (1 - templates) * self._false_positive,
            dtype=float,
        )

    def sample_dataset(
        self,
        key: random.PRNGKeyArray,
        n_samples: int,
        probs: Optional[Float[Array, " n_classes"]] = None,
        logits: Optional[Float[Array, " n_classes"]] = None,
    ) -> tuple[Int[Array, " n_samples"], Int[Array, "n_samples 6 6"]]:
        """Samples the data set.

        Args:
            key: JAX random key
            n_samples: number of samples to draw
            probs: prevalence :math:`\\log P(class)` vector.
            logits: :math:`\\log P(class)` vector,
              defining the log-prevalence of each class.

        Returns:
            labels, shape (n_samples,).
              Each entry is in the set {0, 1, ..., n_classes-1}
            images, each image is a binary array of shape (6, 6)

        Note:
            If neither `probs` nor `logits` are provided, a uniform
            distribution is assumed over all classes.
            You cannot provide both arguments at the same time.

        Raises:
            ValueError if both `probs` and `logits` are specified
        """
        if probs is not None and logits is not None:
            raise ValueError("Only one of `probs` and `logits` can be provided.")

        # Try to set up logits using probabilities
        if probs is not None:
            logits = jnp.log(probs)

        # If logits are still not set, we assume uniform distribution
        if logits is None:
            logits = jnp.zeros(self._n_classes)

        key_labels, key_features = random.split(key)

        labels = random.categorical(key_labels, logits, shape=(n_samples,))
        mixings = self.mixing[labels, :, :]

        features = jnp.asarray(random.bernoulli(key_features, mixings), dtype=int)

        return labels, features
