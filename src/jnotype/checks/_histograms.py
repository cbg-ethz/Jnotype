"""Plotting histograms of data."""

from typing import Sequence, Union, Literal

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def calculate_quantiles(
    samples: Float[Array, "n_samples dimension"],
    quantiles: Float[Array, " n_quantiles"],
) -> Float[Array, "n_quantiles dimension"]:
    """Calculates quantiles.

    Args:
        samples: shape (n_samples, dimension)
        quantiles: shape (n_quantiles,)

    Returns:
        quantile value for each dimension,
          shape (n_quantiles, dimension)
    """
    return jnp.quantile(samples, axis=0, q=quantiles)


def apply_histogram(
    draws: Float[Array, "n_datasets n_values"],
    bins: Union[int, Sequence[float], np.ndarray],
    density: bool,
) -> np.ndarray:
    """Maps `jnp.histogram` over several vectors of values
    to contruct several histograms.

    Args:
        draws: shape (n_samples, values)
        bins: number of bins or bin edges
        density: whether to normalize the histogram

    Returns:
        histogram counts, shape (n_samples, bins)
    """
    _, bins = jnp.histogram(draws[0], bins=bins, density=density)

    def f(sample):
        """Auxiliary function used for jax.vmap"""
        return jnp.histogram(sample, bins=bins, density=density)[0]

    return jax.vmap(f)(draws)


def plot_histograms(
    ax: plt.Axes,
    data: np.ndarray,
    bins: Union[int, Sequence[float], np.ndarray],
    alpha_main: float = 1.0,
    color_main: str = "C2",
    main_type: Literal[None, "mean", "median"] = "median",
    alpha_uncertainty: float = 0.1,
    color_uncertainty: str = "C2",
    uncertainty_type: Literal[None, "trajectories", "quantiles"] = "quantiles",
    density: bool = False,
    quantiles: Sequence[tuple[float, float]] = ((0.025, 0.975), (0.25, 0.75)),
) -> np.ndarray:
    """Plots histograms of data.

    Args:
        ax: axes to plot in
        data: array of shape (n_samples, counts) or (counts,)
        bins: number of bins or bin edges
        alpha: transparency of histogram
        plot_trajectories: whether individual trajectories should be plotted
        plot_error_bars: whether error bars should be plotted
    """
    # Make sure the data is of shape (n_samples, counts)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[None, :]

    _, bins = np.histogram(data[0], bins=bins)

    histograms = apply_histogram(data, bins=bins, density=density)

    # Plot mean/median
    if main_type == "mean":
        histograms_main = np.mean(histograms, axis=0)
        ax.stairs(histograms_main, bins, alpha=alpha_main, color=color_main)
    elif main_type == "median":
        histograms_main = np.median(histograms, axis=0)
        ax.stairs(histograms_main, bins, alpha=alpha_main, color=color_main)
    elif main_type is None:
        pass
    else:
        raise ValueError(f"Unknown main type {main_type}")

    # Plot uncertainty
    if uncertainty_type == "trajectories":
        for cnts in histograms:
            ax.stairs(cnts, bins, alpha=alpha_uncertainty, color=color_uncertainty)
    elif uncertainty_type == "quantiles":
        for qs in quantiles:
            assert len(qs) == 2
            qs_sorted = np.asarray(sorted(qs))
            quants = calculate_quantiles(histograms, quantiles=qs_sorted)
            ax.stairs(
                quants[1],
                bins,
                alpha=alpha_uncertainty,
                color=color_uncertainty,
                fill=True,
                baseline=quants[0],
            )
    elif uncertainty_type is None:
        pass
    else:
        raise ValueError(f"Unknown uncertainty type {uncertainty_type}")
