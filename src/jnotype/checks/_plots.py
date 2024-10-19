"""Plotting utilities for performing
graphical prior and posterior predictive checking."""

import copy
from contextlib import contextmanager
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Array

from jnotype.checks._histograms import calculate_quantiles


def _wrap_array(y):
    """A simple wrapper, working in a similar
    manner as the Maybe functor:

    Optional[ArrayLike] -> Optional[NumPy Array]
    """
    if y is not None:
        return np.array(y)
    else:
        return None


# Global default plotting parameters,
# similarly as in Matplotlib
rcParams: dict = {
    "color_data": "C0",
    "color_simulations": "C1",
    "quantiles": ((0.025, 0.975), (0.25, 0.75)),  # tuple[tuple[float, float], ...]
    "uncertainty_type": "quantiles",  # Literal["quantiles", "trajectories", "none"]
    "summary_type": "median",  #  Literal["mean", "median", "none"]
    "trajectory_linewidth": 0.01,
    "num_trajectories": 50,
    "summary_linewidth": 1.0,
    "summary_markersize": 3.0**2,
    "summary_marker": ".",  # Use "none" for no marker
    "data_linewidth": 1.0,
    "data_markersize": 3.0**2,
    "data_marker": ".",
}


@contextmanager
def rc_context(rc: dict):
    """Context manager maintained
    in the same manner as in Matplotlib."""
    original = copy.deepcopy(rcParams)
    try:
        rcParams.update(rc)
        yield
    finally:
        rcParams.update(original)


def _plot_quantiles(
    ax: plt.Axes,
    x_axis: Float[Array, " n_points"],
    ys: Float[Array, "n_simulations n_points"],
    color: Optional[str],
    alpha: Optional[float],
    quantiles: Optional[Sequence[tuple[float, float]]],
) -> None:
    """Plots uncertainty in terms of quantiles
    calculated separately for each coordinate."""
    # If quantiles are None, use the default value
    if quantiles is None:
        quantiles = rcParams["quantiles"]
    # Ensure that quantiles is a list
    quantiles = list(quantiles)  # type: ignore

    # TODO(Pawel): Ensure that the quantiles are nested.

    # If alpha is not set, calculate a reasonable value
    if alpha is None:
        alpha = min(0.2, 1 / (1 + len(quantiles)))

    if color is None:
        color = rcParams["color_simulations"]

    # Plot quantiles
    for q_min, q_max in quantiles:
        if q_min >= q_max:
            raise ValueError(f"Quantile {q_min} >= {q_max}.")
        y_quant = calculate_quantiles(
            ys,
            quantiles=np.array([q_min, q_max], dtype=float),  # type: ignore
        )
        ax.fill_between(
            x_axis,
            np.asarray(y_quant[0, :]),  # type: ignore
            np.asarray(y_quant[-1, :]),  # type: ignore
            color=color,
            alpha=alpha,
            edgecolor=None,
        )


def _plot_trajectories(
    ax: plt.Axes,
    x_axis: Float[Array, " n_points"],
    y_simulated: Float[Array, "n_simulations n_points"],
    color: Optional[str],
    alpha: Optional[float],
    num_trajectories: Optional[int],
    trajectories_subsample_key: int,
    trajectory_linewidth: Optional[float],
) -> None:
    """Plots uncertainty in terms
    of individual simulated samples."""
    if num_trajectories is None:
        num_trajectories = rcParams["num_trajectories"]
    assert num_trajectories is not None

    if num_trajectories <= 0:
        raise ValueError("num_trajectories has to be at least 1")
    if trajectory_linewidth is None:
        trajectory_linewidth = rcParams["trajectory_linewidth"]
    if color is None:
        color = rcParams["color_simulations"]
    if alpha is None:
        alpha = min(0.1, 1 / (1 + num_trajectories))

    num_simulations = y_simulated.shape[0]

    indices = np.arange(num_simulations)
    if num_trajectories < num_simulations:
        rng = np.random.default_rng(trajectories_subsample_key)
        indices = rng.choice(indices, size=num_trajectories, replace=False)

    for index in indices:
        ax.plot(
            x_axis,
            y_simulated[index],
            linewidth=trajectory_linewidth,
            color=color,
            alpha=alpha,
        )


def _plot_main_summary(
    ax: plt.Axes,
    x_axis: Float[Array, " n_points"],
    y_simulated: Float[Array, "n_simulations n_points"],
    summary_type: Literal["mean", "median", "none", "default"],
    summary_linewidth: Optional[float],
    summary_markersize: Optional[float],
    summary_marker: str,
    color: Optional[str],
) -> None:
    """Plots some "summary" such as mean or median of the
    statistic from simulations."""
    if summary_type == "default":
        summary_type = rcParams["summary_type"]
    if summary_linewidth is None:
        summary_linewidth = rcParams["summary_linewidth"]
    if summary_markersize is None:
        summary_markersize = rcParams["summary_markersize"]
    if color is None:
        color = rcParams["color_simulations"]
    if summary_marker == "default":
        summary_marker = rcParams["summary_marker"]

    y = None
    if summary_type == "none":
        return
    elif summary_type == "median":
        y = np.median(y_simulated, axis=0)
    elif summary_type == "mean":
        y = np.mean(y_simulated, axis=0)
    else:
        raise ValueError(f"Simulated summary type {summary_type} not known.")

    assert y is not None
    assert y.shape == x_axis.shape

    # Now plot the summary statistic
    ax.plot(
        x_axis,
        y,
        c=color,
        marker=summary_marker,
        markersize=summary_markersize,
        linewidth=summary_linewidth,
    )


def _plot_data(
    ax: plt.Axes,
    x_axis: Float[Array, " n_points"],
    y: Float[Array, " n_points"],
    data_linewidth: Optional[float],
    data_markersize: Optional[float],
    data_marker: str,
    color: Optional[str],
) -> None:
    """Plots the data."""
    if data_linewidth is None:
        data_linewidth = rcParams["data_linewidth"]
    if data_markersize is None:
        data_markersize = rcParams["data_markersize"]
    if color is None:
        color = rcParams["color_data"]
    if data_marker == "default":
        data_marker = rcParams["data_marker"]

    if x_axis.shape != y.shape:
        raise ValueError("x and y have different shapes")

    ax.plot(
        x_axis,
        y,
        c=color,
        marker=data_marker,
        markersize=data_markersize,
        linewidth=data_linewidth,
    )


def _plot_uncertainty(
    ax: plt.Axes,
    x_axis: Float[Array, " n_points"],
    y_simulated: Float[Array, "n_simulations n_points"],
    color_simulated: Optional[str],
    uncertainty_type: Literal["default", "none", "trajectories", "quantiles"],
    uncertainty_alpha: Optional[float],
    quantiles: Optional[Sequence[tuple[float, float]]],
    num_trajectories: Optional[int],
    trajectory_linewidth: Optional[float],
    trajectories_subsample_key: int,
) -> None:
    """Plots uncertainty either in terms
    of quantiles or trajectories."""
    # If we use the default settings, look for the right ones:
    if uncertainty_type == "default":
        uncertainty_type = rcParams["uncertainty_type"]

    # Now decide how to plot the uncertainty
    if uncertainty_type == "none":
        return  # We don't have to plot anything
    elif uncertainty_type == "quantiles":
        _plot_quantiles(
            ax=ax,
            x_axis=x_axis,
            ys=y_simulated,
            color=color_simulated,
            alpha=uncertainty_alpha,
            quantiles=quantiles,
        )
    elif uncertainty_type == "trajectories":
        _plot_trajectories(
            ax=ax,
            x_axis=x_axis,
            y_simulated=y_simulated,
            color=color_simulated,
            alpha=uncertainty_alpha,
            num_trajectories=num_trajectories,
            trajectories_subsample_key=trajectories_subsample_key,
            trajectory_linewidth=trajectory_linewidth,
        )
    else:
        raise ValueError(f"Uncertainty {uncertainty_type} not recognized")


def plot_summary_statistic(
    ax: plt.Axes,
    x_axis: Optional[Float[Array, " n_points"]] = None,
    y_data: Optional[Float[Array, " n_points"]] = None,
    y_simulated: Optional[Float[Array, "n_simulations n_points"]] = None,
    color_data: Optional[str] = None,
    color_simulations: Optional[str] = None,
    summary_type: Literal["none", "default", "mean", "median"] = "default",
    summary_linewidth: Optional[float] = None,
    summary_markersize: Optional[float] = None,
    summary_marker: str = "default",
    uncertainty_type: Literal[
        "none", "default", "trajectories", "quantiles"
    ] = "default",
    uncertainty_alpha: Optional[float] = None,
    # Settings for plotting uncertainty
    quantiles: Optional[Sequence[tuple[float, float]]] = None,
    num_trajectories: Optional[int] = None,
    trajectories_subsample_key: int = 42,
    trajectory_linewidth: Optional[float] = None,
    # Settings for plotting data
    data_linewidth: Optional[float] = None,
    data_markersize: Optional[float] = None,
    data_marker: str = "default",
    residuals: bool = False,
    residuals_type: Literal[None, "mean", "median"] = None,
) -> None:
    """Plots a summary statistic together with uncertainty.

    Args:
        ax: axes on which the plot is done
        x_axis: positions on the x_axis, shape (n_points,).
            Leave as `None` to infer from the data.
        y_data: summary statistic of the data, shape (n_points,),
            where `n_points` is the dimensionality of the
            summary statistic vector. Leave as `None` to not plot.
        y_simulated: summary statistic corresponding to simulations,
            shape (n_simulations, n_points)
    """
    # Try to wrap all arraylike objects
    x_axis, y_data, y_simulated = (
        _wrap_array(x_axis),  # type: ignore
        _wrap_array(y_data),  # type: ignore
        _wrap_array(y_simulated),  # type: ignore
    )

    if x_axis is None:
        if y_data is not None:
            x_axis = np.arange(y_data.shape[0])  # type: ignore
        elif y_simulated is not None:
            x_axis = np.arange(y_simulated.shape[-1])  # type: ignore
        else:
            raise ValueError("No data to plot")
    assert x_axis is not None

    n_points = x_axis.shape[0]
    if y_data is not None:
        if y_data.shape != (n_points,):
            raise ValueError("Data has wrong shape.")
    if y_simulated is not None:
        if len(y_simulated.shape) != 2 or y_simulated.shape[-1] != n_points:
            raise ValueError("Simulated data has wrong shape.")

    # Transform data
    if residuals:
        if y_simulated is None:
            raise ValueError("For residual plot one has to provide simulated data.")
        # Try to infer residuals_type from summary_type, if not provided
        if residuals_type is None and summary_type in ["median", "mean"]:
            residuals_type = summary_type  # type: ignore

        if residuals_type is None:
            raise ValueError("Residuals type could not be automatically inferred.")
        elif residuals_type == "mean":
            y_perfect = np.mean(y_simulated, axis=0)
        elif residuals_type == "median":
            y_perfect = np.mean(y_simulated, axis=0)
        else:
            raise ValueError(f"Residuals type {residuals_type} not known.")

        # Calculate the residuals
        y_simulated = y_simulated - y_perfect[None, :]
        if y_data is not None:
            y_data = y_data - y_perfect

    # Plot simulated data
    if y_simulated is not None:
        # Start by plotting uncertainty
        _plot_uncertainty(
            ax=ax,
            x_axis=x_axis,
            y_simulated=y_simulated,
            color_simulated=color_simulations,
            uncertainty_type=uncertainty_type,
            uncertainty_alpha=uncertainty_alpha,
            quantiles=quantiles,
            num_trajectories=num_trajectories,
            trajectories_subsample_key=trajectories_subsample_key,
            trajectory_linewidth=trajectory_linewidth,
        )

        # Now plot the main summary
        _plot_main_summary(
            ax=ax,
            x_axis=x_axis,
            y_simulated=y_simulated,
            summary_type=summary_type,
            summary_linewidth=summary_linewidth,
            summary_markersize=summary_markersize,
            summary_marker=summary_marker,
            color=color_simulations,
        )

    if y_data is not None:
        # Plot real data
        _plot_data(
            ax=ax,
            x_axis=x_axis,
            y=y_data,
            data_linewidth=data_linewidth,
            data_markersize=data_markersize,
            data_marker=data_marker,
            color=color_data,
        )
