import matplotlib.pyplot as plt
import numpy as np

from jnotype.checks import plot_histograms


def test_plot_histograms(
    save_artifact,
    tmp_path,
) -> None:
    rng = np.random.default_rng(42)

    fig, axs = plt.subplots(2, 2, figsize=(6, 6))

    draws = rng.normal(size=(200, 300))

    ax = axs[0, 0]
    ax.set_title("Median and quantiles")
    plot_histograms(
        ax=ax,
        data=draws,
        bins=np.linspace(-3, 3, 20),
        main_type="median",
        uncertainty_type="quantiles",
        alpha_uncertainty=0.1,
        quantiles=[(0.05, 0.95), (0.25, 0.75)],
        density=True,
    )

    ax = axs[0, 1]
    ax.set_title("Mean and trajectories")
    plot_histograms(
        ax=ax,
        data=draws,
        bins=np.linspace(-3, 3, 20),
        main_type="mean",
        uncertainty_type="trajectories",
        alpha_uncertainty=0.1,
        quantiles=[(0.05, 0.95), (0.25, 0.75)],
        density=True,
    )

    ax = axs[1, 0]
    ax.set_title("Few bins")
    plot_histograms(
        ax=ax,
        data=draws,
        bins=np.linspace(-3, 3, 10),
        main_type="median",
        uncertainty_type="quantiles",
        density=True,
    )

    ax = axs[1, 1]
    ax.set_title("Counts, no uncertainty")
    plot_histograms(
        ax=ax,
        data=draws,
        bins=np.linspace(-3, 3, 20),
        main_type="mean",
        uncertainty_type=None,
        density=False,
    )

    if save_artifact:
        path = tmp_path / "test_plot_histograms.pdf"
        fig.tight_layout()
        fig.savefig(path)
