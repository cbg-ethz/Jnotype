"""Subpackage with generic predictive checks."""

from jnotype.checks._histograms import plot_histograms, calculate_quantiles
from jnotype.checks._statistics import (
    calculate_mcc,
    calculate_mutation_frequencies,
    calculate_number_of_mutations_histogram,
)

__all__ = [
    "plot_histograms",
    "calculate_quantiles",
    "calculate_mcc",
    "calculate_mutation_frequencies",
    "calculate_number_of_mutations_histogram",
]
