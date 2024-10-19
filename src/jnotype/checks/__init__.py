"""Subpackage with generic predictive checks."""

from jnotype.checks._histograms import plot_histograms, calculate_quantiles
from jnotype.checks._statistics import (
    calculate_mcc,
    calculate_mutation_frequencies,
    calculate_number_of_mutations_histogram,
    convert_genotypes_to_integers,
    convert_integers_to_genotypes,
    calculate_atoms_occurrence,
    subsample_pytree,
    simulate_summary_statistic,
)
from jnotype.checks._plots import rc_context, rcParams, plot_summary_statistic

__all__ = [
    "plot_histograms",
    "calculate_quantiles",
    "calculate_mcc",
    "calculate_mutation_frequencies",
    "calculate_number_of_mutations_histogram",
    "rc_context",
    "rcParams",
    "plot_summary_statistic",
    "convert_genotypes_to_integers",
    "convert_integers_to_genotypes",
    "calculate_atoms_occurrence",
    "subsample_pytree",
    "simulate_summary_statistic",
]
