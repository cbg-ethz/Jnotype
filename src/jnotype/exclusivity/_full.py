"""Flexible mutual exclusivity model."""

from itertools import product

import numpy as np

import jax.numpy as jnp

from jaxtyping import Float, Int, Array
import numpyro
import numpyro.distributions as dist

import jnotype.exclusivity._bernoulli_mixtures as bmm


def _map_genotype_array_to_str(arr: Int[Array, "n_samples n_features"]) -> list:
    """Maps a genotype array to a list of strings."""
    a = np.array(arr)
    return ["".join(map(str, point.tolist())) for point in a]


def _all_genotypes_possible(n_genes: int) -> list[str]:
    """Generates a list of all possible genotypes."""
    return _map_genotype_array_to_str(list(product("01", repeat=n_genes)))


def _get_bernoulli_mixture(
    mixing_weights: Float[Array, " n_components"],
    bernoulli_probs: Float[Array, "n_components n_features"],
) -> dist.Distribution:
    """Creates a NumPyro distribution corresponding to
    a Bernoulli mixture model"""
    return dist.MixtureSameFamily(
        mixing_distribution=dist.CategoricalProbs(mixing_weights),
        component_distribution=dist.BernoulliProbs(bernoulli_probs).to_event(
            reinterpreted_batch_ndims=1
        ),
    )


def _truncated_spike_and_slab(
    spike_prob: float,
    spike_width: float,
    slab_width: float,
    truncation: float,
) -> dist.Distribution:
    """A truncated spike and slab distribution.

    Args:
        spike_prob: spike probability
        spike_width: spike width
        slab_width: slab width
        truncation: truncates the (0, 1) interval
          to move from the boundary for numerical stability
    """
    locs = jnp.zeros(2, dtype=float)
    scales = jnp.array([spike_width, slab_width], dtype=float)

    low = truncation
    high = 1.0 - truncation

    component_dist = dist.TruncatedNormal(
        loc=locs,
        scale=scales,
        low=low,
        high=high,
    )

    mix = dist.CategoricalProbs(jnp.array([spike_prob, 1.0 - spike_prob]))
    return dist.MixtureSameFamily(mix, component_dist)


def extended_model(
    data: Int[Array, "n_samples n_genes"] = None,
    posterior: bool = True,
    n_genes=None,
    dirichlet_prior_weight: float = 2.0,
    spike_prob: float = 0.999,
    spike_width: float = 0.01,
    slab_width: float = 1.0,
    boundary_truncation: float = 0.005,
    impurity_scale: float = 0.05,
    impurity_high: float = 0.2,
    fpr_loc: float = 0.0,
    fpr_scale: float = 0.05,
    fpr_high: float = 0.2,
    fnr_loc: float = 0.0,
    fnr_scale: float = 0.05,
    fnr_high: float = 0.2,
    independent_high: float = 1.0,
    use_preprocessing: bool = True,
):
    """Builds the extended model.

    Args:
        posterior: if True, samples from the posterior. Otherwise, from the prior
        use_preprocessing: whether to use O(2**G) artificial samples, rather than O(N)
            samples in sampling by the use of preprocessing.
    """
    if n_genes is None:
        n_genes = data.shape[1]

    # Sample the coverage
    spike_and_slab_prior = _truncated_spike_and_slab(
        truncation=boundary_truncation,
        spike_prob=spike_prob,
        spike_width=spike_width,
        slab_width=slab_width,
    )
    coverage = numpyro.sample("coverage", spike_and_slab_prior)

    # Partition of the exclusive components into different genes
    exclusive_mixture_weights = numpyro.sample(
        "exclusive_weights", dist.Dirichlet(dirichlet_prior_weight * jnp.ones(n_genes))
    )

    all_mixture_weights = numpyro.deterministic(
        "mixture_weights",
        jnp.concatenate(
            (
                # The 0th component: independent model
                jnp.full(fill_value=1.0 - coverage, shape=(1,)),
                # All the other components: exclusivity
                coverage * exclusive_mixture_weights,
            )
        ),
    )

    # Probabilities in the independent component
    probs_independent = numpyro.sample(
        "_component_independent",
        dist.Uniform(
            jnp.zeros(n_genes) + boundary_truncation,
            independent_high - boundary_truncation,
        ),
    )

    # Add impurity parameter and create probabilities for the exclusive components
    impurity = numpyro.sample(
        "impurity",
        dist.TruncatedNormal(
            0.0, impurity_scale, low=boundary_truncation, high=impurity_high
        ),
    )
    probs_exclusive = impurity + jnp.eye(n_genes) * (1.0 - impurity)

    # Create a matrix with probabilities for all the components
    components_noiseless = numpyro.deterministic(
        "components_noiseless",
        jnp.concatenate(
            (
                probs_independent[jnp.newaxis, :],
                probs_exclusive,
            ),
            axis=0,
        ),
    )

    # False positive and false negative rates
    fpr = numpyro.sample(
        "fpr",
        dist.TruncatedNormal(
            fpr_loc, fpr_scale, low=boundary_truncation, high=fpr_high
        ),
    )
    fnr = numpyro.sample(
        "fnr",
        dist.TruncatedNormal(
            fnr_loc, fnr_scale, low=boundary_truncation, high=fnr_high
        ),
    )

    components_noisy = numpyro.deterministic(
        "components_noisy",
        bmm.adjust_mixture_components_for_noise(
            mixture_components=components_noiseless,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
        ),
    )

    # If we don't sample from the posterior, we finish here
    if not posterior:
        return

    # If we want to sample from the posterior, we define the observational
    # distribution and optionally use some preprocessing
    dist_observed = _get_bernoulli_mixture(
        mixing_weights=all_mixture_weights,
        bernoulli_probs=components_noisy,
    )

    if not use_preprocessing:
        with numpyro.plate("n_patients", data.shape[0]):
            numpyro.sample("data", dist_observed, obs=data)
    else:
        genotypes_str = _all_genotypes_possible(n_genes)
        genotypes_template_array = jnp.asarray(
            [[int(bit) for bit in word] for word in genotypes_str]
        )

        counts_dict = {s: 0 for s in genotypes_str}
        for patient in _map_genotype_array_to_str(data):
            counts_dict[patient] += 1

        counts = np.zeros(2**n_genes, dtype=int)
        for i, gtype in enumerate(genotypes_str):
            counts[i] = counts_dict[gtype]

        numpyro.factor(
            "loglikehood",
            jnp.dot(counts, dist_observed.log_prob(genotypes_template_array)),
        )
