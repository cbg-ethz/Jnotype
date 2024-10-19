import jax
import jax.numpy as jnp
import jnotype.checks._statistics as st
import numpy.testing as npt

import pytest


@pytest.mark.parametrize("n_genes", [3, 4])
@pytest.mark.parametrize("n_samples", [2, 3, 5])
def test_genotypes_integers_inverse1(n_genes: int, n_samples: int) -> None:
    nums = jnp.arange(n_samples)

    genotypes = st.convert_integers_to_genotypes(nums, n_genes=n_genes)
    nums_ = st.convert_genotypes_to_integers(genotypes)
    npt.assert_allclose(nums, nums_)


@pytest.mark.parametrize("n_genes", [3, 4])
@pytest.mark.parametrize("n_samples", [2, 3, 5])
def test_genotypes_integers_inverse2(n_genes: int, n_samples: int) -> None:
    key = jax.random.PRNGKey(n_genes * n_samples + 5)
    genotypes = jax.random.bernoulli(key, p=0.5, shape=(n_samples, n_genes))

    nums = st.convert_genotypes_to_integers(genotypes)
    genotypes_ = st.convert_integers_to_genotypes(nums, n_genes=n_genes)

    npt.assert_allclose(genotypes, genotypes_)


@pytest.mark.parametrize("n_genes", [3, 4])
def test_genotypes_integers_inverse3_boundary(n_genes: int, n_samples: int = 5) -> None:
    genotypes1 = jnp.ones((n_samples, n_genes), dtype=int)
    genotypes2 = jnp.zeros((n_samples, n_genes), dtype=int)

    nums1 = st.convert_genotypes_to_integers(genotypes1)
    nums2 = st.convert_genotypes_to_integers(genotypes2)

    npt.assert_allclose(nums1, jnp.full(shape=(n_samples,), fill_value=2**n_genes - 1))
    npt.assert_allclose(nums2, jnp.zeros(shape=(n_samples,), dtype=int))
