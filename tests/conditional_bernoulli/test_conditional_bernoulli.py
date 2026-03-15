from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

import jnotype.conditional_bernoulli as cb


def _enumerate_binary_vectors(g: int, n: int) -> np.ndarray:
    vectors = np.asarray(list(product([0, 1], repeat=g)), dtype=int)
    return vectors[vectors.sum(axis=1) == n]


def _bruteforce_logZ(log_theta: np.ndarray, n: int) -> float:
    if n == 0:
        return 0.0
    ys = _enumerate_binary_vectors(g=log_theta.shape[0], n=n)
    terms = ys @ log_theta
    m = float(np.max(terms))
    return m + float(np.log(np.sum(np.exp(terms - m))))


def test_calculate_logZ_matches_bruteforce() -> None:
    g = 5
    log_theta = np.array([0.4, -0.2, 0.1, 1.0, -0.7])

    logZ = np.asarray(cb.calculate_logZ(jnp.asarray(log_theta), n_max=g))
    assert logZ.shape == (g + 1,)

    for n in range(g + 1):
        brute = _bruteforce_logZ(log_theta, n)
        assert np.allclose(logZ[n], brute, atol=1e-6)


def test_logpmf_normalizes_to_one_for_fixed_n() -> None:
    g = 4
    n = 2
    log_theta = jnp.array([0.3, -0.1, 1.0, -0.4])

    ys = _enumerate_binary_vectors(g=g, n=n)
    log_probs = np.asarray(
        [
            cb.conditional_bernoulli_logpmf(jnp.asarray(y), log_theta=log_theta, n=n)
            for y in ys
        ]
    )

    assert np.all(np.isfinite(log_probs))
    assert np.isclose(np.sum(np.exp(log_probs)), 1.0, atol=1e-6)

    wrong_y = jnp.array([1, 1, 1, 0], dtype=int)
    assert np.isneginf(
        float(cb.conditional_bernoulli_logpmf(wrong_y, log_theta=log_theta, n=n))
    )


def test_generate_loglikelihood_matches_direct_evaluation() -> None:
    key = jax.random.PRNGKey(1)
    g = 6
    n_samples = 20

    simplex = jnp.array([0.05, 0.1, 0.2, 0.15, 0.3, 0.2])
    log_theta = cb.simplex_to_log_theta(simplex)
    ns = jax.random.randint(key, shape=(n_samples,), minval=0, maxval=g + 1)
    ys = cb.sample_conditional_bernoulli(
        jax.random.PRNGKey(2), ns=ns, log_theta=log_theta
    )

    fn = cb.generate_loglikelihood(ys)
    ll_factory = float(fn(log_theta))
    ll_direct = float(cb.conditional_bernoulli_loglikelihood(ys, log_theta=log_theta))

    assert np.isclose(ll_factory, ll_direct, atol=1e-6)


def test_sampling_preserves_requested_number_of_ones() -> None:
    g = 7
    ns = jnp.array([0, 1, 2, 3, 6, 7], dtype=int)
    simplex = jnp.array([0.1, 0.05, 0.12, 0.18, 0.2, 0.15, 0.2])
    log_theta = cb.simplex_to_log_theta(simplex)

    ys = cb.sample_conditional_bernoulli(
        jax.random.PRNGKey(3), ns=ns, log_theta=log_theta
    )

    assert ys.shape == (ns.shape[0], g)
    np.testing.assert_array_equal(np.asarray(ys.sum(axis=-1)), np.asarray(ns))
    np.testing.assert_array_equal(np.asarray(ys[0]), np.zeros(g, dtype=int))
    np.testing.assert_array_equal(np.asarray(ys[-1]), np.ones(g, dtype=int))


def test_component_and_mixture_loglikelihood_match_bruteforce() -> None:
    ys = jnp.asarray(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )

    component_simplex = jnp.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    component_log_theta = cb.simplex_to_log_theta(component_simplex)

    mixing_logits = jnp.log(jnp.array([0.7, 0.3]))

    ll_matrix = np.asarray(
        cb.conditional_bernoulli_component_loglikelihood_matrix(
            ys=ys,
            component_log_theta=component_log_theta,
        )
    )

    # Brute-force component likelihood for validation
    brute_component = np.zeros_like(ll_matrix)
    for i, y in enumerate(np.asarray(ys)):
        n = int(y.sum())
        for k, log_theta in enumerate(np.asarray(component_log_theta)):
            brute_component[i, k] = y @ log_theta - _bruteforce_logZ(log_theta, n)

    assert np.allclose(ll_matrix, brute_component, atol=1e-6)

    ll_mixture = float(
        cb.conditional_bernoulli_mixture_loglikelihood(
            ys=ys,
            mixing_logits=mixing_logits,
            component_log_theta=component_log_theta,
        )
    )

    logw = np.asarray(jax.nn.log_softmax(mixing_logits))
    brute_mixture = 0.0
    for i in range(ys.shape[0]):
        terms = brute_component[i, :] + logw
        m = float(np.max(terms))
        brute_mixture += m + float(np.log(np.sum(np.exp(terms - m))))

    assert np.isclose(ll_mixture, brute_mixture, atol=1e-6)


def test_mixture_sampling_preserves_n() -> None:
    ns = jnp.array([0, 1, 2, 3, 4, 2, 1], dtype=int)

    component_simplex = jnp.array(
        [
            [0.6, 0.2, 0.1, 0.1],
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    mixing_logits = jnp.log(jnp.array([0.2, 0.5, 0.3]))

    ys, z = cb.sample_conditional_bernoulli_mixture(
        key=jax.random.PRNGKey(11),
        ns=ns,
        mixing_logits=mixing_logits,
        component_log_theta=cb.simplex_to_log_theta(component_simplex),
        return_assignments=True,
    )

    np.testing.assert_array_equal(np.asarray(ys.sum(axis=-1)), np.asarray(ns))
    assert ys.shape == (ns.shape[0], component_simplex.shape[1])
    assert z.shape == (ns.shape[0],)
    assert np.all((np.asarray(z) >= 0) & (np.asarray(z) < component_simplex.shape[0]))
