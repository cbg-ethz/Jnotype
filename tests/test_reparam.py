import jax
import jax.numpy as jnp
import jnotype._reparam as re
import numpy.testing as npt
import pytest

FUNCS = {
    "dot": lambda x: jnp.dot(x, 2 * x),
    "sum": lambda x: jnp.sum(x**2 + 3 * x - 2),
}


def test_array_conversions_simple():
    a = jnp.asarray([0, 1, 0])
    b = jnp.asarray([1, -1, 1])

    npt.assert_array_equal(re.array_to_01(b), a)
    npt.assert_array_equal(re.array_to_pos_neg(a), b)

    for f_name, f in FUNCS.items():
        print("Function: ", f_name)
        npt.assert_allclose(
            f(a),
            re.func_to_pos_neg(f)(b),
        )
        npt.assert_allclose(
            f(b),
            re.func_to_01(f)(a),
        )


@pytest.mark.parametrize("shape", [(30,), (10, 2)])
def test_array_conversions(shape, key=42):
    key = jax.random.PRNGKey(key)
    a = jax.random.bernoulli(key, p=0.3, shape=shape)
    a_ = (-1) ** a

    b = re.array_to_pos_neg(a)
    assert b.shape == a.shape

    npt.assert_array_equal(b, a_)
    npt.assert_array_equal(re.array_to_01(b), a)

    for f_name, f in FUNCS.items():
        f = jax.vmap(f)
        print("Function: ", f_name)
        npt.assert_allclose(
            f(a),
            re.func_to_pos_neg(f)(b),
        )
        npt.assert_allclose(
            f(b),
            re.func_to_01(f)(a),
        )


@pytest.mark.parametrize("input_list", [True, False])
def test_empirical_distribution_example(input_list: bool):
    vectors = [
        # ---
        jnp.asarray([1, 1, 0]),
        jnp.asarray([1, 1, 0]),
        # ---
        jnp.asarray([0, 1, 0]),
        jnp.asarray([0, 1, 0]),
        # ---
        jnp.asarray([1, 1, 1]),
        # ---
        jnp.asarray([0, 0, 0]),
    ]
    if not input_list:
        vectors = jnp.asarray(vectors)

    dist = re.empirical_binary_vector_distribution(vectors)
    assert dist.n_datapoints == 6
    assert dist.n_atoms == 4
    npt.assert_array_equal(dist.counts, jnp.asarray([2, 2, 1, 1]))
    npt.assert_allclose(dist.weights, jnp.asarray([2 / 6, 2 / 6, 1 / 6, 1 / 6]))

    atoms = jnp.asarray(
        [
            [1, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ]
    )

    npt.assert_array_equal(dist.atoms, atoms)


@pytest.mark.parametrize("shape", [(15, 1), (10, 2), (30, 2)])
@pytest.mark.parametrize("function_desc", FUNCS.items())
def test_summaries_empirical_distribution(shape, function_desc, key=42):
    key = jax.random.PRNGKey(key)
    data = jax.random.bernoulli(key, p=0.3, shape=shape)

    dist = re.empirical_binary_vector_distribution(data)

    name, function = function_desc
    print(name)
    f_out = jax.vmap(function)(data)

    assert pytest.approx(dist.calculate_function_mean(function)) == jnp.mean(f_out)
    assert pytest.approx(dist.calculate_function_sum(function)) == jnp.sum(f_out)
