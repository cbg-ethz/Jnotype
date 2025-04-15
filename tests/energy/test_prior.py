import jax.numpy as jnp
import jnotype.energy._prior as pr

import numpy.testing as npt


def test_create_matrix_symmetric_interaction() -> None:
    desired = jnp.array(
        [
            [1, 2, 3],
            [2, 5, 7],
            [3, 7, 8],
        ],
        dtype=float,
    )

    obtained = pr.create_symmetric_interaction_matrix(
        diagonal=jnp.array([1, 5, 8], dtype=float),
        offdiagonal=jnp.array([2, 3, 7], dtype=float),
    )

    npt.assert_allclose(desired, obtained)
