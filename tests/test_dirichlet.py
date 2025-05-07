import jax.numpy as jnp
import jnotype._dirichlet as dp
import pytest


def test_dirichlet_multinomial_reduces():
    x = jnp.array([101, 20, 4, 1])
    log_p = jnp.log(jnp.array([0.6, 0.05, 0.05, 0.29]))
    alpha = 1e8

    mult = dp.log_multinomial(x, log_p)
    dir_mul = dp.log_dirichlet_multinomial(x, log_p, jnp.log(alpha))

    assert dir_mul == pytest.approx(mult, rel=0.01)
