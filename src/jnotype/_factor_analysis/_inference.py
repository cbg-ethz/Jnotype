"""Sampling from the posterior distribution."""

from typing import Callable

from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
from jax import random

from jnotype._csp import sample_csp_prior, sample_csp_gibbs
from jnotype._factor_analysis._gibbs_backend import (
    gibbs_sample_mixing,
    gibbs_sample_traits,
)

Sample = dict


def initial_sample(
    key,
    Y: Float[Array, "points observed"],
    sigma2: Float[Array, " observed"],
    max_traits: int,
    expected_occupied: float,
    prior_shape: float,
    prior_scale: float,
) -> Sample:
    """Initializes the first sample.

    Args:
        key: JAX PRNG key
        Y: observed data
        sigma2: noise variance for each observed dimension,
          which is assumed to be known
        max_traits: maximum number of traits
        expected_occupied: expected number of traits
        prior_shape: shape parameter of the trait variances prior
        prior_scale: scale parameter of the trait variances prior

    Note:
        Currently `sigma2` is not sampled, but just copied from the input argument.
    """
    n_points, n_observed = Y.shape[0], Y.shape[1]
    key, *subkeys = random.split(key, 10)

    csp_sample = sample_csp_prior(
        key=subkeys[0],
        k=max_traits,
        expected_occupied=expected_occupied,
        prior_shape=prior_shape,
        prior_scale=prior_scale,
    )

    eta = random.normal(subkeys[1], shape=(n_points, max_traits))
    # Make Lambda smaller and smaller in the initial sample
    variances_initial = jnp.exp(-jnp.arange(0, max_traits))
    lambd = (
        random.normal(subkeys[2], shape=(n_observed, max_traits))
        * jnp.sqrt(variances_initial)[None, :]
    )

    return {
        "eta": eta,
        "lambda": lambd,
        "sigma2": sigma2,
        "csp": csp_sample,
    }


def generate_sampling_step(
    Y,
    csp_shape: float = 2.0,
    csp_scale: float = 2.0,
    csp_theta_inf: float = 0.01,
    csp_expected: float = 5.0,
    jit_it: bool = True,
) -> Callable:
    """Creates a Gibbs Markov kernel
    of signature
      (key, sample) -> sample

    Args:
        Y: observed data
        csp_shape: shape parameter of the trait variances prior
        csp_scale: scale parameter of the trait variances prior
        csp_theta_inf: trait variance for inactive (shrunk) traits
        csp_expected: expected number of traits
        jit_it: whether to JIT-compile the kernel
    """

    def _sample_gibbs(
        key,
        sample,
    ) -> dict:
        subkeys = random.split(key, 4)
        # Sample lambda
        lambd = gibbs_sample_mixing(
            subkeys[0],
            theta=sample["csp"]["variance"],
            sigma2=sample["sigma2"],
            eta=sample["eta"],
            Y=Y,
        )
        # We don't sample sigma2, just copy it
        sigma2 = sample["sigma2"]

        # Sample eta
        eta = gibbs_sample_traits(
            subkeys[2],
            lambd=lambd,
            sigma2=sigma2,
            Y=Y,
        )

        # Sample CSP parameters
        csp = sample_csp_gibbs(
            subkeys[3],
            coefficients=lambd,
            structure=jnp.ones_like(lambd),
            omega=sample["csp"]["omega"],
            expected_occupied=csp_expected,
            prior_shape=csp_shape,
            prior_scale=csp_scale,
            theta_inf=csp_theta_inf,
        )

        return {
            "eta": eta,
            "lambda": lambd,
            "sigma2": sigma2,
            "csp": csp,
        }

    if jit_it:
        return jax.jit(_sample_gibbs)
    else:
        return _sample_gibbs
