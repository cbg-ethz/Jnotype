# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import jax
import jax.numpy as jnp
import jnotype._dirichlet as dp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC


# %%
def ll_fn(theta, y):
    return jnp.sum(y * jnp.log(theta) + (1 - y) * jnp.log1p(-theta))


def ll_same_fn(theta, y):
    """Loglikelihood function corresponding to the independent model
    with the same parameter for each locus."""
    ps = jnp.ones(y.shape[-1], dtype=float) * 0.5
    return ll_fn(ps, y)


# %%
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

p_true = jnp.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
n_genes = len(p_true)
n_samples = 5000
data = jax.random.bernoulli(key, p=p_true, shape=(n_samples, len(p_true)))

# %%
data2 = [[0] * n_genes] * 8_000
data2 = jnp.asarray(data2)

data = jnp.concatenate([data, data2])

# %%
# data = jnp.asarray([[1, 1, 0, 0, 0]] * 300 + [[0, 0, 1, 1, 1]] * 200)

# %%
loglike_multinomial = dp.construct_multinomial_loglikelihood(data, ll_fn)


def multinomial_model():
    conc1 = numpyro.sample("conc1", dist.Uniform(1, 10))
    conc0 = numpyro.sample("conc0", dist.Uniform(1, 10))
    ps = numpyro.sample("probs", dist.Beta(conc1, conc0 * jnp.ones(n_genes)))

    numpyro.factor("loglikelihood", loglike_multinomial(ps))


loglike_dp = dp.construct_dirichlet_multinomial_loglikelihood(data, ll_fn)


def dp_model():
    alpha = numpyro.sample("alpha", dist.HalfCauchy(5.0))
    conc1 = numpyro.sample("conc1", dist.Uniform(1, 10))
    conc0 = numpyro.sample("conc0", dist.Uniform(1, 10))
    ps = numpyro.sample("probs", dist.Beta(conc1, conc0 * jnp.ones(n_genes)))

    numpyro.factor("loglikelihood", loglike_dp(ps, alpha))


loglike_dummy = dp.construct_dirichlet_multinomial_loglikelihood(data, ll_same_fn)


def dummy_dp_model():
    alpha = numpyro.sample("alpha", dist.HalfCauchy(5.0))
    # p = numpyro.sample("probs", dist.Uniform(0, 1))
    numpyro.factor("loglikelihood", loglike_dummy(0.5, alpha))


# %%
mcmc = MCMC(NUTS(multinomial_model), num_warmup=1000, num_samples=1000, num_chains=4)
key, subkey = jax.random.split(key)
mcmc.run(subkey)
mcmc.print_summary()

# %%
mcmc = MCMC(NUTS(dp_model), num_warmup=1000, num_samples=1000, num_chains=4)
key, subkey = jax.random.split(key)
mcmc.run(subkey)
mcmc.print_summary()

# %%
mcmc = MCMC(NUTS(dummy_dp_model), num_warmup=1000, num_samples=1000, num_chains=4)
key, subkey = jax.random.split(key)
mcmc.run(subkey)
mcmc.print_summary()

# %%
loglike_multinomial(p_true)

# %%
loglike_multinomial(1.9 * p_true)

# %%
