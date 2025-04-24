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
n_samples = 1000
data = jax.random.bernoulli(key, p=p_true, shape=(n_samples, len(p_true)))

# %%
data2 = [[1] * n_genes] * 200
data2 = jnp.asarray(data2)

data = jnp.concatenate([data, data2])

# %%
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC


def reparam_ps(x):
    return x  #  * p_true


loglike_multinomial = dp.construct_multinomial_loglikelihood(data, ll_fn)


def multinomial_model():
    conc1 = numpyro.sample("conc1", dist.Uniform(1, 10))
    conc0 = numpyro.sample("conc0", dist.Uniform(1, 10))
    ps = numpyro.sample("probs", dist.Beta(conc1, conc0 * jnp.ones(n_genes)))

    numpyro.factor("loglikelihood", loglike_multinomial(reparam_ps(ps)))


loglike_dp = dp.construct_dirichlet_multinomial_loglikelihood(data, ll_fn)


def dp_model():
    alpha = numpyro.sample("alpha", dist.HalfCauchy(5.0))
    conc1 = numpyro.sample("conc1", dist.Uniform(1, 10))
    conc0 = numpyro.sample("conc0", dist.Uniform(1, 10))
    ps = numpyro.sample("probs", dist.Beta(conc1, conc0 * jnp.ones(n_genes)))

    numpyro.factor("loglikelihood", loglike_dp(reparam_ps(ps), alpha))


loglike_perturbed = dp.construct_perturbed_loglikelihood(data, ll_fn)


def perturbed_model():
    alpha = numpyro.sample("alpha", dist.HalfCauchy(5.0))
    eta_conc = 10.0
    eta = numpyro.sample("eta", dist.Beta(eta_conc * 0.2, eta_conc * 0.8))
    # eta = numpyro.sample("eta", dist.TruncatedNormal(loc=0.01, scale=0.2, low=0.001, high=1-0.001))

    conc1 = numpyro.sample("conc1", dist.Uniform(1, 10))
    conc0 = numpyro.sample("conc0", dist.Uniform(1, 10))
    ps = numpyro.sample("probs", dist.Beta(conc1, conc0 * jnp.ones(n_genes)))

    numpyro.factor("loglikelihood", loglike_perturbed(reparam_ps(ps), alpha, eta))


loglike_dummy = dp.construct_dirichlet_multinomial_loglikelihood(data, ll_same_fn)


def dummy_dp_model():
    alpha = numpyro.sample("alpha", dist.HalfCauchy(5.0))
    # p = numpyro.sample("probs", dist.Uniform(0, 1))
    numpyro.factor("loglikelihood", loglike_dummy(0.5, alpha))


# %% [markdown]
# ## Inference in the parametric model

# %%
mcmc = MCMC(NUTS(multinomial_model), num_warmup=1000, num_samples=1000, num_chains=4)
key, subkey = jax.random.split(key)
mcmc.run(subkey)
mcmc.print_summary()

# %% [markdown]
# ## Inference in the mixture of Dirichlet processes model

# %%
mcmc = MCMC(NUTS(dp_model), num_warmup=1000, num_samples=1000, num_chains=4)
key, subkey = jax.random.split(key)
mcmc.run(subkey)
mcmc.print_summary()

# %% [markdown]
# ## Inference in the perturbed model

# %%
mcmc = MCMC(NUTS(perturbed_model), num_warmup=1000, num_samples=1000, num_chains=4)
key, subkey = jax.random.split(key)
mcmc.run(subkey)
mcmc.print_summary()

# %% [markdown]
# ## Inference in the dummy model

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
