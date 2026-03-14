import jax
import jax.numpy as jnp
import jnotype._dirichlet as dp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC


class Model:
    def train(self, data: jax.Array, key):
        pass

    def sample_predictive(
        self,
        key,
        n_samples: int,
    ) -> jax.Array:
        pass


def sample_empirical_distribution_generator(data):
    n = data.shape[0]

    def sample_fn(key):
        idx = jax.random.randint(key, minval=0, maxval=n, shape=())
        return data[idx, ...]

    return sample_fn


def sample_polya_urn(
    key,
    n_samples: int,
    weight1: float,
    sample_fn1,
    weight2: float,
    sample_fn2,
) -> jax.Array:
    """
    Draw `n_samples` iid from H ~ Dirichlet(a P + b Q)
    using the Polya‐urn predictive scheme with a dict for atom lookup.
    """
    m = weight1 + weight2

    samples = []
    unique_atoms = []
    counts = []
    # map from atom‐tuple to its index in unique_atoms/counts
    idx_map = {}

    loop_key = key
    for _ in range(n_samples):
        # decide new vs. existing
        loop_key, key_flag = jax.random.split(loop_key)
        total = sum(counts)
        prob_new = m / (m + total)
        u = float(jax.random.uniform(key_flag, ()))

        if u < prob_new:
            # sample a fresh atom from aP + bQ
            loop_key, key_mix = jax.random.split(loop_key)
            v = float(jax.random.uniform(key_mix, ()))
            loop_key, key_s = jax.random.split(loop_key)
            if v < weight1 / m:
                atom = sample_fn1(key_s)
            else:
                atom = sample_fn2(key_s)
        else:
            # reinforce existing atom
            loop_key, key_choice = jax.random.split(loop_key)
            cs = jnp.array(counts, dtype=jnp.float32)
            probs = cs / cs.sum()
            idx = int(jax.random.choice(key_choice, len(counts), p=probs))
            atom = unique_atoms[idx]

        # cast atom to an integer tuple key
        atom_key = tuple(map(int, jnp.ravel(atom)))

        # lookup/update via dict
        if atom_key in idx_map:
            i = idx_map[atom_key]
            counts[i] += 1
        else:
            i = len(unique_atoms)
            unique_atoms.append(atom)
            counts.append(1)
            idx_map[atom_key] = i

        samples.append(atom)

    return jnp.stack(samples, axis=0)


class SimpleBootstrap(Model):
    def __init__(self):
        self._data = None
        self._N = None
        self._G = None

    def train(self, data: jax.Array, key):
        self._data = data
        self._N = data.shape[0]
        self._G = data.shape[1]

    def sample_predictive(
        self,
        key,
        n_samples=None,
    ):
        assert self._G is not None

        indices = jax.random.randint(key, shape=(n_samples,), minval=0, maxval=self._N)
        return jax.vmap(lambda i: self._data[i])(indices)


class DirichletPriorModel(Model):
    def __init__(self, alpha: float):
        self._data = None

        assert alpha > 0
        self._alpha = alpha

        self._N = None
        self._G = None

    def train(self, data: jax.Array, key):
        self._data = data
        self._N = data.shape[0]
        self._G = data.shape[1]

    def sample_predictive(
        self,
        key,
        n_samples,
    ):
        assert self._G is not None
        G: int = self._G

        def sample_uniform(k):
            y = jax.random.bernoulli(k, p=0.5, shape=(G,))
            return jnp.array(y, dtype=self._data.dtype)

        assert self._N is not None
        N: int = self._N

        return sample_polya_urn(
            key=key,
            n_samples=n_samples,
            weight1=self._alpha,
            weight2=N,
            sample_fn1=sample_uniform,
            sample_fn2=sample_empirical_distribution_generator(self._data),
        )


def ll_fn(theta, y):
    return jnp.sum(y * jnp.log(theta) + (1 - y) * jnp.log1p(-theta))


class ParametricModel(Model):
    def __init__(
        self,
        n_warmup: int = 1000,
        n_chains: int = 4,
        n_mcmc_samples: int = 1000,
    ):
        self._n_warmup = n_warmup
        self._n_chains = n_chains
        self._n_mcmc_samples = n_mcmc_samples

        self._dtype = None

        self.mcmc = None

    def train(self, data, key):
        n_genes = data.shape[-1]

        loglike_multinomial = dp.construct_multinomial_loglikelihood(data, ll_fn)

        self._dtype = data.dtype

        def multinomial_model():
            conc1 = numpyro.sample("conc1", dist.Uniform(1, 10))
            conc0 = numpyro.sample("conc0", dist.Uniform(1, 10))
            ps = numpyro.sample("probs", dist.Beta(conc1, conc0 * jnp.ones(n_genes)))

            numpyro.factor("loglikelihood", loglike_multinomial(ps))

        mcmc = MCMC(
            NUTS(multinomial_model),
            num_warmup=self._n_warmup,
            num_samples=self._n_mcmc_samples,
            num_chains=self._n_chains,
        )
        mcmc.run(key)
        self.mcmc = mcmc

    def sample_predictive(
        self,
        key,
        n_samples,
    ):
        key_sample_idx, key_multi = jax.random.split(key)

        sample_idx = jax.random.randint(
            key_sample_idx,
            shape=(),
            minval=0,
            maxval=self._n_mcmc_samples * self._n_chains,
        )

        ps = self.mcmc.get_samples()["probs"][sample_idx]

        def sample_fn(k):
            y = jax.random.bernoulli(k, p=ps)
            return jnp.array(y, dtype=self._dtype)

        samples = jax.vmap(sample_fn)(jax.random.split(key_multi, n_samples))
        return jnp.stack(samples, axis=0)


class AdjustableModel(Model):
    def __init__(
        self,
        n_warmup: int = 1000,
        n_chains: int = 4,
        n_mcmc_samples: int = 1000,
    ):
        self._n_warmup = n_warmup
        self._n_chains = n_chains
        self._n_mcmc_samples = n_mcmc_samples

        self._data = None
        self._dtype = None

        self.mcmc = None

    def train(self, data, key):
        n_genes = data.shape[-1]

        self._data = data
        self._dtype = data.dtype

        loglike_dp = dp.construct_dirichlet_multinomial_loglikelihood(data, ll_fn)

        def dp_model():
            alpha = numpyro.sample("alpha", dist.HalfCauchy(5.0))
            conc1 = numpyro.sample("conc1", dist.Uniform(1, 10))
            conc0 = numpyro.sample("conc0", dist.Uniform(1, 10))
            ps = numpyro.sample("probs", dist.Beta(conc1, conc0 * jnp.ones(n_genes)))
            numpyro.factor("loglikelihood", loglike_dp(ps, alpha))

        mcmc = MCMC(
            NUTS(dp_model),
            num_warmup=self._n_warmup,
            num_samples=self._n_mcmc_samples,
            num_chains=self._n_chains,
        )
        mcmc.run(key)
        self.mcmc = mcmc

    def sample_predictive(
        self,
        key,
        n_samples,
    ):
        key_sample_idx, key_polya = jax.random.split(key)

        sample_idx = jax.random.randint(
            key_sample_idx,
            shape=(),
            minval=0,
            maxval=self._n_mcmc_samples * self._n_chains,
        )

        samples = self.mcmc.get_samples()
        alpha = samples["alpha"][sample_idx]
        ps = samples["probs"][sample_idx]

        def sample_fn(k):
            y = jax.random.bernoulli(k, p=ps)
            return jnp.array(y, dtype=self._dtype)

        N: int = self._data.shape[0]

        return sample_polya_urn(
            key=key_polya,
            n_samples=n_samples,
            weight1=alpha,
            weight2=N,
            sample_fn1=sample_fn,
            sample_fn2=sample_empirical_distribution_generator(self._data),
        )
