import jax
import jax.numpy as jnp
import jnotype._dirichlet as dp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC


class Model:
    def train(self, data: jax.Array):
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
    key: jax.random.KeyArray,
    n_samples: int,
    weight1: float,
    sample_fn1,  # key -> Array
    weight2: float,
    sample_fn2,  # key -> Array
) -> jax.Array:
    """
    Draw `n_samples` iid from a single random measure
      H ~ Dirichlet(a P + b Q)
    using the Polya‐urn predictive scheme.

    Returns:
      samples: Array of shape (n_samples, *event_shape)
    """
    m = weight1 + weight2
    samples = []
    unique_atoms = []  # list of JAX arrays seen so far
    counts = []  # corresponding integer counts

    # we'll thread one key through the loop
    loop_key = key

    for _ in range(n_samples):
        # split off one key for deciding new vs existing
        loop_key, key_flag = jax.random.split(loop_key)
        total_count = sum(counts)  # Python int
        prob_new = m / (m + total_count)  # float

        # draw uniform to decide
        u = float(jax.random.uniform(key_flag, ()))

        if u < prob_new:
            # --- draw a "new" atom from aP + bQ ---
            loop_key, key_mix = jax.random.split(loop_key)
            v = float(jax.random.uniform(key_mix, ()))
            if v < weight1 / m:
                # sample from P
                key_mix, key_s = jax.random.split(key_mix)
                atom = sample_fn1(key_s)
            else:
                # sample from Q
                key_mix, key_s = jax.random.split(key_mix)
                atom = sample_fn2(key_s)
        else:
            # --- reinforce an existing atom ---
            loop_key, key_choice = jax.random.split(loop_key)
            cs = jnp.array(counts, dtype=jnp.float32)
            probs = cs / cs.sum()
            idx = int(jax.random.choice(key_choice, len(counts), p=probs))
            atom = unique_atoms[idx]

        # record the draw
        # check if we've seen it before
        found = False
        for i, a in enumerate(unique_atoms):
            if jnp.array_equal(a, atom):
                counts[i] += 1
                found = True
                break
        if not found:
            unique_atoms.append(atom)
            counts.append(1)

        samples.append(atom)

    # stack along axis 0
    return jnp.stack(samples, axis=0)


class DirichletPriorModel(Model):
    def __init__(self, alpha: float):
        self._data = None

        assert alpha > 0
        self._alpha = alpha

        self._N = None
        self._G = None

    def train(self, data: jax.Array):
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

    def train(self, data):
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
