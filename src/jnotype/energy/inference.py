import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Float
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import (
    Normal,
    Uniform,
    Beta,
    CategoricalProbs,
    MixtureGeneral,
)

from workflow import (
    create_symmetric_interaction_matrix,
    number_of_interactions_quadratic,
)

from _sampling import generate_all_binary_vectors


class InferenceModel:
    """Base class for Bayesian models in NumPyro."""

    def __call__(self, X: jnp.ndarray):
        return self.model(X)

    def model(self, X: jnp.ndarray):
        raise NotImplementedError("Subclasses must implement model(...)")

    @property
    def name(self) -> str:
        return self.__class__.__name__


class IsingSpikeAndSlabBayes(InferenceModel):
    """
    Bayesian Ising with spike-slab prior on a GxG matrix Theta, y in {0,1}.
    """

    def __init__(self, prior_sigma_max: Float):
        """
        prior_sigma_max: upper bound for the slab scale in Uniform(0, prior_sigma_max).
        """
        self.prior_sigma_max = prior_sigma_max

    def model(self, X: jnp.ndarray):
        """
        X: shape (N, G) in {0,1}, the observed binary data
        sample:
          - Theta_{ii} ~ Normal(0, sigma^2)   (diagonal)
          - Theta_{ij} ~ 0 w.p. pi, else Normal(0, sigma^2)  for i<j
        under the Ising model, p(y) ~ exp(y^T Theta y) / Z
        """
        N, G = X.shape

        # Hyperpriors
        pi = numpyro.sample("pi", Beta(1, 5))  # Bernoulli "spike" prob
        sigma = numpyro.sample("sigma", Uniform(0, self.prior_sigma_max))  # slab scale

        # sample the interaction matrix theta
        with numpyro.plate("diag_plate", G):
            diag_vals = numpyro.sample("diag_vals", Normal(0, sigma))

        mix_probs = jnp.array([pi, 1 - pi])
        total_offdiag = number_of_interactions_quadratic(G)
        with numpyro.plate("offdiag_plate", total_offdiag):
            # We'll use MixtureGeneral => CategoricalProbs(mix_probs), and list of 2 Normals
            # => each sample => scalar
            mixture = MixtureGeneral(
                mixing_distribution=CategoricalProbs(probs=mix_probs),
                component_distributions=[
                    Normal(loc=0.0, scale=0.01),
                    Normal(loc=0.0, scale=sigma),
                ],
            )

            off_diag_vals = numpyro.sample("theta_offdiag", mixture)
        theta = create_symmetric_interaction_matrix(diag_vals, off_diag_vals)
        all_binary_vec = generate_all_binary_vectors(G)

        def energy_fn(yy):
            return yy.T @ theta @ yy

        energies = jax.vmap(energy_fn)(all_binary_vec)
        log_weights = -energies
        logZ = jax.scipy.special.logsumexp(log_weights)  # log of normalization constant

        data_energies = jax.vmap(lambda c: c.T @ theta @ c)(X)
        ll = -jnp.sum(data_energies) - N * logZ

        # 7) Combine into posterior log-prob
        numpyro.factor("ising_loglike", ll)


class InferenceEngine:
    """MCMC engine that runs NUTS on the given model."""

    def __init__(self, num_warmup, num_samples, random_seed=0):
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.rng_key = jax.random.PRNGKey(random_seed)

    def run(self, model: InferenceModel, X: jnp.ndarray):
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples)
        mcmc.run(self.rng_key, X=X)
        samples = mcmc.get_samples()

        print(f"\nResults for {model.name}:")
        mcmc.print_summary()
        return samples


if __name__ == "__main__":
    # Generate synthetic data
    rng_key = jax.random.PRNGKey(42)
    X = jnp.load("data/G_5/N_50/genotypes.npy")
    model = IsingSpikeAndSlabBayes(prior_sigma_max=5.0)
    engine = InferenceEngine(num_warmup=200, num_samples=500)
    samples = engine.run(model, X)
