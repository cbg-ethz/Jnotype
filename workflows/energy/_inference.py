import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
from jaxtyping import Float, Array, Int
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import (
    Normal,
    Uniform,
    Beta,
    CategoricalProbs,
    MixtureGeneral,
)

from _utils import (
    create_symmetric_interaction_matrix,
    number_of_interactions_quadratic,
)

# TODO(allen): correctly import these like the rest of the workflow files
from jnotype.energy._sampling import generate_all_binary_vectors
from jnotype.energy._dfd import discrete_fisher_divergence
from _dfd_beta import *
import optax
from typing import Callable


class InferenceModel:
    """Base class for Bayesian models in NumPyro."""

    def __call__(self, X: Int[Array, "N G"]):
        return self.model(X)

    def model(self, X: Int[Array, "N G"]):
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

    def model(self, X: Int[Array, "N G"]):
        """
        X: observed genotypes
        sample:
          - Theta_{ii} ~ Normal(0, sigma^2)   (diagonal)
          - Theta_{ij} ~ 0 w.p. pi, else Normal(0, sigma^2)  for i<j
        under the Ising model, p(y) ~ exp(y^T Theta y) / Z
        """
        N, G = X.shape

        # Hyperpriors for spike probability and slab scale
        # pi = numpyro.sample("pi", Beta(1, 5))
        pi = 0.1

        # sigma = numpyro.sample(
        #     "sigma", Uniform(low=5, high=self.prior_sigma_max)  # type: ignore
        # )
        sigma = 10

        # sample the interaction matrix theta
        with numpyro.plate("diag_plate", G):
            diag_vals = numpyro.sample("diag_vals", Normal(0, 5))  # type: ignore

        mix_probs = jnp.array([pi, 1 - pi])
        total_offdiag = number_of_interactions_quadratic(G)
        with numpyro.plate("offdiag_plate", total_offdiag):
            mixture = MixtureGeneral(
                mixing_distribution=CategoricalProbs(probs=mix_probs),
                component_distributions=[
                    Normal(loc=0.0, scale=0.01),
                    Normal(loc=0.0, scale=sigma),  # type: ignore
                ],
            )

            off_diag_vals = numpyro.sample("theta_offdiag", mixture)  # type: ignore
        theta = create_symmetric_interaction_matrix(diag_vals, off_diag_vals)  # type: ignore
        all_binary_vec = generate_all_binary_vectors(G)

        def energy_fn(yy):
            return yy.T @ theta @ yy

        total_energies = jax.vmap(energy_fn)(all_binary_vec)
        logZ = jax.scipy.special.logsumexp(
            -total_energies
        )  # log of normalization constant

        data_energies = jax.vmap(lambda y: y.T @ theta @ y)(X)
        ll = -jnp.sum(data_energies) - N * logZ
        numpyro.factor("ising_loglike", ll)


class DFD(InferenceModel):

    def __init__(self, prior_sigma_max: Float):
        self.prior_sigma_max = prior_sigma_max

    def _gaussian_log_prior(self, diag_vals, off_diag_vals, sigma):
        """
        Compute the total log prior of sampled parameters under Normal(0, sigma)

        Args:
            diag_vals: array of shape [G]
            off_diag_vals: array of shape [G*(G-1)//2]
            sigma: scalar std dev of the prior

        Returns:
            total log prior (scalar)
        """
        prior_dist = Normal(0.0, sigma)
        logp_diag = prior_dist.log_prob(diag_vals).sum()
        logp_offdiag = prior_dist.log_prob(off_diag_vals).sum()
        return logp_diag + logp_offdiag

    def model(self, X: Int[Array, "N G"]):
        N, G = X.shape

        # prior theta
        num_diag, num_offdiag = G, number_of_interactions_quadratic(G)
        with numpyro.plate("diag_plate", num_diag):
            diag_vals = numpyro.sample("diag_vals", Normal(0, 5))  # type: ignore
        with numpyro.plate("offdiag_plate", num_offdiag):
            off_diag_vals = numpyro.sample(
                "off_diag_vals", Normal(0, self.prior_sigma_max)  # type: ignore
            )

        theta = create_symmetric_interaction_matrix(diag_vals, off_diag_vals)  # type: ignore

        # no need for normalization constant
        def log_q(yy):
            return -yy.T @ theta @ yy

        beta = 50
        dfd = discrete_fisher_divergence(log_q, X)
        numpyro.factor("dfd_loss", -beta * N * dfd)

        # theta_log_prior = self._gaussian_log_prior(
        #     diag_vals, off_diag_vals, self.prior_sigma_max
        # )
        # optimizer = optax.adam(learning_rate=1e-3)
        # key = jrandom.PRNGKey(42)
        # LEARNING_RATE = 0.01
        # OPT_STEPS = 200
        # B = 500
        # beta_key, loss_key = jrandom.split(key)
        # calculated_beta, loss_values = calibrate_beta(
        #     beta_key,
        #     X,
        #     B,
        #     dfd_loss_fn=discrete_fisher_divergence,
        #     self._gaussian_log_prior,
        #     theta,
        #     optimizer,
        #     OPT_STEPS,
        # )

        # # calculate the optimal beta
        # beta = optimal_beta(X, log_q, Normal(0, self.prior_sigma_max))


class InferenceEngine:
    """MCMC engine that runs NUTS on the given model."""

    def __init__(self, num_warmup, num_samples, num_chains, random_seed=0):
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.rng_key = jax.random.PRNGKey(random_seed)

    def run(self, model: InferenceModel, X: jnp.ndarray):
        kernel = NUTS(model)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )
        mcmc.run(self.rng_key, X=X)
        samples = mcmc.get_samples()

        print(f"\nResults for {model.name}:")
        mcmc.print_summary()
        return samples


if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(42)
    X = jnp.load("data/G_5/N_50/genotypes.npy")
    model = DFD(prior_sigma_max=5.0)
    engine = InferenceEngine(num_warmup=200, num_samples=500, num_chains=2)
    samples = engine.run(model, X)
