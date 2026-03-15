import json
from itertools import permutations

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist

from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import Adam

import jnotype.conditional_bernoulli as cbmodel


matplotlib.use("agg")


# Complex demo defaults (requested scenario)
DEFAULT_CONFIG = {
    "run_name": "default",
    "n_samples": 200,
    "n_genes": 50,
    "k_x": 5,
    "k_n": 3,
    "n_warmup": 500,
    "n_posterior": 500,
    "n_ppc_draws": 250,
    "n_svi_steps": 4000,
    "svi_lr": 0.01,
    "inference_methods": ["nuts", "svi"],
    "seed_simulate": 20260315,
    "seed_inference": 20260316,
    "seed_ppc": 20260317,
    "mix_x_true": [0.24, 0.19, 0.21, 0.18, 0.18],
    "mix_n_true": [0.25, 0.45, 0.30],
    "n_means_true": [0.14, 0.45, 0.76],
    "n_kappas_true": [170.0, 220.0, 180.0],
}

CFG = {**DEFAULT_CONFIG, **config}

RUN_NAME_RAW = str(CFG["run_name"])
RUN_NAME = "".join(
    ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_"
    for ch in RUN_NAME_RAW
).strip("._")
if not RUN_NAME:
    raise ValueError(
        f"Invalid `run_name={RUN_NAME_RAW!r}`; must contain at least one valid char."
    )
WORKDIR = f"generated/cbmm/conditional-bernoulli-mixture-demo/{RUN_NAME}"
workdir: WORKDIR

N_SAMPLES = int(CFG["n_samples"])
N_GENES = int(CFG["n_genes"])
K_X = int(CFG["k_x"])
K_N = int(CFG["k_n"])
N_WARMUP = int(CFG["n_warmup"])
N_POSTERIOR = int(CFG["n_posterior"])
N_PPC_DRAWS = int(CFG["n_ppc_draws"])
N_SVI_STEPS = int(CFG["n_svi_steps"])
SVI_LR = float(CFG["svi_lr"])

SEED_SIMULATE = int(CFG["seed_simulate"])
SEED_INFERENCE = int(CFG["seed_inference"])
SEED_PPC = int(CFG["seed_ppc"])

METHODS = tuple(str(m).lower() for m in CFG["inference_methods"])
METHOD_SET = {"nuts", "svi"}
if not METHODS:
    raise ValueError("`inference_methods` must contain at least one method.")
unknown_methods = sorted(set(METHODS) - METHOD_SET)
if unknown_methods:
    raise ValueError(
        f"Unknown inference methods: {unknown_methods}. Supported: {sorted(METHOD_SET)}"
    )

METHOD_PATTERN = "|".join(sorted(set(METHODS)))
wildcard_constraints:
    method=METHOD_PATTERN

MIX_X_TRUE = jnp.asarray(CFG["mix_x_true"], dtype=float)
MIX_N_TRUE = jnp.asarray(CFG["mix_n_true"], dtype=float)
N_MEANS_TRUE = jnp.asarray(CFG["n_means_true"], dtype=float)
N_KAPPAS_TRUE = jnp.asarray(CFG["n_kappas_true"], dtype=float)

if K_X < 2:
    raise ValueError(f"Need at least two X-components, got k_x={K_X}.")
if MIX_X_TRUE.shape[0] != K_X:
    raise ValueError(f"`mix_x_true` length must be {K_X}.")
if K_N < 2:
    raise ValueError(f"Need at least two N-components, got k_n={K_N}.")
if MIX_N_TRUE.shape[0] != K_N:
    raise ValueError(f"`mix_n_true` length must be {K_N}.")
if N_MEANS_TRUE.shape[0] != K_N:
    raise ValueError(f"`n_means_true` length must be {K_N}.")
if N_KAPPAS_TRUE.shape[0] != K_N:
    raise ValueError(f"`n_kappas_true` length must be {K_N}.")


def alpha_beta_from_mean_kappa(means, kappas):
    alpha = means * kappas
    beta = (1.0 - means) * kappas
    return alpha, beta


def method_seed(base_seed, method, offset=0):
    offsets = {"nuts": 0, "svi": 10_000}
    return int(base_seed + offsets[method] + offset)


def build_simplex_theta_true(n_components, n_genes):
    grid = jnp.linspace(0.0, 1.0, n_genes)[None, :]
    centers = jnp.linspace(0.08, 0.92, n_components)[:, None]
    widths = jnp.linspace(0.06, 0.16, n_components)[:, None]

    primary = jnp.exp(-0.5 * ((grid - centers) / widths) ** 2)
    secondary_center = 1.0 - centers
    secondary_width = 0.65 * widths + 0.02
    secondary = 0.25 * jnp.exp(
        -0.5 * ((grid - secondary_center) / secondary_width) ** 2
    )

    baseline = 0.04 + 0.01 * jnp.cos(2.0 * jnp.pi * grid)
    raw = baseline + primary + secondary
    return raw / raw.sum(axis=1, keepdims=True)


def sample_n_mixture(key, n_samples, n_genes, mixing_probs, alpha, beta):
    key_z, key_n = jax.random.split(key)

    z = jax.random.categorical(key_z, jnp.log(mixing_probs), shape=(n_samples,))
    n_dist = dist.BetaBinomial(
        total_count=n_genes,
        concentration1=alpha[z],
        concentration0=beta[z],
    )
    ns = n_dist.sample(key_n)
    return jnp.asarray(ns, dtype=int), z


def cbmm_demo_model(Y):
    _, n_genes = Y.shape

    # -- Conditional Bernoulli part for X | N --
    mix_x = numpyro.sample("mix_x", dist.Dirichlet(2.0 * jnp.ones(K_X)))

    theta_logits = numpyro.sample(
        "theta_logits",
        dist.Normal(0.0, 1.0).expand([K_X, n_genes]).to_event(2),
    )
    log_theta = jax.nn.log_softmax(theta_logits, axis=-1)
    numpyro.deterministic("simplex_theta", jnp.exp(log_theta))

    # -- Flexible model for N using a mixture of Beta-Binomial distributions --
    mix_n = numpyro.sample("mix_n", dist.Dirichlet(4.0 * jnp.ones(K_N)))

    mean_gaps = numpyro.sample("n_mean_gaps", dist.Dirichlet(8.0 * jnp.ones(K_N + 1)))
    means_raw = jnp.cumsum(mean_gaps[:-1])
    means = numpyro.deterministic(
        "n_means",
        jnp.clip(means_raw, a_min=1e-3, a_max=1.0 - 1e-3),
    )

    kappas_raw = numpyro.sample(
        "n_kappas",
        dist.Gamma(40.0 * jnp.ones(K_N), 40.0 / 140.0 * jnp.ones(K_N)),
    )
    kappas = numpyro.deterministic(
        "n_kappas_clipped",
        jnp.clip(kappas_raw, 20.0, 400.0),
    )

    alpha_n = numpyro.deterministic("n_alpha", means * kappas + 1e-6)
    beta_n = numpyro.deterministic("n_beta", (1.0 - means) * kappas + 1e-6)

    n_obs = jnp.asarray(Y.sum(axis=-1), dtype=int)

    # Log-likelihood for N (indicators marginalized out)
    n_component_dist = dist.BetaBinomial(
        total_count=n_genes,
        concentration1=alpha_n[None, :],
        concentration0=beta_n[None, :],
    )
    logp_n_components = n_component_dist.log_prob(n_obs[:, None])
    log_mix_n = jnp.log(jnp.clip(mix_n, a_min=1e-30, a_max=1.0))
    ll_n = jnp.sum(logsumexp(logp_n_components + log_mix_n[None, :], axis=-1))

    # Log-likelihood for X | N (component indicators marginalized out)
    ll_x_components = cbmodel.conditional_bernoulli_component_loglikelihood_matrix(
        ys=Y,
        component_log_theta=log_theta,
        n_obs=n_obs,
    )
    log_mix_x = jnp.log(jnp.clip(mix_x, a_min=1e-30, a_max=1.0))
    ll_x = jnp.sum(logsumexp(ll_x_components + log_mix_x[None, :], axis=-1))

    numpyro.factor("loglikelihood_n", ll_n)
    numpyro.factor("loglikelihood_x", ll_x)


rule all:
    input:
        simulated="demo/simulated_data.npz",
        metadata="demo/simulation_metadata.json",
        diagnostics=expand(
            "demo/inference_diagnostics_{method}.json",
            method=METHODS,
        ),
        posterior=expand(
            "demo/posterior_ordered_{method}.npz",
            method=METHODS,
        ),
        ppc=expand(
            "demo/posterior_predictive_{method}.npz",
            method=METHODS,
        ),
        clustering_metrics=expand(
            "demo/clustering_metrics_{method}.json",
            method=METHODS,
        ),
        clustering_posteriors=expand(
            "demo/posterior_assignments_{method}.npz",
            method=METHODS,
        ),
        clustering_confusion=expand(
            "plots/{method}/clustering_confusion_matrix.pdf",
            method=METHODS,
        ),
        hist_n=expand(
            "plots/{method}/histogram_n.pdf",
            method=METHODS,
        ),
        hist_gene_counts=expand(
            "plots/{method}/histogram_gene_counts.pdf",
            method=METHODS,
        ),
        corr_matrix=expand(
            "plots/{method}/correlation_matrix.pdf",
            method=METHODS,
        ),
        corr_pairs_ranked=expand(
            "plots/{method}/correlation_pairs_ranked.pdf",
            method=METHODS,
        ),


rule simulate_conditional_bernoulli_mixture_data:
    output:
        npz="demo/simulated_data.npz",
        metadata="demo/simulation_metadata.json",
    run:
        key = jax.random.PRNGKey(SEED_SIMULATE)
        key_n, key_x = jax.random.split(key)

        simplex_theta_true = build_simplex_theta_true(K_X, N_GENES)
        mix_x_true = MIX_X_TRUE

        mix_n_true = MIX_N_TRUE
        means_true = N_MEANS_TRUE
        kappas_true = N_KAPPAS_TRUE
        alpha_true, beta_true = alpha_beta_from_mean_kappa(means_true, kappas_true)

        ns, z_n = sample_n_mixture(
            key=key_n,
            n_samples=N_SAMPLES,
            n_genes=N_GENES,
            mixing_probs=mix_n_true,
            alpha=alpha_true,
            beta=beta_true,
        )

        Y, z_x = cbmodel.sample_conditional_bernoulli_mixture(
            key=key_x,
            ns=ns,
            mixing_logits=jnp.log(mix_x_true),
            component_log_theta=jnp.log(simplex_theta_true),
            return_assignments=True,
        )

        np.savez_compressed(
            output.npz,
            Y=np.asarray(Y, dtype=np.int32),
            ns=np.asarray(ns, dtype=np.int32),
            z_x=np.asarray(z_x, dtype=np.int32),
            z_n=np.asarray(z_n, dtype=np.int32),
            mix_x_true=np.asarray(mix_x_true),
            simplex_theta_true=np.asarray(simplex_theta_true),
            mix_n_true=np.asarray(mix_n_true),
            means_true=np.asarray(means_true),
            kappas_true=np.asarray(kappas_true),
            alpha_true=np.asarray(alpha_true),
            beta_true=np.asarray(beta_true),
        )

        metadata = {
            "run_name": RUN_NAME,
            "n_samples": int(N_SAMPLES),
            "n_genes": int(N_GENES),
            "n_components_x": int(K_X),
            "n_components_n": int(K_N),
            "inference_methods": list(METHODS),
            "description": (
                "Conditional Bernoulli mixture demo with configurable complexity; "
                "X-components and N-mixture components are marginalized in inference."
            ),
        }
        with open(output.metadata, "w") as fh:
            json.dump(metadata, fh, indent=2)


rule infer_demo_model:
    input:
        data="demo/simulated_data.npz",
    output:
        posterior="demo/posterior_raw_{method}.npz",
        diagnostics="demo/inference_diagnostics_{method}.json",
    run:
        method = wildcards.method
        arr = np.load(str(input.data))
        Y = jnp.asarray(arr["Y"], dtype=int)

        if method == "nuts":
            seed = method_seed(SEED_INFERENCE, method)
            kernel = NUTS(cbmm_demo_model, target_accept_prob=0.88)
            mcmc = MCMC(
                kernel,
                num_warmup=N_WARMUP,
                num_samples=N_POSTERIOR,
                progress_bar=False,
            )
            mcmc.run(jax.random.PRNGKey(seed), Y=Y)
            samples = mcmc.get_samples()

            np.savez_compressed(
                output.posterior,
                **{k: np.asarray(v) for k, v in samples.items()},
            )

            diagnostics = {
                "method": method,
                "seed": int(seed),
                "n_warmup": int(N_WARMUP),
                "n_posterior": int(N_POSTERIOR),
            }
            extra = mcmc.get_extra_fields()
            if "accept_prob" in extra:
                diagnostics["mean_accept_prob"] = float(
                    np.mean(np.asarray(extra["accept_prob"]))
                )
            if "diverging" in extra:
                diagnostics["n_divergent"] = int(
                    np.sum(np.asarray(extra["diverging"]))
                )

        elif method == "svi":
            seed = method_seed(SEED_INFERENCE, method)

            guide = AutoMultivariateNormal(cbmm_demo_model)
            optimizer = Adam(SVI_LR)
            svi = SVI(cbmm_demo_model, guide, optimizer, loss=Trace_ELBO())
            result = svi.run(
                jax.random.PRNGKey(seed),
                N_SVI_STEPS,
                Y=Y,
                progress_bar=False,
            )

            posterior_latents = guide.sample_posterior(
                jax.random.PRNGKey(seed + 1),
                result.params,
                sample_shape=(N_POSTERIOR,),
            )

            theta_logits = posterior_latents["theta_logits"]
            simplex_theta = jax.nn.softmax(theta_logits, axis=-1)

            n_mean_gaps = posterior_latents["n_mean_gaps"]
            means_raw = jnp.cumsum(n_mean_gaps[..., :-1], axis=-1)
            n_means = jnp.clip(means_raw, a_min=1e-3, a_max=1.0 - 1e-3)

            n_kappas_raw = posterior_latents["n_kappas"]
            n_kappas_clipped = jnp.clip(n_kappas_raw, 20.0, 400.0)
            n_alpha = n_means * n_kappas_clipped + 1e-6
            n_beta = (1.0 - n_means) * n_kappas_clipped + 1e-6

            samples = {k: np.asarray(v) for k, v in posterior_latents.items()}
            samples["simplex_theta"] = np.asarray(simplex_theta)
            samples["n_means"] = np.asarray(n_means)
            samples["n_kappas_clipped"] = np.asarray(n_kappas_clipped)
            samples["n_alpha"] = np.asarray(n_alpha)
            samples["n_beta"] = np.asarray(n_beta)

            np.savez_compressed(output.posterior, **samples)

            losses = np.asarray(result.losses, dtype=float)
            diagnostics = {
                "method": method,
                "seed": int(seed),
                "n_svi_steps": int(N_SVI_STEPS),
                "svi_lr": float(SVI_LR),
                "elbo_final": float(losses[-1]),
                "elbo_min": float(np.min(losses)),
                "elbo_last100_mean": float(np.mean(losses[-100:])),
            }

        else:
            raise ValueError(f"Unsupported method: {method}")

        with open(output.diagnostics, "w") as fh:
            json.dump(diagnostics, fh, indent=2)


rule order_component_labels_posthoc:
    input:
        posterior="demo/posterior_raw_{method}.npz",
    output:
        posterior="demo/posterior_ordered_{method}.npz",
    run:
        arr = np.load(str(input.posterior))
        samples = {k: arr[k] for k in arr.files}

        simplex_theta = samples["simplex_theta"]  # (S, K_X, G)
        mix_x = samples["mix_x"]  # (S, K_X)

        # Ordering score: expected feature index under each component simplex.
        g = simplex_theta.shape[-1]
        score = np.einsum("skg,g->sk", simplex_theta, np.arange(g, dtype=float))
        order = np.argsort(score, axis=1)

        samples["mix_x"] = np.take_along_axis(mix_x, order, axis=1)
        samples["simplex_theta"] = np.take_along_axis(
            simplex_theta,
            order[:, :, None],
            axis=1,
        )

        if "theta_logits" in samples:
            samples["theta_logits"] = np.take_along_axis(
                samples["theta_logits"],
                order[:, :, None],
                axis=1,
            )

        np.savez_compressed(output.posterior, **samples)


rule evaluate_component_assignments:
    input:
        data="demo/simulated_data.npz",
        posterior="demo/posterior_ordered_{method}.npz",
    output:
        metrics="demo/clustering_metrics_{method}.json",
        posteriors="demo/posterior_assignments_{method}.npz",
        confusion_plot="plots/{method}/clustering_confusion_matrix.pdf",
    run:
        method = wildcards.method
        data = np.load(str(input.data))
        post = np.load(str(input.posterior))

        if "z_x" not in data.files:
            raise ValueError(
                "Simulated data file does not contain true X-component labels `z_x`."
            )

        if K_X > 8:
            raise ValueError(
                "Current permutation-based label alignment is capped at K_X <= 8."
            )

        Y = np.asarray(data["Y"], dtype=np.int32)
        z_true = np.asarray(data["z_x"], dtype=np.int32)
        ns = np.asarray(Y.sum(axis=-1), dtype=np.int32)

        mix_x = np.asarray(post["mix_x"], dtype=float)
        simplex_theta = np.asarray(post["simplex_theta"], dtype=float)

        n_draws = mix_x.shape[0]
        n_samples = Y.shape[0]
        n_components = mix_x.shape[1]

        post_probs = np.zeros((n_samples, n_components), dtype=float)
        y_jax = jnp.asarray(Y, dtype=int)
        ns_jax = jnp.asarray(ns, dtype=int)

        for d in range(n_draws):
            ll = cbmodel.conditional_bernoulli_component_loglikelihood_matrix(
                ys=y_jax,
                component_log_theta=jnp.log(jnp.asarray(simplex_theta[d])),
                n_obs=ns_jax,
            )
            ll = np.asarray(ll)
            logits = ll + np.log(np.clip(mix_x[d], 1e-30, 1.0))[None, :]
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            post_probs += probs

        post_probs /= float(n_draws)
        pred_raw = post_probs.argmax(axis=1)

        confusion_raw = np.zeros((n_components, n_components), dtype=np.int32)
        np.add.at(confusion_raw, (z_true, pred_raw), 1)

        best_perm = tuple(range(n_components))
        best_trace = -1
        for perm in permutations(range(n_components)):
            trace = int(np.trace(confusion_raw[:, perm]))
            if trace > best_trace:
                best_trace = trace
                best_perm = perm

        inv_perm = np.empty(n_components, dtype=np.int32)
        for new_label, old_label in enumerate(best_perm):
            inv_perm[old_label] = new_label

        pred_aligned = inv_perm[pred_raw]
        probs_aligned = post_probs[:, best_perm]
        confusion_aligned = confusion_raw[:, best_perm]

        correct = pred_aligned == z_true
        max_prob = probs_aligned.max(axis=1)
        entropy = -np.sum(
            probs_aligned * np.log(np.clip(probs_aligned, 1e-30, 1.0)),
            axis=1,
        )
        entropy_norm = entropy / np.log(float(n_components))

        onehot_true = np.eye(n_components, dtype=float)[z_true]
        brier = np.mean(np.sum((probs_aligned - onehot_true) ** 2, axis=1))

        metrics = {
            "method": method,
            "n_samples": int(n_samples),
            "n_components": int(n_components),
            "best_label_permutation_old_to_new": [int(v) for v in best_perm],
            "accuracy_raw": float(np.mean(pred_raw == z_true)),
            "accuracy_aligned": float(np.mean(correct)),
            "mean_max_posterior_prob": float(np.mean(max_prob)),
            "mean_max_posterior_prob_correct": float(
                np.mean(max_prob[correct]) if np.any(correct) else np.nan
            ),
            "mean_max_posterior_prob_incorrect": float(
                np.mean(max_prob[~correct]) if np.any(~correct) else np.nan
            ),
            "mean_entropy_normalized": float(np.mean(entropy_norm)),
            "brier_score": float(brier),
        }

        with open(output.metrics, "w") as fh:
            json.dump(metrics, fh, indent=2)

        np.savez_compressed(
            output.posteriors,
            z_true=z_true,
            pred_raw=pred_raw,
            pred_aligned=pred_aligned,
            posterior_probs_raw=post_probs,
            posterior_probs_aligned=probs_aligned,
            confusion_raw=confusion_raw,
            confusion_aligned=confusion_aligned,
            best_perm=np.asarray(best_perm, dtype=np.int32),
            max_posterior_prob=max_prob,
            normalized_entropy=entropy_norm,
        )

        row_totals = np.clip(confusion_aligned.sum(axis=1, keepdims=True), 1, None)
        confusion_row_norm = confusion_aligned / row_totals

        fig, ax = plt.subplots(figsize=(4.8, 3.9), dpi=250)
        im = ax.imshow(confusion_row_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")

        for i in range(n_components):
            for j in range(n_components):
                count = confusion_aligned[i, j]
                frac = confusion_row_norm[i, j]
                text_color = "white" if frac > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{count}\\n({frac:.2f})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

        ticks = np.arange(n_components)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"C{k}" for k in ticks])
        ax.set_yticklabels([f"C{k}" for k in ticks])
        ax.set_xlabel("Predicted component (aligned)")
        ax.set_ylabel("True component")
        ax.set_title(
            f"{method.upper()} confusion (acc={metrics['accuracy_aligned']:.3f})"
        )
        fig.tight_layout()
        fig.savefig(str(output.confusion_plot), bbox_inches="tight")


rule posterior_predictive_demo:
    input:
        data="demo/simulated_data.npz",
        posterior="demo/posterior_ordered_{method}.npz",
    output:
        ppc="demo/posterior_predictive_{method}.npz",
    run:
        method = wildcards.method
        data = np.load(str(input.data))
        post = np.load(str(input.posterior))

        Y_obs = np.asarray(data["Y"], dtype=np.int32)
        n_samples, n_genes = Y_obs.shape

        mix_x = np.asarray(post["mix_x"])
        simplex_theta = np.asarray(post["simplex_theta"])
        mix_n = np.asarray(post["mix_n"])
        alpha_n = np.asarray(post["n_alpha"])
        beta_n = np.asarray(post["n_beta"])

        n_draws_available = mix_x.shape[0]
        n_draws = min(N_PPC_DRAWS, n_draws_available)

        rng = np.random.default_rng(method_seed(SEED_PPC, method))
        draw_indices = rng.choice(n_draws_available, size=n_draws, replace=False)

        ns_ppc = np.zeros((n_draws, n_samples), dtype=np.int32)
        y_ppc = np.zeros((n_draws, n_samples, n_genes), dtype=np.int32)

        key = jax.random.PRNGKey(method_seed(SEED_PPC, method, offset=1))
        draw_keys = jax.random.split(key, n_draws)

        for d, (draw_idx, draw_key) in enumerate(zip(draw_indices, draw_keys)):
            key_n, key_y = jax.random.split(draw_key)

            ns, _ = sample_n_mixture(
                key=key_n,
                n_samples=n_samples,
                n_genes=n_genes,
                mixing_probs=jnp.asarray(mix_n[draw_idx]),
                alpha=jnp.asarray(alpha_n[draw_idx]),
                beta=jnp.asarray(beta_n[draw_idx]),
            )

            y_draw = cbmodel.sample_conditional_bernoulli_mixture(
                key=key_y,
                ns=ns,
                mixing_logits=jnp.log(jnp.asarray(mix_x[draw_idx])),
                component_log_theta=jnp.log(jnp.asarray(simplex_theta[draw_idx])),
            )

            ns_ppc[d, :] = np.asarray(ns, dtype=np.int32)
            y_ppc[d, :, :] = np.asarray(y_draw, dtype=np.int32)

        np.savez_compressed(
            output.ppc,
            draw_indices=draw_indices,
            ns_ppc=ns_ppc,
            y_ppc=y_ppc,
        )


rule plot_histogram_n:
    input:
        data="demo/simulated_data.npz",
        ppc="demo/posterior_predictive_{method}.npz",
    output:
        pdf="plots/{method}/histogram_n.pdf",
    run:
        method = wildcards.method
        data = np.load(str(input.data))
        ppc = np.load(str(input.ppc))

        Y_obs = np.asarray(data["Y"], dtype=np.int32)
        ns_obs = np.asarray(Y_obs.sum(axis=-1), dtype=np.int32)
        ns_ppc = np.asarray(ppc["ns_ppc"], dtype=np.int32)

        n_genes = Y_obs.shape[1]
        bin_edges = np.arange(-0.5, n_genes + 1.5)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        obs_hist, _ = np.histogram(ns_obs, bins=bin_edges, density=True)

        ppc_hists = np.asarray(
            [
                np.histogram(ns_ppc[i], bins=bin_edges, density=True)[0]
                for i in range(ns_ppc.shape[0])
            ]
        )

        q005 = np.quantile(ppc_hists, 0.005, axis=0)
        q025 = np.quantile(ppc_hists, 0.025, axis=0)
        q05 = np.quantile(ppc_hists, 0.05, axis=0)
        q25 = np.quantile(ppc_hists, 0.25, axis=0)
        q50 = np.quantile(ppc_hists, 0.50, axis=0)
        q75 = np.quantile(ppc_hists, 0.75, axis=0)
        q95 = np.quantile(ppc_hists, 0.95, axis=0)
        q975 = np.quantile(ppc_hists, 0.975, axis=0)
        q995 = np.quantile(ppc_hists, 0.995, axis=0)

        fig, ax = plt.subplots(figsize=(6.2, 3.7), dpi=250)
        ax.fill_between(
            centers,
            q005,
            q995,
            step="mid",
            alpha=0.08,
            color="steelblue",
            label="Posterior predictive 99% CI",
        )
        ax.fill_between(
            centers,
            q025,
            q975,
            step="mid",
            alpha=0.12,
            color="steelblue",
            label="Posterior predictive 95% CI",
        )
        ax.fill_between(
            centers,
            q05,
            q95,
            step="mid",
            alpha=0.20,
            color="steelblue",
            label="Posterior predictive 90% CI",
        )
        ax.fill_between(
            centers,
            q25,
            q75,
            step="mid",
            alpha=0.33,
            color="steelblue",
            label="Posterior predictive 50% CI",
        )
        ax.step(
            centers,
            q50,
            where="mid",
            color="steelblue",
            linewidth=2,
            label="Posterior predictive median",
        )
        ax.step(
            centers,
            obs_hist,
            where="mid",
            color="darkorange",
            linewidth=2,
            label="Observed",
        )

        ax.set_xlabel("N = number of ones per sample")
        ax.set_ylabel("Density")
        ax.set_title(f"Histogram of N ({method.upper()})")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(str(output.pdf))


rule plot_histogram_gene_counts:
    input:
        data="demo/simulated_data.npz",
        ppc="demo/posterior_predictive_{method}.npz",
    output:
        pdf="plots/{method}/histogram_gene_counts.pdf",
    run:
        method = wildcards.method
        data = np.load(str(input.data))
        ppc = np.load(str(input.ppc))

        Y_obs = np.asarray(data["Y"], dtype=np.int32)
        y_ppc = np.asarray(ppc["y_ppc"], dtype=np.int32)

        n_samples = Y_obs.shape[0]
        obs_gene_counts = Y_obs.sum(axis=0)

        bin_edges = np.linspace(-0.5, n_samples + 0.5, 26)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        obs_hist, _ = np.histogram(obs_gene_counts, bins=bin_edges, density=True)

        ppc_hists = np.asarray(
            [
                np.histogram(y_ppc[i].sum(axis=0), bins=bin_edges, density=True)[0]
                for i in range(y_ppc.shape[0])
            ]
        )

        q005 = np.quantile(ppc_hists, 0.005, axis=0)
        q025 = np.quantile(ppc_hists, 0.025, axis=0)
        q05 = np.quantile(ppc_hists, 0.05, axis=0)
        q25 = np.quantile(ppc_hists, 0.25, axis=0)
        q50 = np.quantile(ppc_hists, 0.50, axis=0)
        q75 = np.quantile(ppc_hists, 0.75, axis=0)
        q95 = np.quantile(ppc_hists, 0.95, axis=0)
        q975 = np.quantile(ppc_hists, 0.975, axis=0)
        q995 = np.quantile(ppc_hists, 0.995, axis=0)

        fig, ax = plt.subplots(figsize=(6.2, 3.7), dpi=250)
        ax.fill_between(
            centers,
            q005,
            q995,
            step="mid",
            alpha=0.08,
            color="seagreen",
            label="Posterior predictive 99% CI",
        )
        ax.fill_between(
            centers,
            q025,
            q975,
            step="mid",
            alpha=0.12,
            color="seagreen",
            label="Posterior predictive 95% CI",
        )
        ax.fill_between(
            centers,
            q05,
            q95,
            step="mid",
            alpha=0.20,
            color="seagreen",
            label="Posterior predictive 90% CI",
        )
        ax.fill_between(
            centers,
            q25,
            q75,
            step="mid",
            alpha=0.33,
            color="seagreen",
            label="Posterior predictive 50% CI",
        )
        ax.step(
            centers,
            q50,
            where="mid",
            color="seagreen",
            linewidth=2,
            label="Posterior predictive median",
        )
        ax.step(
            centers,
            obs_hist,
            where="mid",
            color="darkorange",
            linewidth=2,
            label="Observed",
        )

        ax.set_xlabel("Gene-wise counts c_i = sum_j X_{j,i}")
        ax.set_ylabel("Density across genes")
        ax.set_title(f"Gene-count histogram ({method.upper()})")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(str(output.pdf))


rule plot_correlation_matrix:
    input:
        data="demo/simulated_data.npz",
        ppc="demo/posterior_predictive_{method}.npz",
    output:
        pdf="plots/{method}/correlation_matrix.pdf",
    run:
        method = wildcards.method
        data = np.load(str(input.data))
        ppc = np.load(str(input.ppc))

        Y_obs = np.asarray(data["Y"], dtype=np.int32)
        y_ppc = np.asarray(ppc["y_ppc"], dtype=np.int32)

        def _safe_corr(x):
            corr = np.corrcoef(x, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr, 1.0)
            return corr

        corr_obs = _safe_corr(Y_obs)
        corr_ppc = np.asarray([_safe_corr(y_ppc[i]) for i in range(y_ppc.shape[0])])
        corr_ppc_med = np.quantile(corr_ppc, 0.50, axis=0)
        corr_ppc_q005 = np.quantile(corr_ppc, 0.005, axis=0)
        corr_ppc_q025 = np.quantile(corr_ppc, 0.025, axis=0)
        corr_ppc_q05 = np.quantile(corr_ppc, 0.05, axis=0)
        corr_ppc_q25 = np.quantile(corr_ppc, 0.25, axis=0)
        corr_ppc_q75 = np.quantile(corr_ppc, 0.75, axis=0)
        corr_ppc_q95 = np.quantile(corr_ppc, 0.95, axis=0)
        corr_ppc_q975 = np.quantile(corr_ppc, 0.975, axis=0)
        corr_ppc_q995 = np.quantile(corr_ppc, 0.995, axis=0)

        corr_delta = corr_obs - corr_ppc_med
        corr_band_50 = corr_ppc_q75 - corr_ppc_q25
        corr_band_90 = corr_ppc_q95 - corr_ppc_q05
        corr_band_95 = corr_ppc_q975 - corr_ppc_q025
        corr_band_99 = corr_ppc_q995 - corr_ppc_q005

        vmax = np.max(np.abs([corr_obs, corr_ppc_med]))
        vmax = max(float(vmax), 0.2)
        dmax = np.max(np.abs(corr_delta))
        dmax = max(float(dmax), 0.1)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=250)

        im0 = axes[0].imshow(corr_obs, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[0].set_title("Observed corr(X_i, X_j)")
        axes[0].set_xlabel("j")
        axes[0].set_ylabel("i")

        im1 = axes[1].imshow(corr_ppc_med, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[1].set_title("PPC median corr(X_i, X_j)")
        axes[1].set_xlabel("j")
        axes[1].set_ylabel("i")

        im2 = axes[2].imshow(corr_delta, cmap="PiYG", vmin=-dmax, vmax=dmax)
        axes[2].set_title("Observed - PPC median")
        axes[2].set_xlabel("j")
        axes[2].set_ylabel("i")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        cbar0 = fig.colorbar(im0, ax=axes[:2], shrink=0.8, fraction=0.03, pad=0.02)
        cbar0.set_label("Correlation")
        cbar1 = fig.colorbar(im2, ax=axes[2], shrink=0.8, fraction=0.08, pad=0.02)
        cbar1.set_label("Difference")

        off_diag = ~np.eye(corr_band_50.shape[0], dtype=bool)
        median_band_50 = np.median(corr_band_50[off_diag])
        median_band_90 = np.median(corr_band_90[off_diag])
        median_band_95 = np.median(corr_band_95[off_diag])
        median_band_99 = np.median(corr_band_99[off_diag])
        fig.suptitle(
            f"Correlation PPC ({method.upper()}) "
            f"band widths: 50%={median_band_50:.3f}, "
            f"90%={median_band_90:.3f}, 95%={median_band_95:.3f}, "
            f"99%={median_band_99:.3f}",
            y=1.03,
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(str(output.pdf), bbox_inches="tight")


rule plot_correlation_pairs_ranked:
    input:
        data="demo/simulated_data.npz",
        ppc="demo/posterior_predictive_{method}.npz",
    output:
        pdf="plots/{method}/correlation_pairs_ranked.pdf",
    run:
        method = wildcards.method
        data = np.load(str(input.data))
        ppc = np.load(str(input.ppc))

        Y_obs = np.asarray(data["Y"], dtype=np.int32)
        y_ppc = np.asarray(ppc["y_ppc"], dtype=np.int32)

        def _safe_corr(x):
            corr = np.corrcoef(x, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr, 1.0)
            return corr

        corr_obs = _safe_corr(Y_obs)
        corr_ppc = np.asarray([_safe_corr(y_ppc[i]) for i in range(y_ppc.shape[0])])

        tri_i, tri_j = np.triu_indices(corr_obs.shape[0], k=1)
        obs_pairs = corr_obs[tri_i, tri_j]
        ppc_pairs = corr_ppc[:, tri_i, tri_j]

        ppc_mean = ppc_pairs.mean(axis=0)
        ppc_q005 = np.quantile(ppc_pairs, 0.005, axis=0)
        ppc_q025 = np.quantile(ppc_pairs, 0.025, axis=0)
        ppc_q05 = np.quantile(ppc_pairs, 0.05, axis=0)
        ppc_q25 = np.quantile(ppc_pairs, 0.25, axis=0)
        ppc_q75 = np.quantile(ppc_pairs, 0.75, axis=0)
        ppc_q95 = np.quantile(ppc_pairs, 0.95, axis=0)
        ppc_q975 = np.quantile(ppc_pairs, 0.975, axis=0)
        ppc_q995 = np.quantile(ppc_pairs, 0.995, axis=0)

        order = np.argsort(ppc_mean)[::-1]
        x = np.arange(order.shape[0])
        obs_sorted = obs_pairs[order]
        mean_sorted = ppc_mean[order]
        q005_sorted = ppc_q005[order]
        q025_sorted = ppc_q025[order]
        q05_sorted = ppc_q05[order]
        q25_sorted = ppc_q25[order]
        q75_sorted = ppc_q75[order]
        q95_sorted = ppc_q95[order]
        q975_sorted = ppc_q975[order]
        q995_sorted = ppc_q995[order]

        fig, ax = plt.subplots(figsize=(10.8, 4.0), dpi=250)
        ax.fill_between(
            x,
            q005_sorted,
            q995_sorted,
            color="steelblue",
            alpha=0.08,
            linewidth=0,
            label="PPC 99% CI",
        )
        ax.fill_between(
            x,
            q025_sorted,
            q975_sorted,
            color="steelblue",
            alpha=0.12,
            linewidth=0,
            label="PPC 95% CI",
        )
        ax.fill_between(
            x,
            q05_sorted,
            q95_sorted,
            color="steelblue",
            alpha=0.20,
            linewidth=0,
            label="PPC 90% CI",
        )
        ax.fill_between(
            x,
            q25_sorted,
            q75_sorted,
            color="steelblue",
            alpha=0.33,
            linewidth=0,
            label="PPC 50% CI",
        )
        ax.plot(
            x,
            mean_sorted,
            color="steelblue",
            linewidth=1.8,
            label="PPC mean",
        )
        ax.scatter(
            x,
            obs_sorted,
            color="darkorange",
            s=8,
            alpha=0.65,
            label="Observed",
            zorder=3,
        )
        ax.axhline(0.0, color="0.35", linewidth=0.8, alpha=0.8)

        ax.set_xlabel("Pair index (sorted by decreasing PPC mean correlation)")
        ax.set_ylabel("Correlation")
        ax.set_title(f"Ranked pairwise correlations ({method.upper()})")
        ax.set_xlim(0, max(1, x.shape[0] - 1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=2, loc="upper right")
        fig.tight_layout()
        fig.savefig(str(output.pdf), bbox_inches="tight")
