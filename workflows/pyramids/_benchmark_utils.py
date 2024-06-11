from collections import defaultdict

import numpy as np


def generate_dataset(n_points: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n_points, 3))
    A = rng.binomial(1, p=[0.1, 0.4, 0.1, 0.5], size=(n_points, 4))
    index = np.argsort(["".join(map(str, a)) for a in A])
    A = A[index, :]

    states = np.hstack((A, X))
    n_covariates = states.shape[1]

    n_genes_per_covariate = 5
    n_additional_genes = 3
    n_genes = n_genes_per_covariate * n_covariates + n_additional_genes

    coefs = np.zeros((n_genes, n_covariates))
    effect_size = 4.0

    for i in range(n_covariates):
        coefs[i * n_genes_per_covariate : (i + 1) * n_genes_per_covariate, i] = (
            effect_size
        )

    if n_additional_genes > 0:
        coefs[-n_additional_genes:, :] = effect_size * rng.binomial(
            1, 0.5, size=(n_additional_genes, n_covariates)
        )

    offset = -5
    logits = offset + np.einsum("nf,gf->ng", states, coefs)
    ps = 1 / (1 + np.exp(-logits))
    Y = rng.binomial(1, ps)

    return {
        "Y": Y,
        "X": X,
        "A": A,
        "coefficients_X": coefs[:, -3:],
        "mutual_information": np.array(mutual_information(A, A)),
    }


def calculate_probabilities(samples):
    counts = defaultdict(int)
    total_samples = len(samples)

    for s in samples:
        counts[s] += 1

    probabilities = {k: v / total_samples for k, v in counts.items()}
    return probabilities


def mutual_information(X_samples, Y_samples):
    assert len(X_samples) == len(Y_samples), "Mismatched sample sizes"

    # Joint probabilities P(X, Y)
    joint_samples = [(tuple(x), tuple(y)) for x, y in zip(X_samples, Y_samples)]
    joint_probabilities = calculate_probabilities(joint_samples)

    # Marginal probabilities P(X) and P(Y)
    X_probabilities = calculate_probabilities([tuple(x) for x in X_samples])
    Y_probabilities = calculate_probabilities([tuple(y) for y in Y_samples])

    MI = 0

    for (x, y), p_xy in joint_probabilities.items():
        p_x = X_probabilities.get(x, 0)
        p_y = Y_probabilities.get(y, 0)

        if p_x > 0 and p_y > 0:
            MI += p_xy * np.log2(p_xy / (p_x * p_y))

    return MI
