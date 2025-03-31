# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

import jax.numpy as jnp
from pathlib import Path
import numpy as np
from _params import SPIKE_WEIGHT, SCALE

WEIGHTS_DIR = Path("data/G_3/weights.npy")
BAYES_WEIGHTS_DIR = Path("data/G_3/N_1000/bayes/sampled_genotypes.npy")
DFD_WEIGHTS_DIR = Path("data/G_3/N_1000/dfd/sampled_genotypes.npy")

gt_theta = jnp.load(WEIGHTS_DIR)
gt_theta_diag = np.diag(gt_theta)
gt_theta_off_diag = gt_theta - np.diag(np.diag(gt_theta))
# extract the upper triangular part of the matrix
gt_theta_off_diag = gt_theta_off_diag[np.triu_indices(gt_theta_off_diag.shape[0], 1)]
gt_sampled_bayes = jnp.load(BAYES_WEIGHTS_DIR, allow_pickle=True)
gt_sampled_bayes = np.expand_dims(gt_sampled_bayes, axis=0)
gt_sampled_dfd = jnp.load(DFD_WEIGHTS_DIR, allow_pickle=True)
gt_sampled_dfd = np.expand_dims(gt_sampled_dfd, axis=0)

bayes_diag = np.mean(gt_sampled_bayes[0]["diag_vals"], axis=0)
bayes_off_diag = np.mean(gt_sampled_bayes[0]["off_diag_vals"], axis=0)
dfd_diag = np.mean(gt_sampled_dfd[0]["diag_vals"], axis=0)
dfd_off_diag = np.mean(gt_sampled_dfd[0]["off_diag_vals"], axis=0)

gt_sampled_bayes

# bayes pi and scale
bayes_pi = np.mean(gt_sampled_bayes[0]["pi"], axis=0)
bayes_scale = np.mean(gt_sampled_bayes[0]["sigma"], axis=0)
SPIKE_WEIGHT, bayes_pi.item(), SCALE, bayes_scale.item()

gt_theta_diag, bayes_diag, dfd_diag

gt_theta_off_diag, bayes_off_diag, dfd_off_diag
