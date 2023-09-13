import dataclasses
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xarray as xr
import sys
import io
import json
import seaborn as sns
from scipy.stats import chi2

from sklearn.preprocessing import StandardScaler, OneHotEncoder

import formulaic as fm
from lifelines import CoxPHFitter
from lifelines.calibration import survival_probability_calibration

from jnotype.pyramids import TwoLayerPyramidSampler, TwoLayerPyramidSamplerNonparametric
from jnotype.sampling import ListDataset


matplotlib.use("agg")


@dataclasses.dataclass
class AnalysisParams:
  # --- Data specification ---
  allowed_types: list[str]

  # --- Bayesian pyramids ---
  # Model specification
  design_formula: str = "std_age + gender + type + stage"
  n_clusters: int = 4
  max_binary_codes: int = 8
  expected_binary_codes: float = 3.0
  # MCMC parameters
  n_warmup: int = 12_000
  n_steps: int = 3000
  thinning: int = 5

  # --- Survival analysis ---
  penalizer: float = 0.05

pancancer_list = ["BRCA", "LGG", "HNSC", "PRAD", "THCA", "OV", "LUAD"]

ANALYSES = {
  # "AML": AnalysisParams(allowed_types=["LAML"]),
  "BRCA": AnalysisParams(allowed_types=["BRCA"]),
  "BRCA-Minimal": AnalysisParams(allowed_types=["BRCA"], design_formula="std_age + gender"),
  "Brain": AnalysisParams(allowed_types=["LGG", "GBM"]),
  "Brain-Minimal": AnalysisParams(allowed_types=["LGG", "GBM"], design_formula="std_age + gender"),
  "Pancancer": AnalysisParams(allowed_types=pancancer_list),
  "Pancancer-Minimal": AnalysisParams(allowed_types=pancancer_list, design_formula="std_age + gender"),
}

N_BOOTSTRAP: int = 20
BOOTSTRAP_INDICES = list(range(1, N_BOOTSTRAP + 1))

rule all:
  input:
    survival_plots = expand("generated/TCGA/{analysis}/summary/survival/plot.pdf", analysis=ANALYSES.keys()),
    latent_traits_plots = expand("generated/TCGA/{analysis}/summary/latent_traits.pdf", analysis=ANALYSES.keys()),
    effect_sizes_plot = expand("generated/TCGA/{analysis}/summary/effect_sizes.pdf", analysis=ANALYSES.keys())


rule download_clinical:
  output: "data/TCGA/raw/clinical-information.tsv"
  shell: "wget https://raw.githubusercontent.com/cbg-ethz/graphClust_NeurIPS/main/tcga_analysis/data/tcga-clinical-information.txt -O {output}"


rule download_mutation:
  output: "data/TCGA/raw/mutation-matrix.tsv"
  shell: "wget https://raw.githubusercontent.com/cbg-ethz/graphClust_NeurIPS/main/tcga_analysis/data/binary-mutationCovariate-matrix.txt -O {output}"


rule preprocess_data:
  input:
    clinical = "data/TCGA/raw/clinical-information.tsv",
    mutation = "data/TCGA/raw/mutation-matrix.tsv"
  output:
    clinical = "generated/TCGA/{analysis}/preprocessed/clinical-information.csv",
    mutation = "generated/TCGA/{analysis}/preprocessed/mutation-matrix.csv"
  run:
    # Read mutation matrix and remove binarised covariates
    mutations = pd.read_csv(input.mutation, sep="\t", index_col=0)
    mutations = mutations.drop(["Age", "Gender", "Stage"], axis="columns")
    # Read clinical information and remove binarised (derived) information
    clinical = pd.read_csv(input.clinical, sep="\t", index_col=0)
    clinical = clinical.drop(["age.bin", "gender.bin", "stage.bin"], axis="columns")

    # Select cancers corresponding to selected tissue types
    allowed_tissue_types = ANALYSES[wildcards.analysis].allowed_types
    clinical = clinical[clinical["type"].isin(allowed_tissue_types)]

    # Align data frames
    mutations, clinical = mutations.align(clinical, join="inner", axis=0)
    
    # Remove mutations which are constant for all patients
    mutations = mutations.loc[:, mutations.nunique() > 1]
    
    # Add standardized age
    clinical['std_age'] = (clinical['age'] - clinical['age'].mean()) / clinical['age'].std()

    mutations.to_csv(output.mutation, index=True)
    clinical.to_csv(output.clinical, index=True)


rule bootstrap_data:
  input:
    clinical = "generated/TCGA/{analysis}/preprocessed/clinical-information.csv",
    mutation = "generated/TCGA/{analysis}/preprocessed/mutation-matrix.csv"
  output:
    clinical = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/clinical-information.csv",
    mutation = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/mutation-matrix.csv"
  run:
    clinical = pd.read_csv(input.clinical, index_col=0)
    mutation = pd.read_csv(input.mutation, index_col=0)
    assert len(clinical) == len(mutation), "Data frames are of different length"

    # Bootstrap samples from both data frames
    rng = np.random.default_rng(int(wildcards.bootstrap))
    idx = rng.choice(np.arange(len(clinical)), size=len(clinical), replace=True)
    bootstrap_clinical = clinical.iloc[idx]
    bootstrap_mutation = mutation.iloc[idx]

    bootstrap_clinical.to_csv(output.clinical, index=True)
    bootstrap_mutation.to_csv(output.mutation, index=True)


rule fit_pyramid:
  input:
    clinical = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/clinical-information.csv",
    mutation = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/mutation-matrix.csv"
  output:
    posterior_samples = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/posterior_samples.nc",
    observed_covariates = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/observed_covariates.csv"
  run:
    clinical = pd.read_csv(input.clinical, index_col=0)
    mutations = pd.read_csv(input.mutation, index_col=0)
    assert len(clinical) == len(mutations), "Data frames are of different length"

    specs = ANALYSES[wildcards.analysis]

    # Create design matrix, encoding observed covariates
    observed_covariates = fm.model_matrix(specs.design_formula, clinical)
    observed_covariates = observed_covariates.drop("Intercept", axis=1)
    observed_covariates.to_csv(output.observed_covariates, index=True)

    design_matrix = observed_covariates.values

    dataset = ListDataset(thinning=specs.thinning, dimensions=TwoLayerPyramidSamplerNonparametric.dimensions())

    sampler = TwoLayerPyramidSamplerNonparametric(
      datasets=[dataset],
      observed=mutations.values,
      observed_covariates=design_matrix,
      dirichlet_prior=np.ones(specs.n_clusters) / specs.n_clusters,
      n_clusters=specs.n_clusters,
      max_binary_codes=specs.max_binary_codes,
      expected_binary_codes=specs.expected_binary_codes,
      verbose=True,
      warmup=specs.n_warmup,
      steps=specs.n_steps,
      inactive_latent_variance_theta_inf = 0.1**2,
      intercept_prior_variance=1.0**2,
      pseudoprior_variance=0.1**2,
      seed=int(wildcards.bootstrap),
    )
    sampler.run()
    dataset.dataset.to_netcdf(output.posterior_samples)


rule extract_latents:
  input:
    posterior_samples = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/posterior_samples.nc",
  output:
    latent_traits = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/latent_traits.npz"
  run:
    dataset = xr.open_dataset(input.posterior_samples)
    # Get means of latent traits.
    latent_traits_means = dataset["latent_traits"].mean(axis=0).values
    # Get variances of coefficients attributed to latent traits
    variances = dataset["latent_variances"].mean(axis=0).values

    # Select the baseline category for each trait (i.e., what is 0 and what is 1).
    # We define the baseline category as the more prevalent one, so that the mean is smaller
    latent_traits_ordered = np.zeros_like(latent_traits_means)
    for k in range(latent_traits_means.shape[1]):
      if latent_traits_means[:, k].mean() > 0.5:
        latent_traits_ordered[:, k] = 1 - latent_traits_means[:, k]
      else:
        latent_traits_ordered[:, k] = latent_traits_means[:, k]

    # Now we need to remove "wrong" latent traits.
    # By "wrong" we will understand the following:
    #   - It appears in too few patients.
    #   - It has very small variance (i.e., uncertainty of it for all patients is almost identical)
    #   - The variance of associated coefficients is too small. (I.e., it's inactive)   
    is_too_rare = np.mean(latent_traits_ordered, axis=0) < 0.05
    is_constant = np.std(latent_traits_ordered, axis=0) < 0.05
    has_zero_variance = variances < 0.1
    is_wrong = is_too_rare | is_constant | has_zero_variance

    latent_traits_ordered = latent_traits_ordered[:, ~is_wrong]

    # We have pre-selected some traits.
    # However, their order may be arbitrary and inconsistent between bootstraps.
    # Hence, let's order the traits from most prevalent to the least.

    order = latent_traits_ordered.mean(axis=0).argsort()[::-1]
    latent_traits_ordered = latent_traits_ordered[:, order]

    np.savez(
      output.latent_traits,
      latent_traits=latent_traits_ordered,
      is_too_rare=is_too_rare,
      is_constant=is_constant,
      order=order,
    )


class CaptureStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._new_stdout = io.StringIO()
        sys.stdout = self._new_stdout
        return self._new_stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def fit_cph_model(
  design_matrix: pd.DataFrame,
  summary_file: str,
  coefficients_file: str,
  penalizer: float,
) -> CoxPHFitter:
  # Fit the model
  cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
  cph.fit(design_matrix, duration_col='time', event_col='event')

  # Save dataframe with coefficients
  cph.summary.to_csv(coefficients_file, index=True)

  # Save human-readable summary
  with CaptureStdout() as captured:
    cph.print_summary(decimals=3, style="ascii")
  with open(summary_file, "w") as f:  
      f.write(captured.getvalue())
  # Return fitted model
  return cph

DESIGN_FORMULA = "event + time + gender + std_age + stage + type"

def normalize_time(design_matrix):
  time_original = design_matrix["time"].values.copy()
  # design_matrix["time_original"] = time_original
  design_matrix["time"] = 1e-6 + time_original / time_original.max()

def get_simple_design_matrix(clinical: pd.DataFrame, filename: str) -> pd.DataFrame:
  design_formula = DESIGN_FORMULA
  design_matrix = fm.model_matrix(design_formula, clinical)
  design_matrix = design_matrix.drop("Intercept", axis=1)
  normalize_time(design_matrix)

  design_matrix.index = np.arange(len(design_matrix))
  design_matrix.to_csv(filename, index=True)
  
  return design_matrix


def get_extended_design_matrix(clinical: pd.DataFrame, latent_traits: np.ndarray, filename: str) -> pd.DataFrame:
  assert len(clinical) == len(latent_traits), "Data sets are of different length"

  # We want to use the following values:
  design_formula = DESIGN_FORMULA

  # If we have latent traits, we add them to the design matrix
  traits_names = [f"Trait_{i}" for i in range(1, 1+latent_traits.shape[1])]
  traits_df = pd.DataFrame(latent_traits, columns=traits_names, index=clinical.index)

  if len(traits_df.columns) > 0:
    design_formula = DESIGN_FORMULA + " + " + "+".join(traits_df.columns)

  design_matrix = fm.model_matrix(design_formula, pd.concat([clinical, traits_df], axis=1))
  design_matrix = design_matrix.drop("Intercept", axis=1)
  normalize_time(design_matrix)

  design_matrix.index = np.arange(len(design_matrix))
  design_matrix.to_csv(filename, index=True)
  return design_matrix


def compute_p_value(extended: CoxPHFitter, restricted: CoxPHFitter) -> dict:
    diff_df = len(extended.params_) - len(restricted.params_)
    LR = 2 * (extended.log_likelihood_ - restricted.log_likelihood_)
    if diff_df > 0:
      p_value = chi2.sf(LR, diff_df)
    else:
      p_value = None
    return {
      "degrees_of_freedom_difference": diff_df,
      "likelihood_ratio_statistic": LR,
      "likelihood_ratio_p_value": p_value,
      "p_value_is_numeric": p_value is not None,
    }


def compute_c_index(extended: CoxPHFitter, restricted: CoxPHFitter) -> dict:
    conc_restricted = restricted.concordance_index_
    conc_extended = extended.concordance_index_
    conc_improvement = conc_extended - conc_restricted
    return {
      "concordance_restricted": conc_restricted,
      "concordance_extended": conc_extended,
      "concordance_improvement": conc_improvement,
    }


def compute_in_sample_calibration(
  extended_model: CoxPHFitter,
  extended_design: pd.DataFrame,
  restricted_model: CoxPHFitter,
  restricted_design: pd.DataFrame,
  axs = None,
) -> dict:
  if axs is None:
    fig, axs = plt.subplots(1, 2)
  
  t0 = restricted_design["time"].max()
  _ = extended_design["time"].max()
  assert abs(t0 - _) < 0.01, "Time points are not the same"

  # Note that ICI and E50 are errors, i.e., lower is better
  _, ici_restricted, e50_restricted = survival_probability_calibration(restricted_model, restricted_design.reset_index(), t0=t0, ax=axs[0])
  _, ici_extended, e50_extended = survival_probability_calibration(extended_model, extended_design.reset_index(), t0=t0, ax=axs[1])

  ici_improvement = ici_restricted - ici_extended
  e50_improvment = e50_restricted - e50_extended
  return {
    "ici_restricted": ici_restricted,
    "ici_extended": ici_extended,
    "ici_improvement": ici_improvement,
    "e50_restricted": e50_restricted,
    "e50_extended": e50_extended,
    "e50_improvment": e50_improvment,
  }


rule fit_survival:
  input:
    clinical = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/clinical-information.csv",
    latent_traits = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/latent_traits.npz"
  output:
     extended_design_matrix = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/extended_design_matrix.csv",
     extended_ascii_file = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/extended_summary.txt",
     extended_survival_coef = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/extended_coef.csv",
     restricted_design_matrix = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/restricted_design_matrix.csv",
     restricted_ascii_file = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/restricted_summary.txt",
     restricted_survival_coef = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/restricted_coef.csv",
     difference_summary = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/difference_summary.json",
     # residuals_arrays = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/residuals.npz",
     residuals_plot = "generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/residuals.pdf"
  run:
    clinical = pd.read_csv(input.clinical, index_col=0)
    latent_traits = np.load(input.latent_traits)["latent_traits"]
    
    restricted_design_matrix = get_simple_design_matrix(clinical, filename=output.restricted_design_matrix)
    extended_design_matrix = get_extended_design_matrix(clinical=clinical, latent_traits=latent_traits, filename=output.extended_design_matrix)

    spec = ANALYSES[wildcards.analysis]

    cph_restricted = fit_cph_model(
      design_matrix=restricted_design_matrix,
      summary_file=output.restricted_ascii_file,
      coefficients_file=output.restricted_survival_coef,
      penalizer=spec.penalizer,
    )

    cph_extended = fit_cph_model(
      design_matrix=extended_design_matrix,
      summary_file=output.extended_ascii_file,
      coefficients_file=output.extended_survival_coef,
      penalizer=spec.penalizer,
    )

    basic_info = {"n_latent_traits": latent_traits.shape[1], "n_points": len(clinical)}

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    difference_dict = {
      **basic_info,
      **compute_p_value(extended=cph_extended, restricted=cph_restricted),
      **compute_c_index(extended=cph_extended, restricted=cph_restricted),
      **compute_in_sample_calibration(
        extended_model=cph_extended,
        extended_design=extended_design_matrix,
        restricted_model=cph_restricted,
        restricted_design=restricted_design_matrix,
        axs=(axs[0], axs[1]),
      ),
    }

    fig.tight_layout()
    fig.savefig(output.residuals_plot)

    with open(output.difference_summary, "w") as f:
      json.dump(difference_dict, f)


rule plot_survival_difference:
  input: 
    lambda wildcards: expand("generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/difference_summary.json", bootstrap=BOOTSTRAP_INDICES, analysis=wildcards.analysis)
  output:
    survival_plot = "generated/TCGA/{analysis}/summary/survival/plot.pdf",
    assembled_csv = "generated/TCGA/{analysis}/summary/survival/assembled.csv"
  run:
    _tmp_lst = []
    for sample_path in input:
      with open(sample_path) as f:
        d = json.load(f)
        _tmp_lst.append(d)
    
    assembled_df = pd.DataFrame(_tmp_lst)
    assembled_df.to_csv(output.assembled_csv, index=False)

    original_len = len(assembled_df)

    # Remove samples with no latent traits
    assembled_df = assembled_df.loc[assembled_df["n_latent_traits"] > 0]
    new_len = len(assembled_df)

    if new_len < original_len:
      print(f"Removed {original_len - new_len} samples with no latent traits")

    if new_len == 0:
      print("No samples with latent traits")
      raise Exception("No samples with latent traits")
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle(f"Survival analysis summary, $N={len(assembled_df)}$")

    ax = axs[0]
    ax.set_ylabel("Counts")
    ax.set_xlabel("Log $p$-value")
    ax.hist(np.log10(assembled_df["likelihood_ratio_p_value"].values), color="blue", alpha=0.5, bins=[-12, -10, -5, -3, np.log10(0.05), 0])
    ax.axvline(np.log10(0.05), color="black", linestyle=":", linewidth=1, alpha=0.8)

    def _plot_hist(ax, vals):
      ax.hist(relative_improvement, bins=10, color="blue", alpha=0.5)
      ax.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.8)
      ax.axvline(np.median(relative_improvement), color="orangered", linestyle="-", linewidth=2)

    ax = axs[1]
    ax.set_xlabel("$c_+$")
    _plot_hist(ax, assembled_df["concordance_improvement"].values)

    ax = axs[2]
    ax.set_xlabel("$\\mathrm{ICI}_+$")
    _plot_hist(ax, assembled_df["ici_improvement"].values)

    ax = axs[3]
    ax.set_xlabel("$\\mathrm{E50}_+$")
    _plot_hist(ax, assembled_df["e50_improvment"].values) 

    fig.tight_layout()
    fig.savefig(output.survival_plot)


rule plot_latent_traits:
  input: 
    lambda wildcards: expand("generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/difference_summary.json", bootstrap=BOOTSTRAP_INDICES, analysis=wildcards.analysis)
  output:
    latent_traits_plot = "generated/TCGA/{analysis}/summary/latent_traits.pdf"
  run:
    _tmp_lst = []
    for sample_path in input:
      with open(sample_path) as f:
        d = json.load(f)
        _tmp_lst.append(d)
    
    assembled_df = pd.DataFrame(_tmp_lst)

    fig, ax = plt.subplots()
    ax.set_xlabel("Number of latent traits")
    ax.set_ylabel("Counts")
    ax.hist(assembled_df["n_latent_traits"].values, bins=np.arange(-0.5, 9.5, 1), color="blue", alpha=0.5)
    ax.set_xticks(np.arange(0, 10))

    fig.tight_layout()
    fig.savefig(output.latent_traits_plot)


COLORMAP = {
  "Age": "darkgreen",
  "Type": "maroon",
  "Gender": "royalblue",
  "Stage": "gold",
  "Trait": "indigo",
}

rule plot_effect_sizes_colormap:
  output: "generated/TCGA/{analysis}/summary/coefficient_colormap.pdf"
  run:
    color_dict = COLORMAP
    names = list(color_dict.keys())
    colors = [color_dict[name] for name in names]

    fig, ax = plt.subplots(figsize=(3, 2))

    # Create a bar for each name with its associated color
    for i, (name, color) in enumerate(color_dict.items()):
      ax.barh(i, 1, color=color)

    ax.set_xticks([])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)

    # Setting title and adjusting layout
    ax.set_title("Coefficients color map")
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(str(output))


def _color_of_attribute(attr) -> str:
    if "gender" in attr:
        return COLORMAP["Gender"]
    elif "type" in attr:
        return COLORMAP["Type"]
    elif "stage" in attr:
        return COLORMAP["Stage"]
    elif "std_age" in attr:
        return COLORMAP["Age"]
    elif "Trait" in attr:
        return COLORMAP["Trait"]
    else:
        raise ValueError(f"Attr {attr} not known")


rule plot_effect_sizes:
  input: 
    dataframes = lambda wildcards: expand("generated/TCGA/{analysis}/bootstraps/{bootstrap}/survival/extended_coef.csv", bootstrap=BOOTSTRAP_INDICES, analysis=wildcards.analysis),
    colormap = "generated/TCGA/{analysis}/summary/coefficient_colormap.pdf"
  output:
    latent_traits_plot = "generated/TCGA/{analysis}/summary/effect_sizes.pdf"
  run:
    pths = sorted(input.dataframes)
    n_bootstraps = len(pths)

    fig, ax = plt.subplots(figsize=(3, 12))

    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Bootstrap sample")
    ax.axvline(0, color="k", alpha=0.5, linestyle="--")
    ax.set_xlabel("Coefficient value")

    for h in np.linspace(0, 1, n_bootstraps + 1):
      ax.axhline(h, color="k", alpha=0.1)

    for i, pth in enumerate(pths):
      df = pd.read_csv(pth)
      n_attrs = len(df)
      ys = i/n_bootstraps + np.linspace(0.001/n_bootstraps, 0.999/n_bootstraps, n_attrs + 2)[1:-1][::-1]

      for j, row in df.iterrows():
          x_mid = row["coef"]
          x_lower = x_mid - row["coef lower 95%"]
          x_upper = row["coef upper 95%"] - x_mid
          color = _color_of_attribute(row["covariate"])
          ax.errorbar(row["coef"], ys[j], xerr=[[x_lower], [x_upper]], color=color, marker="o", markersize=2)

    fig.tight_layout()
    fig.savefig(output.latent_traits_plot)