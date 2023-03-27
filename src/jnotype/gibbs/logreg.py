"""Logistic regression utilities."""
import jax
import jax.numpy as jnp
from jaxtyping import Int, Float, Array


def calculate_loglikelihood_matrix_from_logits(
    *,
    logits: Float[Array, "N G"],
    observed: Int[Array, "N G"],
) -> Float[Array, "N G"]:
    """Calculates :math:`\\log P(y | logits)` entry-wise.

    Let logit be :math:`l`. We have:

    :math:`P(y=1 | l) = \\sigma(l),`
    where :math:`\\sigma(l) = 1/(1+\\exp(-l))`.

    We calculate :math:`\\log P(y=1 | l)` entry-wise.

    Args:
        logits: log-odds for success for each entry
        observed: binary matrix with the observed outcomes

    Returns:
        for each entry :math:`\\log P(observed | logit)`
    """
    log_p_y1 = jax.nn.log_sigmoid(logits)
    log_p_y0 = jax.nn.log_sigmoid(-logits)
    return log_p_y1 * observed + log_p_y0 * (1 - observed)


def calculate_logits(
    *,
    intercepts: Float[Array, " G"],
    coefficients: Float[Array, "G F"],
    structure: Int[Array, "G F"],
    covariates: Float[Array, "N F"],
) -> Float[Array, "N G"]:
    """Linear model for logit:

    .. math::

       l_{ng} = a_g + \\sum_f s_{gf}\\cdot b_{gf} x_{nf}

    Args:
        intercepts: term :math:`a_g` for each output :math:`g`
        coefficients: terms :math:`b_{gf}` for each output :math:`g`
          and predictor :math:`f`
        structure: binary structure variable, used to decide whether
          to include the variable in the regression
        covariates: matrix :math:`x_{nf}` giving the predictors for each data point
    """
    return intercepts[None, :] + jnp.einsum(
        "GF,NF->NG", coefficients * structure, covariates
    )


def calculate_loglikelihood_matrix_from_variables(
    *,
    intercepts: Float[Array, " G"],
    coefficients: Float[Array, "G F"],
    structure: Float[Array, "G F"],
    covariates: Float[Array, "N F"],
    observed: Int[Array, "N G"],
) -> Float[Array, "N G"]:
    """Calculates the log-likelihood matrix

    .. math::

       \\log P(y_{ng} \\mid l_{ng} )

    where logits :math:`l_{ng}` are modelled
    via linear regression:

    .. math::

       l_{ng} = a_g + \\sum_f s_{gf}\\cdot b_{gf} x_{nf}

    Args:
        intercepts: term :math:`a_g` for each output :math:`g`
        coefficients: terms :math:`b_{gf}` for each output :math:`g`
          and predictor :math:`f`
        structure: binary structure variable, used to decide whether
          to include the variable in the regression
        covariates: matrix :math:`x_{nf}` giving the predictors for each data point
        observed: binary matrix with observations :math:`y_{ng}`

    Returns:
        for each entry :math:`\\log P(observed | logit)`
    """
    logits = calculate_logits(
        intercepts=intercepts,
        coefficients=coefficients,
        structure=structure,
        covariates=covariates,
    )
    return calculate_loglikelihood_matrix_from_logits(
        logits=logits,
        observed=observed,
    )
