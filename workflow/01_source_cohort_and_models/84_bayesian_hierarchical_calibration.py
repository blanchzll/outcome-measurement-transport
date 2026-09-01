# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bayesian hierarchical calibration across five source centres
#
# A common positive calibration slope is combined with centre-specific random
# intercepts. This deliberately avoids centre-specific slopes in sparse centres.
# The posterior is approximated at the joint mode using a Laplace approximation.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(str(_release_path('analysis')))
SEED = 20260827
N_DRAWS = 30000
PREDICTION_COLUMN = "pred_PI_restricted_rf"


@dataclass(frozen=True)
class Prior:
    label: str
    mu_sd: float
    tau_scale: float
    log_beta_sd: float


PRIORS = (
    Prior("primary", 1.5, 1.0, 0.50),
    Prior("stronger_shrinkage", 1.0, 0.5, 0.35),
    Prior("weaker_shrinkage", 2.0, 1.5, 0.75),
)


def objective_and_gradient(theta, y, x, centre_index, prior):
    mu, log_tau, log_beta = theta[:3]
    z = theta[3:]
    tau, beta = np.exp(log_tau), np.exp(log_beta)
    alpha = mu + tau * z
    eta = alpha[centre_index] + beta * x
    p = expit(eta)
    nll = np.sum(np.logaddexp(0.0, eta) - y * eta)
    # Priors, with the Jacobian for log(tau) under a half-normal prior.
    nlp = (
        nll + 0.5 * (mu / prior.mu_sd) ** 2
        + 0.5 * (tau / prior.tau_scale) ** 2 - log_tau
        + 0.5 * (log_beta / prior.log_beta_sd) ** 2
        + 0.5 * np.sum(z**2)
    )
    residual = p - y
    grad = np.empty_like(theta)
    grad[0] = residual.sum() + mu / prior.mu_sd**2
    grad[1] = np.sum(residual * tau * z[centre_index]) + (tau / prior.tau_scale) ** 2 - 1.0
    grad[2] = np.sum(residual * beta * x) + log_beta / prior.log_beta_sd**2
    for j in range(len(z)):
        grad[3 + j] = tau * residual[centre_index == j].sum() + z[j]
    return float(nlp), grad


def numerical_hessian(gradient, theta, step=1e-4):
    hessian = np.empty((len(theta), len(theta)), dtype=float)
    for j in range(len(theta)):
        delta = np.zeros_like(theta)
        delta[j] = step * max(1.0, abs(theta[j]))
        hessian[:, j] = (gradient(theta + delta) - gradient(theta - delta)) / (2 * delta[j])
    return 0.5 * (hessian + hessian.T)


def fit_prior(y, x, centre_index, prior, rng):
    start = np.zeros(3 + int(centre_index.max()) + 1)
    start[1] = np.log(0.5)
    result = minimize(
        lambda t: objective_and_gradient(t, y, x, centre_index, prior),
        start, jac=True, method="L-BFGS-B",
        bounds=[(None, None), (-6, 2), (-2, 2)] + [(None, None)] * (len(start) - 3),
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Posterior mode failed for {prior.label}: {result.message}")
    gradient = lambda t: objective_and_gradient(t, y, x, centre_index, prior)[1]
    hessian = numerical_hessian(gradient, result.x)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    floor = max(1e-7, eigenvalues.max() * 1e-9)
    eigenvalues_regularized = np.maximum(eigenvalues, floor)
    covariance = (eigenvectors / eigenvalues_regularized) @ eigenvectors.T
    draws = rng.multivariate_normal(result.x, covariance, size=N_DRAWS)
    # Preserve optimization bounds in the Gaussian approximation.
    draws[:, 1] = np.clip(draws[:, 1], -6, 2)
    draws[:, 2] = np.clip(draws[:, 2], -2, 2)
    diagnostics = {
        "prior": prior.label,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "negative_log_posterior_at_mode": float(result.fun),
        "gradient_max_abs": float(np.abs(result.jac).max()),
        "hessian_min_eigenvalue_raw": float(eigenvalues.min()),
        "hessian_condition_number_regularized": float(eigenvalues_regularized.max() / eigenvalues_regularized.min()),
        "posterior_approximation": "joint-mode Laplace approximation",
    }
    return draws, diagnostics


def summarize(values):
    values = np.asarray(values, float)
    return {
        "posterior_mean": float(values.mean()),
        "posterior_median": float(np.median(values)),
        "credible_lower_95": float(np.quantile(values, .025)),
        "credible_upper_95": float(np.quantile(values, .975)),
    }


def intercept_at_fixed_slope(y_center, x_center, slope):
    target = float(np.mean(y_center))
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if expit(mid + slope * x_center).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# %%
d = pd.read_csv(ROOT / "secure_work" / "SOURCE_3710_LOCKED_LOCO_PREDICTIONS_SECURE.csv.gz", low_memory=False)
d = d[["Center", "PostopAKI", PREDICTION_COLUMN]].dropna().copy()
d["Center"] = d.Center.astype(str)
centres = sorted(d.Center.unique(), key=lambda value: int(float(value)))
centre_map = {centre: index for index, centre in enumerate(centres)}
centre_index = d.Center.map(centre_map).to_numpy(int)
y = d.PostopAKI.astype(int).to_numpy()
p0 = np.clip(d[PREDICTION_COLUMN].to_numpy(float), 1e-6, 1 - 1e-6)
x = logit(p0)
rng = np.random.default_rng(SEED)

parameter_rows, centre_rows, sensitivity_rows, diagnostics_all = [], [], [], []
for prior in PRIORS:
    draws, diagnostics = fit_prior(y, x, centre_index, prior, rng)
    diagnostics_all.append(diagnostics)
    mu_draw = draws[:, 0]
    tau_draw = np.exp(draws[:, 1])
    beta_draw = np.exp(draws[:, 2])
    z_draw = draws[:, 3:]
    alpha_draw = mu_draw[:, None] + tau_draw[:, None] * z_draw
    beta_median = float(np.median(beta_draw))

    for name, values in (("population_calibration_intercept", mu_draw), ("between_centre_intercept_sd", tau_draw), ("common_calibration_slope", beta_draw)):
        row = {"prior": prior.label, "parameter": name, **summarize(values)}
        parameter_rows.append(row)

    for centre in centres:
        j = centre_map[centre]
        mask = centre_index == j
        observed = int(y[mask].sum())
        raw_expected = float(p0[mask].sum())
        expected_draws = np.empty(N_DRAWS)
        predictive_draws = np.empty(N_DRAWS)
        for start in range(0, N_DRAWS, 1000):
            stop = min(N_DRAWS, start + 1000)
            prob = expit(alpha_draw[start:stop, j, None] + beta_draw[start:stop, None] * x[mask][None, :])
            expected = prob.sum(axis=1)
            expected_draws[start:stop] = expected
            variance = np.sum(prob * (1 - prob), axis=1)
            predictive_draws[start:stop] = np.clip(
                np.rint(rng.normal(expected, np.sqrt(variance))), 0, mask.sum()
            )
        oe_draws = observed / np.maximum(expected_draws, 1e-9)
        row = {
            "prior": prior.label, "center": centre, "n": int(mask.sum()), "events": observed,
            "raw_expected_events": raw_expected, "raw_oe": observed / raw_expected,
            "unpooled_intercept_at_common_slope": intercept_at_fixed_slope(y[mask], x[mask], beta_median),
            "common_slope_for_unpooled_intercept": beta_median,
            **{f"intercept_{k}": v for k, v in summarize(alpha_draw[:, j]).items()},
            **{f"expected_events_{k}": v for k, v in summarize(expected_draws).items()},
            **{f"posterior_oe_{k}": v for k, v in summarize(oe_draws).items()},
            "posterior_predictive_events_lower_95": float(np.quantile(predictive_draws, .025)),
            "posterior_predictive_events_upper_95": float(np.quantile(predictive_draws, .975)),
            "slope_structure": "common_across_centres",
        }
        sensitivity_rows.append(row)
        if prior.label == "primary":
            centre_rows.append(row)

    new_centre_intercept = mu_draw + tau_draw * rng.normal(size=N_DRAWS)
    parameter_rows.append({
        "prior": prior.label, "parameter": "new_centre_predictive_intercept",
        **summarize(new_centre_intercept),
    })

pd.DataFrame(parameter_rows).to_csv(ROOT / "tables" / "Table_bayesian_hierarchical_calibration_parameters.csv", index=False)
pd.DataFrame(centre_rows).to_csv(ROOT / "tables" / "Table_bayesian_hierarchical_calibration_centres.csv", index=False)
pd.DataFrame(sensitivity_rows).to_csv(ROOT / "tables" / "Table_bayesian_hierarchical_calibration_prior_sensitivity.csv", index=False)

audit = {
    "analysis": "Bayesian hierarchical calibration of locked 3,710-patient source-cohort LOCO predictions",
    "prediction": PREDICTION_COLUMN,
    "n": len(d), "events": int(y.sum()), "centres": {centre: {"n": int((d.Center == centre).sum()), "events": int(d.loc[d.Center == centre, "PostopAKI"].sum())} for centre in centres},
    "model": "Bernoulli outcome; common positive calibration slope; centre-specific normally distributed intercepts",
    "primary_priors": {"population_intercept": "Normal(0,1.5)", "between_centre_sd": "half-Normal(0,1)", "log_common_slope": "Normal(0,0.5)"},
    "posterior": f"joint-mode Laplace approximation with {N_DRAWS} Gaussian draws",
    "independent_unit": "unique patient; one eligible operation per patient",
    "reason_for_common_slope": "five centres and only one event in centre 5 do not support centre-specific slope estimation",
    "diagnostics": diagnostics_all,
    "limits": [
        "Credible intervals are conditional on locked predictions and the Laplace approximation.",
        "They do not propagate model-development or preprocessing uncertainty.",
        "Five centres provide limited information about the between-centre variance and new-centre distribution.",
    ],
}
(ROOT / "outputs" / "BAYESIAN_HIERARCHICAL_CALIBRATION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(pd.DataFrame(parameter_rows).loc[lambda x: x.prior.eq("primary")].to_string(index=False))
print(pd.DataFrame(centre_rows)[["center", "n", "events", "raw_oe", "intercept_posterior_median", "intercept_credible_lower_95", "intercept_credible_upper_95"]].to_string(index=False))
