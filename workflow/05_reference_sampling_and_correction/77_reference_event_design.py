# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Reference-event design for local recalibration
# Compare ordinary intercept+slope updating with weakly identity-anchored
# penalised updating in held-out reference samples.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(str(_release_path('analysis')))
CODE, TABLES, SECURE, OUTPUTS = ROOT / "code", ROOT / "tables", ROOT / "secure_work", ROOT / "outputs"
SEED, REPS = 20260827, 1000
FRACTIONS = (0.05, 0.10, 0.20, 0.30)

spec = importlib.util.spec_from_file_location("simulation_core", CODE / "52_measurement_deletion_simulation.py")
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)


def identity_anchored_recalibration(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """MAP update with weak Normal priors: intercept 0 (SD 2.5), slope 1 (SD 1)."""
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    y = np.asarray(y, float)

    def objective(beta: np.ndarray) -> float:
        eta = beta[0] + beta[1] * z
        nll = -np.sum(y * eta - np.logaddexp(0, eta))
        penalty = 0.5 * (beta[0] / 2.5) ** 2 + 0.5 * ((beta[1] - 1.0) / 1.0) ** 2
        return float(nll + penalty)

    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not np.isfinite(fit.fun):
        raise RuntimeError("Penalised calibration failed")
    return float(fit.x[0]), float(fit.x[1])


def update(method: str, p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if method == "unpenalised":
        if np.unique(y).size < 2:
            raise RuntimeError("single-class reference sample")
        _, intercept, slope = core.recalibrate(pd.Series(p), pd.Series(y), intercept_only=False)
        return float(intercept), float(slope)
    return identity_anchored_recalibration(p, y)


def event_band(events: int) -> str:
    if events < 5:
        return "0-4"
    if events < 10:
        return "5-9"
    if events < 20:
        return "10-19"
    return "20+"


rows: list[dict[str, object]] = []
for database, prepared in [
    ("INSPIRE", core.prepare_inspire),
    ("MIMIC", core.prepare_mimic),
    ("EICU", core.prepare_eicu),
]:
    patient, _ = prepared()
    y = patient.y_full.to_numpy(int)
    p = patient.risk.to_numpy(float)
    rng = np.random.default_rng(SEED + {"INSPIRE": 0, "MIMIC": 1, "EICU": 2}[database])
    for replicate in range(REPS):
        order = rng.permutation(len(patient))
        for fraction in FRACTIONS:
            n_reference = max(30, int(np.ceil(fraction * len(patient))))
            reference, evaluation = order[:n_reference], order[n_reference:]
            events = int(y[reference].sum())
            for method in ("unpenalised", "identity_anchored_penalised"):
                row = {
                    "database": database,
                    "replicate": replicate,
                    "reference_fraction": fraction,
                    "reference_n": n_reference,
                    "reference_events": events,
                    "reference_event_band": event_band(events),
                    "method": method,
                    "fit_failed": 0,
                }
                try:
                    intercept, slope = update(method, p[reference], y[reference])
                    updated = expit(intercept + slope * logit(np.clip(p[evaluation], 1e-6, 1 - 1e-6)))
                    metrics = core.weighted_metrics(y[evaluation], updated)
                    row.update({
                        "fitted_intercept": intercept,
                        "fitted_slope": slope,
                        "heldout_oe": metrics["oe"],
                        "heldout_calibration_slope": metrics["calibration_slope"],
                        "heldout_brier": metrics["brier"],
                        "absolute_log_oe": abs(np.log(metrics["oe"])),
                    })
                except Exception:
                    row["fit_failed"] = 1
                rows.append(row)

raw = pd.DataFrame(rows)
raw.to_csv(SECURE / "REFERENCE_EVENT_DESIGN_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")

summary_rows = []
for keys, group in raw.groupby(["database", "reference_fraction", "method"], observed=True):
    usable = group.loc[group.fit_failed.eq(0)]
    summary_rows.append({
        "database": keys[0],
        "reference_fraction": keys[1],
        "method": keys[2],
        "reference_n": int(group.reference_n.iloc[0]),
        "reference_events_median": float(group.reference_events.median()),
        "reference_events_q025": float(group.reference_events.quantile(.025)),
        "reference_events_q975": float(group.reference_events.quantile(.975)),
        "fit_failure_fraction": float(group.fit_failed.mean()),
        "heldout_oe_median": float(usable.heldout_oe.median()),
        "heldout_oe_q025": float(usable.heldout_oe.quantile(.025)),
        "heldout_oe_q975": float(usable.heldout_oe.quantile(.975)),
        "absolute_log_oe_mean": float(usable.absolute_log_oe.mean()),
        "heldout_brier_mean": float(usable.heldout_brier.mean()),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(TABLES / "Table_reference_event_design.csv", index=False)

by_events = raw.groupby(["database", "reference_event_band", "method"], observed=True).agg(
    replicates=("replicate", "size"),
    fit_failure_fraction=("fit_failed", "mean"),
    median_reference_events=("reference_events", "median"),
    median_heldout_oe=("heldout_oe", "median"),
    mean_absolute_log_oe=("absolute_log_oe", "mean"),
).reset_index()
by_events.to_csv(TABLES / "Table_reference_event_count_operating_characteristics.csv", index=False)

audit = {
    "replicates_per_database": REPS,
    "fractions": list(FRACTIONS),
    "evaluation": "held-out retained operational reference",
    "penalised_prior": "intercept Normal(0,2.5^2); slope Normal(1,1^2)",
    "selection_use": "none; comparison prespecified as a stability sensitivity analysis",
    "claim": "reference-event design audit, not an externally validated recalibration rule",
}
(OUTPUTS / "REFERENCE_EVENT_DESIGN_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
