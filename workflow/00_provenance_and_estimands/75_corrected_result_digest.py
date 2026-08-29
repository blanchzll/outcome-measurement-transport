# %% [markdown]
# # Corrected simulation result digest
# Extracts the small set of publication-facing anchors after semantic normalisation.

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"


def one(frame: pd.DataFrame, method: str, metric: str) -> dict | None:
    selected = frame[(frame.method == method) & (frame.metric == metric)]
    if len(selected) != 1:
        return None
    row = selected.iloc[0]
    return {
        key: (None if pd.isna(row.get(key)) else float(row[key]))
        for key in ("n_replicates", "mean", "sd", "q025", "q975", "truth", "bias", "rmse")
        if key in row.index
    }


digest: dict[str, object] = {}
for database in ("INSPIRE", "MIMIC", "EICU"):
    sim = pd.read_csv(TABLES / f"Table_{database.lower()}_simulation_summary.csv")
    condition = sim[
        (sim.mechanism == "mixed_MNAR")
        & (sim.strength == "strong")
        & (sim.retention_target == 0.35)
    ]
    methods = {
        method: {
            metric: one(condition, method, metric)
            for metric in (
                "reconstructed_sensitivity", "event_rate", "oe", "calibration_intercept",
                "calibration_slope", "event_rate_interval_coverage", "event_rate_interval_width",
                "MNAR_event_rate_coverage", "ess", "weight_p99", "weight_max",
            )
            if one(condition, method, metric) is not None
        }
        for method in (
            "full_reference", "naive", "IPAW_design_probability_untruncated",
            "IPAW_design_probability_truncated99", "AIPW_design_probability",
            "recalibration_intercept_slope_apparent", "recalibration_intercept_slope_truth",
            "reference_05pct_recalibration", "reference_10pct_recalibration",
            "reference_20pct_recalibration", "reference_30pct_recalibration",
            "Gamma2_prediction_sensitivity_region",
        )
    }
    failures = pd.read_csv(TABLES / f"Table_{database.lower()}_simulation_failure_diagnostics.csv")
    max_failure = failures.loc[failures.maximum_metric_failure_fraction.idxmax()].to_dict()
    selection = pd.read_csv(TABLES / f"Table_{database.lower()}_pure_label_selection_control.csv")
    selection = selection[
        (selection.retention_target == 0.35)
        & (selection.strength == "strong")
        & selection.mechanism.isin(["risk_MAR", "outcome_MNAR", "mixed_MNAR"])
        & (selection.metric == "event_rate")
    ][["mechanism", "method", "mean", "q025", "q975", "truth", "bias", "rmse"]]
    digest[database] = {
        "condition": "strong mixed_MNAR, retention_target=0.35",
        "methods": methods,
        "maximum_metric_failure": max_failure,
        "pure_label_selection_positive_control": selection.to_dict("records"),
    }

path = OUTPUTS / "CORRECTED_SIMULATION_RESULT_DIGEST.json"
path.write_text(json.dumps(digest, indent=2, default=float) + "\n", encoding="utf-8")
print(json.dumps(digest, indent=2, default=float))
