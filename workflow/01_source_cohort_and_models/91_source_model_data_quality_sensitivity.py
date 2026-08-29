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
# # Prespecified source-model data-quality sensitivity analysis
#
# This audit does not select a new model. It refits the primary perioperative restricted
# random forest with the original centre-specific locked hyperparameters after (1) treating
# implausible values in current predictors as missing, (2) excluding three binary/stage
# inconsistencies, and (3) applying both changes. Estimated preprocessing remains within
# each development fold.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(str(_release_path('source')))
ROOT = Path(str(_release_path('analysis')))
DATA = BASE / "secure_source" / "inter3_deidentified_4014.csv"
LOCK = BASE / "outputs_evidence_closure_20260823" / "loco_4014_corrected_formal" / "model_lock.json"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
SECURE = ROOT / "secure_work"
N_BOOTSTRAP = 1000
SEED = 20260828

sys.path.insert(0, str(BASE))
from analysis import CENTER, TARGET, load_cohort  # noqa: E402
from loco_analysis import (  # noqa: E402
    FEATURE_SET_SPECS,
    bootstrap_metric_ci,
    build_loco_search,
    engineer_loco_features,
    probability_metrics,
)


RANGES = {
    "Age": (16, 105),
    "PreopHb": (30, 220),
    "PreopAlb": (10, 70),
    "PreopCr": (20, 1500),
    "IntraopTransfusion": (0, 20000),
}


def parse_value(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_range_rules(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    result = frame.copy()
    changed = {}
    for variable, (lower, upper) in RANGES.items():
        numeric = pd.to_numeric(result[variable], errors="coerce")
        invalid = numeric.lt(lower) | numeric.gt(upper)
        changed[variable] = int(invalid.sum())
        result.loc[invalid, variable] = np.nan
    return result, changed


def apply_stage_consistency(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    aki = pd.to_numeric(frame[TARGET], errors="coerce")
    stage = pd.to_numeric(frame["AKIStage"], errors="coerce")
    inconsistent = ((aki == 0) & (stage > 0)) | ((aki == 1) & (stage == 0))
    return frame.loc[~inconsistent].copy(), int(inconsistent.sum())


def locked_parameters() -> dict[int, dict]:
    payload = json.loads(LOCK.read_text())
    rows = [
        row for row in payload["fold_locks"]
        if row["feature_set"] == "PI" and row["model"] == "restricted_rf"
    ]
    if len(rows) != 5:
        raise ValueError("The original PI restricted-random-forest lock must contain five centres.")
    return {
        int(row["outer_center"]): {key: parse_value(value) for key, value in row["best_params"].items()}
        for row in rows
    }


def fit_locked_loco(frame: pd.DataFrame, parameters: dict[int, dict]):
    engineered = engineer_loco_features(frame).reset_index(drop=True)
    spec = FEATURE_SET_SPECS["PI"]
    probability = np.full(len(engineered), np.nan)
    fit_rows = []
    for center in sorted(engineered[CENTER].astype(int).unique()):
        validation = engineered[CENTER].astype(int).eq(center).to_numpy()
        development = ~validation
        groups = engineered.loc[development, CENTER].astype(int).to_numpy()
        search = build_loco_search(spec, "restricted_rf", n_inner_centers=len(np.unique(groups)), fast=True)
        search.set_params(param_grid={key: [value] for key, value in parameters[center].items()})
        search.fit(
            engineered.loc[development, list(spec.features)],
            engineered.loc[development, TARGET].astype(int).to_numpy(),
            groups=groups,
        )
        probability[validation] = search.predict_proba(
            engineered.loc[validation, list(spec.features)]
        )[:, 1]
        fit_rows.append({
            "outer_center": center, "development_n": int(development.sum()),
            "validation_n": int(validation.sum()), "locked_parameters": json.dumps(parameters[center], sort_keys=True),
        })
    if not np.isfinite(probability).all():
        raise AssertionError("Every retained patient must receive one locked LOCO probability.")
    y = engineered[TARGET].astype(int).to_numpy()
    groups = engineered[CENTER].astype(int).to_numpy()
    metrics = probability_metrics(y, probability)
    intervals = bootstrap_metric_ci(y, probability, n_bootstrap=N_BOOTSTRAP, seed=SEED, groups=groups)
    return engineered, probability, metrics, intervals, fit_rows


# %%
base = load_cohort(DATA)
parameters = locked_parameters()
range_cleaned, changed = apply_range_rules(base)
stage_consistent, stage_removed = apply_stage_consistency(base)
combined, combined_removed = apply_stage_consistency(range_cleaned)

scenarios = {
    "as_recorded_refit_with_original_locks": base,
    "current_predictor_plausible_ranges_to_missing": range_cleaned,
    "exclude_binary_stage_inconsistency": stage_consistent,
    "combined_data_quality_sensitivity": combined,
}

rows = []
secure_frames = []
all_fits = {}
for index, (name, scenario) in enumerate(scenarios.items()):
    engineered, probability, metrics, intervals, fits = fit_locked_loco(scenario, parameters)
    row = {
        "scenario": name, "n": int(len(engineered)), "events": int(engineered[TARGET].sum()),
        "model": "PI restricted random forest", "hyperparameter_policy": "original five-centre locks",
        "bootstrap_unit": "patient_within_center", "n_bootstrap": N_BOOTSTRAP, **metrics,
    }
    for metric, (lower, upper) in intervals.items():
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper
    rows.append(row)
    secure = engineered[["MajorID", CENTER, TARGET]].copy()
    secure.insert(0, "scenario", name)
    secure["pred_PI_restricted_rf"] = probability
    secure_frames.append(secure)
    all_fits[name] = fits

results = pd.DataFrame(rows)
results.to_csv(TABLES / "Table_source_model_data_quality_sensitivity.csv", index=False)
pd.concat(secure_frames, ignore_index=True).to_csv(
    SECURE / "SOURCE_MODEL_DATA_QUALITY_SENSITIVITY_PREDICTIONS_SECURE.csv.gz",
    index=False, compression="gzip",
)

audit = {
    "scenarios": list(scenarios), "range_rules": RANGES,
    "values_set_to_missing_by_variable": changed,
    "binary_stage_inconsistencies_excluded": stage_removed,
    "combined_inconsistencies_excluded": combined_removed,
    "model": "PI restricted random forest",
    "hyperparameters": "original locked parameters; no model or feature selection",
    "fold_preprocessing": "imputation and all estimated preprocessing fitted in each development fold",
    "patient_level_predictions_public": False,
    "fits": all_fits,
}
(OUTPUTS / "SOURCE_MODEL_DATA_QUALITY_SENSITIVITY_AUDIT.json").write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
        default=lambda value: value.item() if hasattr(value, "item") else str(value),
    ) + "\n",
    encoding="utf-8",
)
print(results[["scenario", "n", "events", "roc_auc", "oe_ratio", "calibration_slope", "brier"]].to_string(index=False))
