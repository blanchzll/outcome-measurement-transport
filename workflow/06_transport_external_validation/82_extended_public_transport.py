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
# # Outcome-blind extended public-database transport model
#
# The extended specification is fixed by common pre-landmark availability and
# unit compatibility, not by validation performance. All imputation, missingness
# indicators and scaling are fitted in the training database only.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import Parallel, delayed

BASE = Path(str(_release_path('source')))
ROOT = Path(str(_release_path('analysis')))
sys.path.insert(0, str(BASE))
from loco_analysis import SUMMARY_METRICS, probability_metrics  # noqa: E402

SEED = 20260827
N_BOOTSTRAP = 500
MINIMAL_CONTINUOUS = ["age", "log_baseline_creatinine"]
EXTENDED_CONTINUOUS = MINIMAL_CONTINUOUS + [
    "baseline_albumin", "baseline_bun", "baseline_glucose", "baseline_sodium",
    "baseline_potassium", "baseline_hemoglobin", "baseline_wbc", "baseline_platelet",
]
CATEGORICAL = ["sex"]
MODEL_SPECS = {
    "minimal": MINIMAL_CONTINUOUS,
    "extended_common": EXTENDED_CONTINUOUS,
}
UNITS = {
    "age": "years", "log_baseline_creatinine": "natural log mg/dL",
    "baseline_albumin": "g/dL", "baseline_bun": "mg/dL",
    "baseline_glucose": "mg/dL", "baseline_sodium": "mmol/L",
    "baseline_potassium": "mmol/L", "baseline_hemoglobin": "g/dL",
    "baseline_wbc": "K/uL", "baseline_platelet": "K/uL", "sex": "category",
}


def load_harmonized(database: str) -> pd.DataFrame:
    if database == "MIMIC-IV":
        d = pd.read_csv(ROOT / "secure_work" / "MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
        source = d
        result = pd.DataFrame({
            "record_id": d.reference_id.astype("string"),
            "hospital": "MIMIC-IV",
            "cluster": d.subject_id.astype("string"),
            "age": pd.to_numeric(d.age, errors="coerce"),
            "sex": d.gender.astype("string").str.upper().map({"M": "Male", "F": "Female"}),
            "outcome": pd.to_numeric(d.Y_longitudinal, errors="coerce"),
            "dense": pd.to_numeric(d.R_dense, errors="coerce"),
        })
    elif database == "eICU":
        d = pd.read_csv(ROOT / "eicu" / "secure" / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
        labs = pd.read_csv(ROOT / "eicu" / "secure" / "EICU_COMMON_PREDICTORS_SECURE.csv.gz")
        d = d.merge(labs.drop(columns="patientunitstayid"), on="reference_id", how="left", validate="one_to_one")
        source = d
        result = pd.DataFrame({
            "record_id": d.reference_id.astype("string"),
            "hospital": d.hospitalid.astype("string"),
            "cluster": d.hospitalid.astype("string"),
            "age": pd.to_numeric(d.age_num, errors="coerce"),
            "sex": d.gender.astype("string").str.strip().str.title(),
            "outcome": pd.to_numeric(d.Y_longitudinal, errors="coerce"),
            "dense": pd.to_numeric(d.R_dense, errors="coerce"),
        })
    else:
        raise ValueError(database)

    baseline = pd.to_numeric(source.baseline_creatinine, errors="coerce")
    result["log_baseline_creatinine"] = np.log(baseline.where(baseline > 0))
    for column in EXTENDED_CONTINUOUS[2:]:
        result[column] = pd.to_numeric(source[column], errors="coerce")
    result["database"] = database
    result = result.loc[result.dense.eq(1) & result.outcome.notna()].copy()
    result["outcome"] = result.outcome.astype(int)
    assert result.record_id.is_unique
    return result.reset_index(drop=True)


def make_model(continuous: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("continuous", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), continuous),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("model", LogisticRegression(C=0.25, solver="lbfgs", max_iter=5000, random_state=SEED)),
    ])


def bootstrap_intervals(y, p, clusters, cluster_bootstrap, seed):
    y, p, clusters = np.asarray(y, int), np.asarray(p, float), np.asarray(clusters)
    rng = np.random.default_rng(seed)
    if cluster_bootstrap:
        unique = pd.unique(clusters)
        index = {label: np.flatnonzero(clusters == label) for label in unique}

    child_seeds = np.random.SeedSequence(seed).spawn(N_BOOTSTRAP)

    def one_draw(child_seed):
        draw_rng = np.random.default_rng(child_seed)
        if cluster_bootstrap:
            drawn = draw_rng.choice(unique, len(unique), replace=True)
            take = np.concatenate([index[label] for label in drawn])
        else:
            take = draw_rng.choice(len(y), len(y), replace=True)
        if np.unique(y[take]).size < 2:
            return None
        return probability_metrics(y[take], p[take])

    draws = Parallel(n_jobs=8, prefer="processes")(
        delayed(one_draw)(child_seed) for child_seed in child_seeds
    )
    samples = {metric: [] for metric in SUMMARY_METRICS}
    for metric in (draw for draw in draws if draw is not None):
        for name in SUMMARY_METRICS:
            value = float(metric[name])
            if math.isfinite(value):
                samples[name].append(value)
    return {name: (float(np.quantile(x, .025)), float(np.quantile(x, .975))) if x else (math.nan, math.nan) for name, x in samples.items()}


# %%
datasets = {name: load_harmonized(name) for name in ("MIMIC-IV", "eICU")}
availability = []
for database, d in datasets.items():
    for variable in EXTENDED_CONTINUOUS + CATEGORICAL:
        availability.append({
            "database": database, "cohort": "dense_reference", "predictor": variable,
            "unit": UNITS[variable], "n": len(d), "n_observed": int(d[variable].notna().sum()),
            "missing_fraction": float(d[variable].isna().mean()),
            "selection_status": "prespecified_common_predictor",
        })
pd.DataFrame(availability).to_csv(ROOT / "tables" / "Table_public_extended_predictor_availability.csv", index=False)

summary_rows, prediction_rows = [], []
for direction_index, (train_name, test_name) in enumerate((("MIMIC-IV", "eICU"), ("eICU", "MIMIC-IV"))):
    train, test = datasets[train_name], datasets[test_name]
    for model_index, (specification, continuous) in enumerate(MODEL_SPECS.items()):
        predictors = continuous + CATEGORICAL
        model = make_model(continuous)
        model.fit(train[predictors], train.outcome)
        probability = model.predict_proba(test[predictors])[:, 1]
        point = probability_metrics(test.outcome.to_numpy(), probability)
        cluster_bootstrap = test_name == "eICU"
        interval = bootstrap_intervals(
            test.outcome.to_numpy(), probability, test.cluster.to_numpy(), cluster_bootstrap,
            SEED + direction_index * 100 + model_index,
        )
        row = {
            "transport_direction": f"{train_name}_to_{test_name}",
            "model_specification": specification,
            "predictors": "|".join(predictors),
            "n_train": len(train), "events_train": int(train.outcome.sum()),
            "n_validation": len(test), "events_validation": int(test.outcome.sum()),
            "prediction_landmark": "ICU admission",
            "endpoint": "0-168 h creatinine-defined AKI among dense-reference stays",
            "local_recalibration": False,
            "bootstrap_unit": "hospital" if cluster_bootstrap else "analytic_record",
            **point,
        }
        for metric, (lower, upper) in interval.items():
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        summary_rows.append(row)
        pred = test[["database", "record_id", "hospital", "outcome"]].copy()
        pred.insert(0, "transport_direction", f"{train_name}_to_{test_name}")
        pred.insert(1, "model_specification", specification)
        pred["predicted_probability"] = probability
        prediction_rows.append(pred)

summary = pd.DataFrame(summary_rows)
summary.to_csv(ROOT / "tables" / "Table_public_extended_bidirectional_transport.csv", index=False)
pd.concat(prediction_rows, ignore_index=True).to_csv(
    ROOT / "secure_work" / "PUBLIC_EXTENDED_TRANSPORT_PREDICTIONS_SECURE.csv.gz", index=False, compression="gzip"
)

audit = {
    "analysis": "prespecified extended common-variable bidirectional transport",
    "model_variants": {name: columns + CATEGORICAL for name, columns in MODEL_SPECS.items()},
    "selection_rule": "common pre-landmark availability and compatible units; no outcome or validation metric used",
    "preprocessing": "training-database median imputation with missingness indicators, scaling and categorical encoding",
    "model": "ridge logistic regression; C=0.25 fixed",
    "local_recalibration": False,
    "bootstrap": {"replicates": N_BOOTSTRAP, "eICU_unit": "hospital", "MIMIC_unit": "analytic record"},
    "limits": [
        "The cohorts are conditioned on dense post-landmark creatinine measurement.",
        "This transports a public-data model and does not externally validate the source surgery-end model.",
        "Improved discrimination, if observed, is descriptive because the extension was not a new clinical model-development exercise.",
    ],
    "patient_level_outputs_delivered": False,
}
(ROOT / "outputs" / "PUBLIC_EXTENDED_TRANSPORT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(summary[["transport_direction", "model_specification", "n_validation", "events_validation", "roc_auc", "oe_ratio", "calibration_slope"]].to_string(index=False))
