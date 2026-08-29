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
# # Bidirectional harmonized transport across MIMIC-IV and eICU
#
# The model specification, prediction landmark, predictors, and operational
# creatinine endpoint are held constant. Each direction is fit once in one
# database and evaluated without local recalibration in the other database.

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

BASE = Path(str(_release_path('source')))
ROOT = Path(str(_release_path('analysis')))
sys.path.insert(0, str(BASE))

from loco_analysis import SUMMARY_METRICS, probability_metrics  # noqa: E402

SEED = 20260827
N_BOOTSTRAP = 1000
PREDICTORS = ("age", "log_baseline_creatinine", "sex")


def load_harmonized(database: str) -> pd.DataFrame:
    if database == "MIMIC-IV":
        path = ROOT / "secure_work" / "MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz"
        frame = pd.read_csv(path)
        age = pd.to_numeric(frame["age"], errors="coerce")
        sex = frame["gender"].astype("string").str.upper().map({"M": "Male", "F": "Female"})
        cluster = frame["subject_id"].astype("string")
        hospital = pd.Series("MIMIC-IV", index=frame.index, dtype="string")
    elif database == "eICU":
        path = ROOT / "eicu" / "secure" / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz"
        frame = pd.read_csv(path)
        age = pd.to_numeric(frame["age_num"], errors="coerce")
        sex = frame["gender"].astype("string").str.strip().str.title()
        cluster = frame["hospitalid"].astype("string")
        hospital = frame["hospitalid"].astype("string")
    else:
        raise ValueError(database)

    baseline = pd.to_numeric(frame["baseline_creatinine"], errors="coerce")
    result = pd.DataFrame(
        {
            "database": database,
            "record_id": frame["reference_id"].astype("string"),
            "cluster": cluster,
            "hospital": hospital,
            "age": age,
            "sex": sex,
            "log_baseline_creatinine": np.log(baseline.where(baseline > 0)),
            "outcome": pd.to_numeric(frame["Y_longitudinal"], errors="coerce"),
            "dense": pd.to_numeric(frame["R_dense"], errors="coerce"),
        }
    )
    result = result.loc[result["dense"].eq(1) & result["outcome"].notna()].copy()
    result["outcome"] = result["outcome"].astype(int)
    if result["record_id"].duplicated().any():
        raise ValueError(f"{database} reference identifiers are not unique")
    return result.reset_index(drop=True)


def make_model() -> Pipeline:
    continuous = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("continuous", continuous, ["age", "log_baseline_creatinine"]),
            ("categorical", categorical, ["sex"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(C=0.25, solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )


def bootstrap_intervals(y, p, clusters=None, seed=SEED):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = {metric: [] for metric in SUMMARY_METRICS}
    if clusters is None:
        cluster_indices = None
    else:
        labels = np.asarray(clusters)
        unique = pd.unique(labels)
        cluster_indices = {label: np.flatnonzero(labels == label) for label in unique}
    for _ in range(N_BOOTSTRAP):
        if cluster_indices is None:
            indices = rng.choice(len(y), len(y), replace=True)
        else:
            drawn = rng.choice(list(cluster_indices), len(cluster_indices), replace=True)
            indices = np.concatenate([cluster_indices[label] for label in drawn])
        try:
            metrics = probability_metrics(y[indices], p[indices])
        except (ValueError, FloatingPointError):
            continue
        for metric in SUMMARY_METRICS:
            value = float(metrics[metric])
            if math.isfinite(value):
                sampled[metric].append(value)
    return {
        metric: (
            float(np.quantile(values, 0.025)) if values else math.nan,
            float(np.quantile(values, 0.975)) if values else math.nan,
        )
        for metric, values in sampled.items()
    }


def summarize_external(train_name, test_name, train, test, probabilities):
    point = probability_metrics(test["outcome"].to_numpy(), probabilities)
    cluster_bootstrap = test_name == "eICU"
    intervals = bootstrap_intervals(
        test["outcome"].to_numpy(),
        probabilities,
        clusters=test["hospital"].to_numpy() if cluster_bootstrap else None,
        seed=SEED + (0 if test_name == "eICU" else 1),
    )
    row = {
        "transport_direction": f"{train_name}_to_{test_name}",
        "training_database": train_name,
        "validation_database": test_name,
        "n_train": len(train),
        "events_train": int(train["outcome"].sum()),
        "n_validation": len(test),
        "events_validation": int(test["outcome"].sum()),
        "prediction_landmark": "ICU admission",
        "endpoint": "0-168 h creatinine-defined KDIGO AKI among dense-reference stays",
        "model_specification": "fixed ridge logistic regression; age, sex, log baseline creatinine; C=0.25",
        "local_recalibration": False,
        "bootstrap_unit": "hospital" if cluster_bootstrap else "analytic_record",
        **point,
    }
    for metric, (lower, upper) in intervals.items():
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper
    return row


# %%
mimic = load_harmonized("MIMIC-IV")
eicu = load_harmonized("eICU")
datasets = {"MIMIC-IV": mimic, "eICU": eicu}
summary_rows = []
hospital_rows = []
secure_predictions = []

for train_name, test_name in (("MIMIC-IV", "eICU"), ("eICU", "MIMIC-IV")):
    train = datasets[train_name]
    test = datasets[test_name]
    model = make_model()
    model.fit(train[list(PREDICTORS)], train["outcome"])
    probabilities = model.predict_proba(test[list(PREDICTORS)])[:, 1]
    summary_rows.append(summarize_external(train_name, test_name, train, test, probabilities))

    secure = test[["database", "record_id", "hospital", "outcome"]].copy()
    secure.insert(0, "transport_direction", f"{train_name}_to_{test_name}")
    secure["predicted_probability"] = probabilities
    secure_predictions.append(secure)

    if test_name == "eICU":
        for hospital, indices in test.groupby("hospital", sort=True).groups.items():
            indices = np.asarray(list(indices), dtype=int)
            y_h = test.loc[indices, "outcome"].to_numpy(dtype=int)
            p_h = probabilities[indices]
            metric = probability_metrics(y_h, p_h)
            hospital_rows.append(
                {
                    "transport_direction": f"{train_name}_to_{test_name}",
                    "hospital": hospital,
                    **metric,
                    "interpretation": "descriptive hospital-level calibration; no local recalibration",
                }
            )

summary = pd.DataFrame(summary_rows)
summary.to_csv(ROOT / "tables" / "Table_public_harmonized_bidirectional_transport.csv", index=False)
pd.DataFrame(hospital_rows).to_csv(
    ROOT / "tables" / "Table_public_harmonized_eicu_hospital_calibration.csv", index=False
)
pd.concat(secure_predictions, ignore_index=True).to_csv(
    ROOT / "secure_work" / "PUBLIC_HARMONIZED_TRANSPORT_PREDICTIONS_SECURE.csv.gz",
    index=False,
    compression="gzip",
)

audit = {
    "analysis": "bidirectional public-database transport with a fixed harmonized model specification",
    "datasets": {
        name: {"n": len(frame), "events": int(frame["outcome"].sum())}
        for name, frame in datasets.items()
    },
    "fixed_predictors": list(PREDICTORS),
    "model": "ridge logistic regression with C=0.25",
    "endpoint": "0-168 h creatinine-defined KDIGO AKI in dense-reference surgical-ICU cohorts",
    "prediction_landmark": "ICU admission",
    "limits": [
        "This validates a minimal public-data model, not the five-centre surgical source model.",
        "Dense-reference conditioning changes the target population and does not recover full-cohort performance.",
        "MIMIC-IV record intervals do not account for unobserved site clustering; eICU intervals resample hospitals.",
        "No local recalibration was performed before external evaluation.",
    ],
    "patient_level_outputs_delivered": False,
}
(ROOT / "outputs" / "PUBLIC_HARMONIZED_TRANSPORT_AUDIT.json").write_text(
    json.dumps(audit, indent=2), encoding="utf-8"
)
print(summary.to_string(index=False))
