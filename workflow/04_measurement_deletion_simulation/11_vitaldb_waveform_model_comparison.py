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
# # VitalDB clinical, duration-adjusted, and waveform-enhanced ridge comparison
#
# The original patient-disjoint split and clinical-table ridge specification
# are reconstructed exactly. A duration-adjusted clinical comparator prevents
# waveform coverage variables from obtaining an unfair advantage by proxying
# operation length. No held-out outcome is used for feature selection or tuning.

# %%
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260830
N_BOOTSTRAP = 1000
BASELINE_EXPECTED_AUC = 0.7044103847356897
METRICS = ("auc", "brier", "oe", "calibration_intercept", "calibration_slope")
NUMERIC_BASELINE = [
    "age", "height", "weight", "bmi", "asa", "preop_hb", "preop_plt", "preop_na", "preop_k",
    "preop_gluc", "preop_alb", "preop_bun", "preop_cr", "baseline_cr", "intraop_ebl", "intraop_uo",
    "intraop_rbc", "intraop_ffp", "intraop_crystalloid", "intraop_colloid", "intraop_eph",
    "intraop_phe", "intraop_epi",
]
CATEGORICAL = [
    "sex", "emop", "department", "optype", "approach", "ane_type", "preop_htn", "preop_dm"
]
WAVEFORM_FEATURES = [
    "art_map_raw_samples", "art_map_covered_seconds", "art_map_coverage_fraction",
    "art_map_median", "art_map_p05", "art_map_sd",
    "art_map_below_65_minutes", "art_map_below_65_fraction_observed",
    "art_map_below_60_minutes", "art_map_below_60_fraction_observed",
    "art_map_below_55_minutes", "art_map_below_55_fraction_observed",
    "art_map_deficit_65_mmHg_minutes", "art_map_twa_deficit_65",
    "hr_raw_samples", "hr_covered_seconds", "hr_coverage_fraction", "hr_median", "hr_sd",
    "hr_above_100_fraction_observed", "nibp_map_count", "nibp_map_median",
]
PRIMARY_COMPARATOR = "duration_adjusted_clinical_ridge"
DELTA_COMPARISONS = {
    "waveform_minus_duration_adjusted_clinical_paired_delta": PRIMARY_COMPARATOR,
    "waveform_minus_historical_clinical_paired_delta": "clinical_table_ridge",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(C=0.25, solver="liblinear", max_iter=2000, random_state=SEED),
            ),
        ]
    )


def digest_ids(values: pd.Series) -> str:
    text = "\n".join(map(str, values.astype(int).tolist()))
    return hashlib.sha256(text.encode()).hexdigest()


def paired_bootstrap(y: np.ndarray, predictions: dict[str, np.ndarray], stress) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 17)
    rows = []
    n = len(y)
    for replicate in range(N_BOOTSTRAP):
        index = rng.integers(0, n, size=n)
        if np.unique(y[index]).size < 2:
            continue
        for label, probability in predictions.items():
            metrics = stress.weighted_metrics(y[index], probability[index])
            rows.append({"replicate": replicate, "model": label, **metrics})
    return pd.DataFrame(rows)


def comparison_table(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    bootstrap: pd.DataFrame,
    stress,
) -> tuple[pd.DataFrame, dict[str, object]]:
    point = {label: stress.weighted_metrics(y, probability) for label, probability in predictions.items()}
    rows: list[dict[str, object]] = []
    for label in predictions:
        group = bootstrap.loc[bootstrap.model.eq(label)]
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "comparison": "model_performance",
                    "model": label,
                    "metric": metric,
                    "estimate": float(point[label][metric]),
                    "ci_lower": float(values.quantile(0.025)) if len(values) else np.nan,
                    "ci_upper": float(values.quantile(0.975)) if len(values) else np.nan,
                    "bootstrap_replicates": int(group.replicate.nunique()),
                }
            )

    wide = bootstrap.pivot(index="replicate", columns="model", values=list(METRICS))
    delta_summary: dict[str, object] = {}
    for comparison, comparator in DELTA_COMPARISONS.items():
        comparison_summary: dict[str, object] = {}
        for metric in METRICS:
            delta = (
                wide[(metric, "waveform_enhanced_ridge")]
                - wide[(metric, comparator)]
            ).dropna()
            estimate = float(
                point["waveform_enhanced_ridge"][metric]
                - point[comparator][metric]
            )
            lower = float(delta.quantile(0.025)) if len(delta) else np.nan
            upper = float(delta.quantile(0.975)) if len(delta) else np.nan
            rows.append(
                {
                    "comparison": comparison,
                    "model": f"waveform_enhanced_ridge_minus_{comparator}",
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_replicates": int(len(delta)),
                }
            )
            comparison_summary[metric] = {
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "direction_note": (
                    "positive favours waveform model" if metric == "auc"
                    else "negative favours waveform model" if metric == "brier"
                    else "difference is descriptive; closeness to the metric target determines calibration"
                ),
            }
        delta_summary[comparison] = comparison_summary
    return pd.DataFrame(rows), {"point_metrics": point, "paired_deltas": delta_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-csv", required=True, type=Path)
    parser.add_argument("--waveform-features", required=True, type=Path)
    parser.add_argument("--stress-module", required=True, type=Path)
    parser.add_argument("--secure-output", required=True, type=Path)
    parser.add_argument("--table-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()

    stress = load_module(args.stress_module, "waveform_stress")
    cases = pd.read_csv(args.case_csv, low_memory=False)
    waveform = pd.read_csv(args.waveform_features, low_memory=False)
    if waveform.caseid.duplicated().any():
        raise ValueError("Waveform feature file must have one row per caseid")
    cases_per_patient = cases.groupby("subjectid")["caseid"].nunique()
    single_patients = cases_per_patient.index[cases_per_patient.eq(1)]
    eligible = cases.loc[
        cases["adult"] & cases["dense_reference"] & cases["subjectid"].isin(single_patients)
    ].copy()
    operation_seconds = pd.to_numeric(eligible["opend"], errors="coerce") - pd.to_numeric(
        eligible["opstart"], errors="coerce"
    )
    eligible["operation_duration_hours"] = operation_seconds.where(operation_seconds > 0) / 3600.0
    eligible["__row_order"] = np.arange(len(eligible))
    eligible = eligible.merge(waveform, on="caseid", how="left", validate="one_to_one", sort=False)
    eligible = eligible.sort_values("__row_order", kind="stable").reset_index(drop=True)

    baseline_numeric = [name for name in NUMERIC_BASELINE if name in eligible and eligible[name].notna().any()]
    categorical = [name for name in CATEGORICAL if name in eligible and eligible[name].notna().any()]
    waveform_numeric = [name for name in WAVEFORM_FEATURES if name in eligible and eligible[name].notna().any()]
    y = eligible["creatinine_event_168h"].astype(int).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_index, test_index = next(splitter.split(eligible, y, groups=eligible["subjectid"]))
    train = eligible.iloc[train_index]
    test = eligible.iloc[test_index]
    if len(test) != 324 or int(test["creatinine_event_168h"].sum()) != 46:
        raise RuntimeError("The frozen VitalDB held-out split no longer matches 324 patients and 46 events")

    probabilities = {}
    model_specs = {
        "clinical_table_ridge": baseline_numeric,
        "duration_adjusted_clinical_ridge": baseline_numeric + ["operation_duration_hours"],
        "waveform_enhanced_ridge": baseline_numeric + ["operation_duration_hours"] + waveform_numeric,
    }
    for label, numeric in model_specs.items():
        features = numeric + categorical
        model = make_model(numeric, categorical)
        model.fit(train[features], train["creatinine_event_168h"].astype(int))
        probabilities[label] = model.predict_proba(test[features])[:, 1]

    y_test = test["creatinine_event_168h"].astype(int).to_numpy()
    bootstrap = paired_bootstrap(y_test, probabilities, stress)
    table, comparison_audit = comparison_table(y_test, probabilities, bootstrap, stress)
    baseline_auc = float(table.loc[
        table.model.eq("clinical_table_ridge") & table.metric.eq("auc"), "estimate"
    ].iloc[0])
    if abs(baseline_auc - BASELINE_EXPECTED_AUC) > 1e-6:
        raise RuntimeError(f"Baseline model did not reproduce: {baseline_auc} != {BASELINE_EXPECTED_AUC}")

    secure = test[["caseid", "subjectid", "creatinine_event_168h"]].copy()
    for label, probability in probabilities.items():
        secure[f"risk_{label}"] = probability
    args.secure_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    secure.to_csv(args.secure_output, index=False, compression="gzip")
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table_output, index=False)

    usable = test.get("art_map_duration_features_usable", pd.Series(False, index=test.index)).fillna(False)
    audit = {
        "analysis": "prespecified VitalDB waveform-enhanced ridge comparison",
        "eligible_n": int(len(eligible)),
        "eligible_events": int(eligible.creatinine_event_168h.sum()),
        "train_n": int(len(train)),
        "train_events": int(train.creatinine_event_168h.sum()),
        "test_n": int(len(test)),
        "test_events": int(test.creatinine_event_168h.sum()),
        "train_caseid_sha256": digest_ids(train.caseid),
        "test_caseid_sha256": digest_ids(test.caseid),
        "baseline_numeric_predictors": baseline_numeric,
        "duration_adjustment_predictor": "operation_duration_hours",
        "categorical_predictors": categorical,
        "waveform_predictors": waveform_numeric,
        "predictor_roles": {
            "clinical_table_ridge": "exact historical continuity model",
            "duration_adjusted_clinical_ridge": (
                "fair primary comparator controlling operation length before waveform coverage is added"
            ),
            "waveform_enhanced_ridge": (
                "duration-adjusted clinical predictors plus prespecified waveform summaries"
            ),
        },
        "operation_duration_hours": {
            "eligible_nonmissing_n": int(eligible.operation_duration_hours.notna().sum()),
            "eligible_median": float(eligible.operation_duration_hours.median()),
            "eligible_q025": float(eligible.operation_duration_hours.quantile(0.025)),
            "eligible_q975": float(eligible.operation_duration_hours.quantile(0.975)),
        },
        "test_usable_art_map_n": int(usable.sum()),
        "test_usable_art_map_percent": float(100 * usable.mean()),
        "baseline_auc_reproduced": baseline_auc,
        "baseline_expected_auc": BASELINE_EXPECTED_AUC,
        "bootstrap_replicates": N_BOOTSTRAP,
        "comparison_metrics": comparison_audit,
        "leakage_boundary": (
            "All clinical predictors and waveform summaries end at opend; the retained creatinine "
            "reference outcome starts after opend. The frozen held-out outcomes are not used for "
            "feature selection, tuning, imputation, scaling, or fitting."
        ),
        "interpretation_boundary": (
            "descriptive incremental value against the duration-adjusted comparator and "
            "stronger-real-model stress-test input; "
            "not model selection, causal hypotension evidence, or clinical impact"
        ),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
