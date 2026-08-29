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
# # Common-feature external transportability stress test
#
# This analysis uses the same six routinely available variables in the local
# five-centre cohort and each public database. Public outcomes remain explicitly
# non-equivalent stress tests rather than confirmatory external validation.

# %%
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (
    CENTER,
    N_JOBS,
    RANDOM_STATE,
    TARGET,
    harmonize_gender_values,
    load_cohort,
    paired_auc_difference,
)
from inspire_external_validation import build_inspire_cohort
from loco_analysis import (
    BASE_MODELS,
    FeatureSetSpec,
    bootstrap_metric_ci,
    build_loco_search,
    calibration_curve_rows,
    engineer_loco_features,
    probability_metrics,
)
from mimic_external_validation import build_mimic_cohort


COMMON = FeatureSetSpec(
    name="C6",
    continuous=("Age", "LogPreopCr", "PreopHb"),
    categorical=("Gender", "Diabetes", "Gastrocolorectal"),
    role="public_database_common_feature_stress_test",
)
ALL_MODELS = BASE_MODELS + ("soft_voting",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-data", required=True, type=Path)
    parser.add_argument("--external-dataset", required=True, choices=("inspire", "mimic"))
    parser.add_argument("--external-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--preop-window-hours", type=float, default=24.0)
    parser.add_argument("--analysis-window-hours", type=float, default=168.0)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def _canonical_binary(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.casefold()
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    result.loc[text.isin({"0", "0.0", "no", "否"})] = "0"
    result.loc[text.isin({"1", "1.0", "yes", "是"})] = "1"
    return result


def _canonical_cancer_site(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    result.loc[numeric.eq(1)] = "1"
    result.loc[numeric.eq(2)] = "2"
    return result


def prepare_development(path: Path) -> pd.DataFrame:
    cohort = engineer_loco_features(load_cohort(path)).reset_index(drop=True)
    return cohort[[CENTER, TARGET] + list(COMMON.features)].copy()


def prepare_external(frame: pd.DataFrame) -> pd.DataFrame:
    required = {TARGET, "Age", "PreopCr", "PreopHb", "Gender", "Diabetes", "Gastrocolorectal"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"External cohort is missing common variables: {missing}")
    result = frame.copy()
    result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
    result["Age"] = pd.to_numeric(result["Age"], errors="coerce")
    # INSPIRE and MIMIC laboratory values are expressed as mg/dL for
    # creatinine and g/dL for haemoglobin. The local source dictionary uses
    # micromol/L and g/L, respectively. Convert before applying the frozen
    # development model.
    result["PreopCr"] = pd.to_numeric(result["PreopCr"], errors="coerce") * 88.4
    result["PreopHb"] = pd.to_numeric(result["PreopHb"], errors="coerce") * 10.0
    result["LogPreopCr"] = np.log(result["PreopCr"].where(result["PreopCr"] > 0))
    gender = harmonize_gender_values(result["Gender"])
    result["Gender"] = gender.astype(object).where(gender.notna(), np.nan)
    diabetes = _canonical_binary(result["Diabetes"])
    result["Diabetes"] = diabetes.astype(object).where(diabetes.notna(), np.nan)
    cancer = _canonical_cancer_site(result["Gastrocolorectal"])
    result["Gastrocolorectal"] = cancer.astype(object).where(cancer.notna(), np.nan)
    result = result.loc[
        result[TARGET].isin([0, 1]) & result["Gastrocolorectal"].notna()
    ].copy()
    result[TARGET] = result[TARGET].astype(int)
    if result[TARGET].nunique() < 2:
        raise ValueError("Matched external cohort does not contain both outcome classes.")
    return result[[TARGET] + list(COMMON.features)].reset_index(drop=True)


def build_external(args: argparse.Namespace) -> pd.DataFrame:
    if args.external_dataset == "inspire":
        raw = build_inspire_cohort(
            args.external_root,
            department="",
            analysis_window_hours=args.analysis_window_hours,
            gastrocolorectal_only=True,
        )
    else:
        raw = build_mimic_cohort(
            args.external_root,
            preop_window_hours=args.preop_window_hours,
        )
    return prepare_external(raw)


def development_loco_predictions(
    cohort: pd.DataFrame,
    *,
    fast: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    centers = sorted(cohort[CENTER].unique().astype(int).tolist())
    predictions = {model: np.full(len(cohort), np.nan) for model in BASE_MODELS}
    locks: list[dict[str, object]] = []
    for held_out in centers:
        train = cohort[CENTER].ne(held_out).to_numpy()
        test = ~train
        groups = cohort.loc[train, CENTER].to_numpy(dtype=int)
        y_train = cohort.loc[train, TARGET].to_numpy(dtype=int)
        fold_probabilities = []
        for model in BASE_MODELS:
            search = build_loco_search(
                COMMON,
                model,
                n_inner_centers=len(np.unique(groups)),
                fast=fast,
            )
            search.fit(
                cohort.loc[train, list(COMMON.features)],
                y_train,
                groups=groups,
            )
            risk = search.predict_proba(
                cohort.loc[test, list(COMMON.features)]
            )[:, 1]
            predictions[model][test] = risk
            fold_probabilities.append(risk)
            locks.append(
                {
                    "outer_center": int(held_out),
                    "model": model,
                    "best_params": {key: str(value) for key, value in search.best_params_.items()},
                    "inner_best_neg_brier": float(search.best_score_),
                    "selection_data": "four_training_centers_only",
                }
            )
    predictions["soft_voting"] = np.mean(
        np.vstack([predictions[model] for model in BASE_MODELS]), axis=0
    )
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("Incomplete common-feature LOCO predictions.")
    return predictions, locks


def fit_full_and_predict(
    development: pd.DataFrame,
    external: pd.DataFrame,
    *,
    fast: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    groups = development[CENTER].to_numpy(dtype=int)
    y = development[TARGET].to_numpy(dtype=int)
    predictions: dict[str, np.ndarray] = {}
    locks: list[dict[str, object]] = []
    for model in BASE_MODELS:
        search = build_loco_search(
            COMMON,
            model,
            n_inner_centers=len(np.unique(groups)),
            fast=fast,
        )
        search.fit(development[list(COMMON.features)], y, groups=groups)
        predictions[model] = search.predict_proba(external[list(COMMON.features)])[:, 1]
        locks.append(
            {
                "model": model,
                "best_params": {key: str(value) for key, value in search.best_params_.items()},
                "inner_best_neg_brier": float(search.best_score_),
                "selection_data": "all_five_development_centers_grouped_only",
            }
        )
    predictions["soft_voting"] = np.mean(
        np.vstack([predictions[model] for model in BASE_MODELS]), axis=0
    )
    return predictions, locks


def metric_table(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    bootstrap: int,
    groups: np.ndarray | None,
    cohort_label: str,
) -> pd.DataFrame:
    rows = []
    for index, model in enumerate(ALL_MODELS):
        risk = predictions[model]
        row = probability_metrics(y, risk)
        intervals = bootstrap_metric_ci(
            y,
            risk,
            n_bootstrap=bootstrap,
            seed=RANDOM_STATE + 60_000 + index,
            groups=groups,
        )
        for metric, (lower, upper) in intervals.items():
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        row.update({"cohort": cohort_label, "feature_set": COMMON.name, "model": model})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_auc_table(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    bootstrap: int,
) -> pd.DataFrame:
    rows = []
    for index, model in enumerate(("restricted_rf", "gradient_boosting", "soft_voting")):
        result = paired_auc_difference(
            y,
            predictions[model],
            predictions["ridge"],
            n_bootstrap=bootstrap,
            seed=RANDOM_STATE + 70_000 + index,
        )
        result.update({"candidate": model, "reference": "ridge", "feature_set": COMMON.name})
        rows.append(result)
    return pd.DataFrame(rows)


def feature_coverage(
    development: pd.DataFrame,
    external: pd.DataFrame,
    external_name: str,
) -> pd.DataFrame:
    rows = []
    for cohort_name, frame in (("development", development), (external_name, external)):
        for feature in COMMON.features:
            rows.append(
                {
                    "cohort": cohort_name,
                    "feature": feature,
                    "n": int(len(frame)),
                    "missing_n": int(frame[feature].isna().sum()),
                    "missing_rate": float(frame[feature].isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)
    development = prepare_development(args.development_data)
    external = build_external(args)
    bootstrap = min(args.bootstrap, 100) if args.fast else args.bootstrap

    development_predictions, outer_locks = development_loco_predictions(
        development, fast=args.fast
    )
    external_predictions, final_locks = fit_full_and_predict(
        development, external, fast=args.fast
    )
    development_metrics = metric_table(
        development[TARGET].to_numpy(dtype=int),
        development_predictions,
        bootstrap=bootstrap,
        groups=development[CENTER].to_numpy(dtype=int),
        cohort_label="development_loco_oof",
    )
    external_metrics = metric_table(
        external[TARGET].to_numpy(dtype=int),
        external_predictions,
        bootstrap=bootstrap,
        groups=None,
        cohort_label=f"external_{args.external_dataset}_matched_gastrocolorectal",
    )
    comparisons = paired_auc_table(
        external[TARGET].to_numpy(dtype=int),
        external_predictions,
        bootstrap=bootstrap,
    )
    calibration = calibration_curve_rows(
        external[TARGET].to_numpy(dtype=int),
        {(COMMON.name, model): risk for model, risk in external_predictions.items()},
    )

    development_metrics.to_csv(args.output / "development_loco_metrics.csv", index=False)
    external_metrics.to_csv(args.output / "external_metrics.csv", index=False)
    comparisons.to_csv(args.output / "paired_auc_differences.csv", index=False)
    calibration.to_csv(args.output / "external_calibration_curve.csv", index=False)
    feature_coverage(development, external, args.external_dataset).to_csv(
        args.output / "common_feature_coverage.csv", index=False
    )
    lock = {
        "analysis_role": "exploratory_public_database_transportability_stress_test",
        "external_dataset": args.external_dataset,
        "population": "gastric_or_colorectal_diagnosis_only",
        "development_n": int(len(development)),
        "development_events": int(development[TARGET].sum()),
        "external_n": int(len(external)),
        "external_events": int(external[TARGET].sum()),
        "feature_set": {
            "name": COMMON.name,
            "continuous": list(COMMON.continuous),
            "categorical": list(COMMON.categorical),
        },
        "unit_harmonization": {
            "PreopCr": "public mg/dL multiplied by 88.4 to local micromol/L",
            "PreopHb": "public g/dL multiplied by 10 to local g/L",
        },
        "models": list(ALL_MODELS),
        "selection_metric": "negative_brier_score",
        "outer_validation": "five_center_leave_one_center_out",
        "external_outcome_not_equivalent_to_source_recorded_PostopAKI": True,
        "external_outcomes_not_used_for_model_or_hyperparameter_selection": True,
        "patient_level_predictions_saved": False,
        "bootstrap": bootstrap,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "python": sys.version,
        "platform": platform.platform(),
        "outer_fold_locks": outer_locks,
        "final_fit_locks": final_locks,
    }
    (args.output / "model_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Common-feature external stress test complete: {args.output}")


if __name__ == "__main__":
    main()
