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
# # Leakage-resistant analysis utilities for postoperative AKI prediction

# %%
"""Leakage-resistant analysis utilities for postoperative AKI prediction."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-aki")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "PostopAKI"
CENTER = "Center"
STABLE_ID = "MajorID"

# 预设两类目标数据版本：4014例（历史稿件）与3710例（当前重建队列）
EXPECTED_SOURCE_SPECS = {(4014, 155), (3710, 152)}
RANDOM_STATE = 20260817
N_JOBS = int(os.environ.get("AKI_N_JOBS", "1"))

# All variables are available no later than the end-of-surgery landmark.
CONTINUOUS_FEATURES = [
    "Age",
    "BMI",
    "PreopHb",
    "PreopAlb",
    "PreopCr",
    "OperationTime",
    "IntraopBloodLoss",
    "IntraopTransfusion",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Diabetes",
    "ASAGrade",
    "Gastrocolorectal",
    "SurgicalApproach",
    "IntraopVasoactive",
]

FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# These variables would violate the prediction landmark or are outcome-adjacent.
FORBIDDEN_PREFIXES = ("Postop", "NonOp")
FORBIDDEN_EXACT = {
    "AKIStage",
    "HospitalDays",
    "PostopHospitalDays",
    "ICUAdmission",
    "ICUDays",
    "VentilatorUse",
    "RRT",
    "Reoperation30d",
    "Readmission30d",
    "Mortality90d",
    "SurgicalComplications",
    "InfectionComplications",
    "Fistula",
    "MotilityDisorder",
    "Bleeding",
    "T_Stage",
    "N_Stage",
    "M_Stage",
    "TNM_Stage",
    "LymphNodesExamined",
    "PositiveLymphNodes",
}


@dataclass(frozen=True)
class CohortSplit:
    development: pd.DataFrame
    external: pd.DataFrame
    development_centers: tuple[int, ...]
    external_centers: tuple[int, ...]


def harmonize_gender_values(series: pd.Series) -> pd.Series:
    """Harmonise investigator-confirmed encodings and preserve unknowns as missing."""
    text = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    result.loc[text.isin({"1", "1.0", "m", "male", "男"})] = "Male"
    result.loc[text.isin({"0", "0.0", "2", "2.0", "f", "female", "女"})] = "Female"
    return result


def load_cohort(path: str | Path) -> pd.DataFrame:
    """Load and validate the sole source-of-truth cohort."""
    frame = pd.read_csv(path, low_memory=False)
    required = set(FEATURES + [TARGET, CENTER, "AKIStage"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mapping = {"No": 0, "Yes": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    mapped = frame[TARGET].map(mapping)
    if mapped.isna().any():
        bad = sorted(frame.loc[mapped.isna(), TARGET].astype(str).unique().tolist())
        raise ValueError(f"Unrecognized target values: {bad}")
    frame = frame.copy()
    frame[TARGET] = mapped.astype(int)
    frame[CENTER] = pd.to_numeric(frame[CENTER], errors="raise").astype(int)
    frame["Gender"] = harmonize_gender_values(frame["Gender"])

    override_rows = os.environ.get("AKI_EXPECTED_SOURCE_ROWS")
    override_events = os.environ.get("AKI_EXPECTED_SOURCE_EVENTS")
    if override_rows is not None and override_events is not None:
        try:
            override_rows_i = int(override_rows)
            override_events_i = int(override_events)
            expected = {(override_rows_i, override_events_i)}
        except ValueError as err:
            raise ValueError("Environment overrides AKI_EXPECTED_SOURCE_ROWS/AKI_EXPECTED_SOURCE_EVENTS must be integers.") from err
    else:
        expected = EXPECTED_SOURCE_SPECS

    observed = (len(frame), int(frame[TARGET].sum()))
    if observed not in expected:
        raise ValueError(
            "Source cohort integrity check failed: expected "
            f"one of {sorted(expected)} rows/events, found {observed}."
        )
    if STABLE_ID not in frame:
        raise ValueError(f"Stable source identifier {STABLE_ID} is required.")
    if frame[STABLE_ID].isna().any() or not frame[STABLE_ID].is_unique:
        raise ValueError(f"{STABLE_ID} must be complete and unique.")
    forbidden_selected = [
        name for name in FEATURES if name in FORBIDDEN_EXACT or name.startswith(FORBIDDEN_PREFIXES)
    ]
    if forbidden_selected:
        raise ValueError(f"Landmark-violating predictors selected: {forbidden_selected}")
    return frame


def split_by_center(
    frame: pd.DataFrame,
    development_centers: Iterable[int] = (3, 4, 5),
    external_centers: Iterable[int] = (1, 2),
) -> CohortSplit:
    development_centers = tuple(int(value) for value in development_centers)
    external_centers = tuple(int(value) for value in external_centers)
    if set(development_centers) & set(external_centers):
        raise ValueError("Development and external center sets overlap.")
    observed = set(frame[CENTER].unique())
    configured = set(development_centers) | set(external_centers)
    if observed != configured:
        raise ValueError(f"Center allocation must cover observed centers {sorted(observed)} exactly.")
    development = frame.loc[frame[CENTER].isin(development_centers)].copy()
    external = frame.loc[frame[CENTER].isin(external_centers)].copy()
    for label, cohort in [("development", development), ("external", external)]:
        if cohort.empty or cohort[TARGET].nunique() != 2:
            raise ValueError(f"{label} cohort does not contain both outcome classes.")
    return CohortSplit(development, external, development_centers, external_centers)


def cohort_audit(frame: pd.DataFrame, split: CohortSplit) -> dict:
    center_table = (
        frame.groupby(CENTER)[TARGET]
        .agg(n="size", events="sum", incidence="mean")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "source_rows": int(len(frame)),
        "source_events": int(frame[TARGET].sum()),
        "source_event_rate": float(frame[TARGET].mean()),
        "development_centers": list(split.development_centers),
        "external_centers": list(split.external_centers),
        "development_rows": int(len(split.development)),
        "development_events": int(split.development[TARGET].sum()),
        "external_rows": int(len(split.external)),
        "external_events": int(split.external[TARGET].sum()),
        "center_summary": center_table,
        "features": FEATURES,
        "excluded_postoperative_or_pathology_variables": sorted(
            name
            for name in frame.columns
            if name in FORBIDDEN_EXACT or name.startswith(FORBIDDEN_PREFIXES)
        ),
    }


def screening_flow_audit(prescreen_path: str | Path, analysis_frame: pd.DataFrame) -> dict:
    """Audit aggregate missing-data exclusions without using prescreen data for modeling."""
    try:
        prescreen = pd.read_csv(prescreen_path, low_memory=False)
    except UnicodeDecodeError:
        prescreen = pd.read_csv(prescreen_path, low_memory=False, encoding="gbk")
    mapping = {"No": 0, "Yes": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    target = prescreen[TARGET].map(mapping)
    if target.isna().any():
        raise ValueError("Unrecognized prescreen target values.")
    prescreen = prescreen.copy()
    prescreen[TARGET] = target.astype(int)
    prescreen[CENTER] = pd.to_numeric(prescreen[CENTER], errors="raise").astype(int)
    before = prescreen.groupby(CENTER)[TARGET].agg(n="size", events="sum")
    after = analysis_frame.groupby(CENTER)[TARGET].agg(n="size", events="sum")
    flow = before.join(after, lsuffix="_prescreen", rsuffix="_analysis", how="outer").fillna(0)
    flow["excluded"] = flow["n_prescreen"] - flow["n_analysis"]
    flow["events_excluded"] = flow["events_prescreen"] - flow["events_analysis"]
    return {
        "criterion_reported_by_investigator": (
            "Exclude records missing essential variables such as age or gender, "
            "or with more than 25% missing data."
        ),
        "prescreen_rows": int(len(prescreen)),
        "prescreen_events": int(prescreen[TARGET].sum()),
        "analysis_rows": int(len(analysis_frame)),
        "analysis_events": int(analysis_frame[TARGET].sum()),
        "excluded_rows": int(len(prescreen) - len(analysis_frame)),
        "excluded_events": int(prescreen[TARGET].sum() - analysis_frame[TARGET].sum()),
        "by_center": flow.reset_index().to_dict(orient="records"),
        "row_level_reproducible_from_uploaded_files": False,
        "limitation": (
            "The uploaded notebooks do not define the exact screening-variable list, "
            "and the prescreen file has no stable patient identifier for linkage."
        ),
    }


def make_preprocessor() -> ColumnTransformer:
    continuous = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [("continuous", continuous, CONTINUOUS_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _ridge() -> LogisticRegression:
    return LogisticRegression(max_iter=5000, solver="lbfgs", penalty="l2", random_state=RANDOM_STATE)


def _forest(fast: bool = False) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=120 if fast else 400,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )


def _boosting(fast: bool = False) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=80 if fast else 220,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )


def build_estimators(fast: bool = False) -> dict[str, object]:
    """Models and tuning are locked before external outcomes are inspected."""
    ridge_pipe = Pipeline([("preprocess", make_preprocessor()), ("model", _ridge())])
    forest_pipe = Pipeline([("preprocess", make_preprocessor()), ("model", _forest(fast))])
    boosting_pipe = Pipeline([("preprocess", make_preprocessor()), ("model", _boosting(fast))])

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + 1)
    ridge = GridSearchCV(
        ridge_pipe,
        {"model__C": [0.03, 0.1, 0.3, 1.0]},
        scoring="neg_brier_score",
        cv=inner_cv,
        n_jobs=N_JOBS,
        refit=True,
    )
    forest = GridSearchCV(
        forest_pipe,
        {"model__max_depth": [4, 7], "model__min_samples_leaf": [8, 20]},
        scoring="neg_brier_score",
        cv=inner_cv,
        n_jobs=N_JOBS,
        refit=True,
    )
    boosting = GridSearchCV(
        boosting_pipe,
        {"model__max_leaf_nodes": [7, 15], "model__l2_regularization": [1.0, 5.0]},
        scoring="neg_brier_score",
        cv=inner_cv,
        n_jobs=N_JOBS,
        refit=True,
    )

    stack_base = [
        ("ridge", Pipeline([("preprocess", make_preprocessor()), ("model", _ridge())])),
        ("forest", Pipeline([("preprocess", make_preprocessor()), ("model", _forest(fast))])),
        ("boosting", Pipeline([("preprocess", make_preprocessor()), ("model", _boosting(fast))])),
    ]
    stacking = StackingClassifier(
        estimators=stack_base,
        final_estimator=LogisticRegression(C=0.1, max_iter=5000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=inner_cv,
        n_jobs=N_JOBS,
        passthrough=False,
    )
    return {"ridge_logistic": ridge, "random_forest": forest, "gradient_boosting": boosting, "stacking": stacking}


def select_threshold_for_sensitivity(y_true: np.ndarray, probabilities: np.ndarray, target: float = 0.80) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    _validate_binary_inputs(y_true, probabilities)
    if not 0 < float(target) <= 1:
        raise ValueError("target sensitivity must be in (0, 1].")
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    eligible = np.flatnonzero(tpr >= target)
    if len(eligible) == 0:
        return float(np.quantile(probabilities, 1.0 - float(np.mean(y_true))))
    # Highest specificity among thresholds reaching the development-only sensitivity target.
    best = eligible[np.argmin(fpr[eligible])]
    return float(thresholds[best])


def _validate_binary_inputs(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    if y_true.ndim != 1 or probabilities.ndim != 1 or len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must be one-dimensional arrays of equal length.")
    if len(y_true) == 0:
        raise ValueError("At least one observation is required.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")


def calibration_parameters(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    _validate_binary_inputs(y_true, probabilities)
    if np.unique(y_true).size < 2:
        return math.nan, math.nan
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    predictor = logit(clipped).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(predictor, y_true)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    _validate_binary_inputs(y_true, probabilities)
    if not 0 <= float(threshold) <= 1:
        raise ValueError("classification threshold must be in [0, 1].")
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    intercept, slope = calibration_parameters(y_true, probabilities)
    return {
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "event_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "mean_predicted_risk": float(np.mean(probabilities)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "threshold": float(threshold),
        "sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "ppv": float(precision_score(y_true, predictions, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else math.nan,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Bootstrap stratification requires a one-dimensional binary outcome.")
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("Bootstrap stratification requires both outcome classes.")
    return np.concatenate(
        [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
    )


def bootstrap_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    _validate_binary_inputs(y_true, probabilities)
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_true, rng)
        try:
            values.append(float(metric(y_true[idx], probabilities[idx])))
        except (ValueError, FloatingPointError):
            continue
    if not values:
        return math.nan, math.nan
    return tuple(np.quantile(values, [0.025, 0.975]).astype(float))


def paired_auc_difference(
    y_true: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    _validate_binary_inputs(y_true, candidate)
    _validate_binary_inputs(y_true, reference)
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    observed = roc_auc_score(y_true, candidate) - roc_auc_score(y_true, reference)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_true, rng)
        values.append(roc_auc_score(y_true[idx], candidate[idx]) - roc_auc_score(y_true[idx], reference[idx]))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"auc_difference": float(observed), "ci_lower": float(lower), "ci_upper": float(upper)}


def net_benefit(y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    _validate_binary_inputs(y_true, probabilities)
    if thresholds.ndim != 1 or len(thresholds) == 0 or not np.isfinite(thresholds).all():
        raise ValueError("thresholds must be a non-empty, finite one-dimensional array.")
    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("Decision-curve thresholds must lie strictly between 0 and 1.")
    n = len(y_true)
    rows = []
    prevalence = float(np.mean(y_true))
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & (y_true == 1)))
        fp = int(np.sum(predicted & (y_true == 0)))
        weight = threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit_model": float(tp / n - fp / n * weight),
                "net_benefit_all": float(prevalence - (1.0 - prevalence) * weight),
                "net_benefit_none": 0.0,
                "triggers_per_100": float(np.mean(predicted) * 100),
                "true_positive": tp,
                "false_positive": fp,
            }
        )
    return pd.DataFrame(rows)


def standardized_differences(development: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for column in CONTINUOUS_FEATURES:
        left = pd.to_numeric(development[column], errors="coerce")
        right = pd.to_numeric(external[column], errors="coerce")
        pooled = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
        smd = (left.mean() - right.mean()) / pooled if pooled > 0 else math.nan
        rows.append({"variable": column, "level": "continuous", "smd": float(smd)})
    for column in CATEGORICAL_FEATURES:
        levels = sorted(set(development[column].dropna().astype(str)) | set(external[column].dropna().astype(str)))
        for level in levels:
            p_left = float(np.mean(development[column].astype(str) == level))
            p_right = float(np.mean(external[column].astype(str) == level))
            pooled = math.sqrt((p_left * (1 - p_left) + p_right * (1 - p_right)) / 2)
            smd = (p_left - p_right) / pooled if pooled > 0 else math.nan
            rows.append({"variable": column, "level": level, "smd": float(smd)})
    return pd.DataFrame(rows)


def guarded_subgroups(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    min_events: int = 20,
    min_nonevents: int = 100,
) -> pd.DataFrame:
    definitions = {
        "age": np.where(pd.to_numeric(frame["Age"], errors="coerce") >= 65, ">=65", "<65"),
        "gender": frame["Gender"].astype(str).to_numpy(),
        "cancer_type": frame["Gastrocolorectal"].astype(str).to_numpy(),
    }
    y = frame[TARGET].to_numpy(dtype=int)
    rows = []
    for subgroup, labels in definitions.items():
        for level in sorted(pd.unique(labels)):
            mask = labels == level
            events = int(y[mask].sum())
            nonevents = int(mask.sum() - events)
            row = {"subgroup": subgroup, "level": str(level), "n": int(mask.sum()), "events": events}
            if events < min_events or nonevents < min_nonevents:
                row.update({"status": "not_estimable", "roc_auc": math.nan, "sensitivity": math.nan, "specificity": math.nan})
            else:
                metrics = evaluate(y[mask], probabilities[mask], threshold)
                row.update(
                    {
                        "status": "estimated",
                        "roc_auc": metrics["roc_auc"],
                        "sensitivity": metrics["sensitivity"],
                        "specificity": metrics["specificity"],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_roc_calibration(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, prob):.3f})")
        CalibrationDisplay.from_predictions(y_true, prob, n_bins=8, strategy="quantile", name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="Geographic external evaluation")
    axes[0].legend(fontsize=8)
    axes[1].set_title("External calibration")
    calibration_upper = min(
        1.0,
        max(
            0.30,
            float(np.nanmax(np.concatenate([np.asarray(prob) for prob in probabilities.values()])) * 1.05),
            float(np.mean(y_true) * 1.05),
        ),
    )
    axes[1].set_xlim(0, calibration_upper)
    axes[1].set_ylim(0, calibration_upper)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
