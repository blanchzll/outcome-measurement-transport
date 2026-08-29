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
# # Leakage-resistant internal-external validation utilities

# %%
"""Internal-external (leave-one-center-out) validation utilities.

This module is deliberately independent from ``run_analysis.py``.  It uses
fixed, named predictor sets and never uses an outer-center result for feature,
model, hyperparameter, or threshold selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis import (
    CATEGORICAL_FEATURES,
    CENTER,
    CONTINUOUS_FEATURES,
    N_JOBS,
    RANDOM_STATE,
    TARGET,
    calibration_parameters,
    harmonize_gender_values,
    net_benefit,
    paired_auc_difference,
)


MISSINGNESS_LIMIT = 0.20
EXPLORATORY_EVENT_LIMIT = 20

# The model set is deliberately small and prespecified.  ``soft_voting`` is
# derived fold-by-fold from these three base models and is not tuned.
BASE_MODELS = ("ridge", "restricted_rf", "gradient_boosting")
ALL_MODELS = BASE_MODELS + ("soft_voting",)
SUMMARY_METRICS = (
    "roc_auc",
    "average_precision",
    "brier",
    "log_loss",
    "oe_ratio",
    "calibration_in_the_large",
    "calibration_intercept_joint",
    "calibration_slope",
)


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    continuous: tuple[str, ...]
    categorical: tuple[str, ...]
    role: str
    run_by_default: bool = True

    @property
    def features(self) -> tuple[str, ...]:
        return self.continuous + self.categorical


P = FeatureSetSpec(
    name="P",
    continuous=("Age", "LogPreopCr", "PreopHb", "PreopAlb"),
    categorical=(
        "Gender",
        "Diabetes",
        "Hypertension",
        "CardiovascularDisease",
        "Gastrocolorectal",
        "NeoadjuvantChemo",
    ),
    role="preoperative_primary",
)

PI = FeatureSetSpec(
    name="PI",
    continuous=P.continuous + ("IntraopTransfusion",),
    categorical=P.categorical + ("SurgicalApproach", "CombinedOrganResection"),
    role="perioperative_incremental",
)

H = FeatureSetSpec(
    name="H",
    continuous=("Age", "PreopHb", "PreopAlb", "PreopCr", "IntraopTransfusion"),
    categorical=("Gender", "Diabetes", "Gastrocolorectal", "SurgicalApproach"),
    role="harmonized_sensitivity",
)

F14 = FeatureSetSpec(
    name="F14",
    continuous=tuple(CONTINUOUS_FEATURES),
    categorical=tuple(CATEGORICAL_FEATURES),
    role="legacy_14_feature_audit",
    run_by_default=False,
)

FEATURE_SET_SPECS = {spec.name: spec for spec in (P, PI, H, F14)}


def _canonical_categorical_code(
    series: pd.Series,
    allowed_codes: tuple[str, ...],
    aliases: dict[str, str] | None = None,
) -> pd.Series:
    """Canonicalise documented codes/labels and reject all other values.

    The cleaned cohort uses human-readable labels for some variables whereas
    the source dictionary uses numeric codes. Both representations are mapped
    to the same documented numeric code. This is deterministic source-code
    validation, not statistical imputation. Values outside the supplied
    dictionary become missing and are handled inside each training fold instead
    of becoming singleton dummy variables.
    """
    text = series.astype("string").str.strip().str.casefold()
    numeric = pd.to_numeric(text, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    for code in allowed_codes:
        result.loc[numeric.eq(float(code))] = code
    for label, code in (aliases or {}).items():
        if code not in allowed_codes:
            raise ValueError(f"Alias {label!r} maps to undocumented code {code!r}.")
        result.loc[text.eq(label.strip().casefold())] = code
    return result


def engineer_loco_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only prespecified row-wise features; no cohort statistics are used."""
    required = {
        "PreopCr",
        "HeartDisease",
        "CerebrovascularDisease",
        CENTER,
    }
    required.update(feature for spec in FEATURE_SET_SPECS.values() for feature in spec.features)
    required.discard("LogPreopCr")
    required.discard("CardiovascularDisease")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing LOCO variables: {missing}")

    result = frame.copy()
    gender = harmonize_gender_values(result["Gender"])
    result["Gender"] = gender.astype(object).where(gender.notna(), np.nan)
    # Source exports use slash/underscore and related text tokens for missing
    # values. This deterministic conversion is not imputation and uses no
    # cohort statistics; all estimated preprocessing remains inside each fold.
    missing_tokens = {"", "_", "/", "na", "n/a", "nan", "null", "none"}
    continuous_fields = {
        variable
        for spec in FEATURE_SET_SPECS.values()
        for variable in spec.continuous
        if variable != "LogPreopCr"
    }
    categorical_fields = {
        variable for spec in FEATURE_SET_SPECS.values() for variable in spec.categorical
    }
    for variable in continuous_fields:
        result[variable] = pd.to_numeric(result[variable], errors="coerce")
    for variable in categorical_fields - {"Gender", "CardiovascularDisease"}:
        text = result[variable].astype("string").str.strip()
        text = text.mask(text.str.lower().isin(missing_tokens))
        result[variable] = text.astype(object).where(text.notna(), np.nan)

    documented_categories = {
        "Diabetes": (("0", "1"), {}),
        "Hypertension": (("0", "1"), {}),
        "Gastrocolorectal": (("1", "2"), {}),
        "NeoadjuvantChemo": (
            ("0", "1"),
            {"no": "0", "yes": "1", "否": "0", "是": "1"},
        ),
        "SurgicalApproach": (
            ("1", "2", "3", "4"),
            {
                "open": "1",
                "laparoscopic": "2",
                "converted": "3",
                "robotic": "4",
                "开腹": "1",
                "腹腔镜": "2",
                "中转开腹": "3",
                "机器人": "4",
            },
        ),
        "CombinedOrganResection": (
            ("0", "1"),
            {"no": "0", "yes": "1", "否": "0", "是": "1"},
        ),
    }
    for variable, (allowed, aliases) in documented_categories.items():
        if variable in result:
            canonical = _canonical_categorical_code(result[variable], allowed, aliases)
            result[variable] = canonical.astype(object).where(canonical.notna(), np.nan)
    creatinine = pd.to_numeric(result["PreopCr"], errors="coerce")
    result["LogPreopCr"] = np.log(creatinine.where(creatinine > 0))

    heart = pd.to_numeric(result["HeartDisease"], errors="coerce")
    cerebrovascular = pd.to_numeric(result["CerebrovascularDisease"], errors="coerce")
    cardiovascular = np.full(len(result), np.nan)
    cardiovascular[(heart == 1) | (cerebrovascular == 1)] = 1.0
    cardiovascular[(heart == 0) & (cerebrovascular == 0)] = 0.0
    result["CardiovascularDisease"] = cardiovascular
    return result


def availability_by_center(
    frame: pd.DataFrame,
    specs: Iterable[FeatureSetSpec] = FEATURE_SET_SPECS.values(),
    missingness_limit: float = MISSINGNESS_LIMIT,
) -> pd.DataFrame:
    """Outcome-blind availability audit; this function never reads TARGET."""
    rows: list[dict] = []
    centers = sorted(pd.to_numeric(frame[CENTER], errors="raise").astype(int).unique())
    for spec in specs:
        for center in centers:
            center_frame = frame.loc[frame[CENTER] == center]
            for variable in spec.features:
                missing_n = int(center_frame[variable].isna().sum())
                missing_rate = float(missing_n / len(center_frame))
                rows.append(
                    {
                        "feature_set": spec.name,
                        "role": spec.role,
                        "run_by_default": spec.run_by_default,
                        "center": int(center),
                        "variable": variable,
                        "n": int(len(center_frame)),
                        "missing_n": missing_n,
                        "missing_rate": missing_rate,
                        "eligible_at_20pct": bool(missing_rate <= missingness_limit),
                    }
                )
    return pd.DataFrame(rows)


def summarize_feature_set_eligibility(availability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set, group in availability.groupby("feature_set", sort=False):
        failures = group.loc[~group["eligible_at_20pct"]]
        rows.append(
            {
                "feature_set": feature_set,
                "eligible_all_centers": bool(failures.empty),
                "max_center_variable_missing_rate": float(group["missing_rate"].max()),
                "n_failed_center_variables": int(len(failures)),
                "failure_detail": "; ".join(
                    f"center={int(row.center)},variable={row.variable},missing={row.missing_rate:.3f}"
                    for row in failures.itertuples()
                ),
                "eligibility_basis": "missingness_only_no_outcome_or_performance_used",
            }
        )
    return pd.DataFrame(rows)


def make_loco_preprocessor(spec: FeatureSetSpec) -> ColumnTransformer:
    """Fold-fitted imputation without missingness indicators or center."""
    continuous = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", continuous, list(spec.continuous)),
            ("categorical", categorical, list(spec.categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_loco_search(
    spec: FeatureSetSpec,
    model_name: str,
    n_inner_centers: int,
    fast: bool = False,
) -> GridSearchCV:
    if n_inner_centers < 2:
        raise ValueError("At least two training centers are required for grouped inner validation.")
    preprocessor = make_loco_preprocessor(spec)
    if model_name == "ridge":
        estimator = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
        grid = {"model__C": [0.1] if fast else [0.03, 0.1, 0.3, 1.0]}
    elif model_name == "restricted_rf":
        estimator = RandomForestClassifier(
            n_estimators=60 if fast else 300,
            max_features="sqrt",
            class_weight=None,
            # GridSearchCV owns parallelism. Keeping the estimator itself
            # single-threaded prevents nested oversubscription on the server.
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        grid = (
            {"model__max_depth": [4], "model__min_samples_leaf": [20]}
            if fast
            else {"model__max_depth": [3, 5], "model__min_samples_leaf": [15, 30]}
        )
    elif model_name == "gradient_boosting":
        # HistGradientBoosting is used with a dense, fold-fitted one-hot
        # representation.  This is intentionally a small, low-variance grid
        # rather than an open-ended algorithm search.
        estimator = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
        grid = (
            {
                "model__max_iter": [100],
                "model__learning_rate": [0.05],
                "model__max_leaf_nodes": [7],
                "model__min_samples_leaf": [20],
            }
            if fast
            else {
                "model__max_iter": [100],
                "model__learning_rate": [0.03, 0.05],
                "model__max_leaf_nodes": [3, 7],
                "model__min_samples_leaf": [20],
            }
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return GridSearchCV(
        Pipeline([("preprocess", preprocessor), ("model", estimator)]),
        param_grid=grid,
        scoring="neg_brier_score",
        cv=GroupKFold(n_splits=n_inner_centers),
        refit=True,
        n_jobs=N_JOBS,
        return_train_score=False,
    )


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray, event_limit: int = EXPLORATORY_EVENT_LIMIT) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.ndim != 1 or len(y_true) != len(probabilities):
        raise ValueError("Outcome and probability arrays must be aligned one-dimensional arrays.")
    if len(y_true) == 0 or not np.isfinite(probabilities).all():
        raise ValueError("Non-empty finite probabilities are required.")
    events = int(y_true.sum())
    expected = float(probabilities.sum())
    intercept_joint, slope = calibration_parameters(y_true, probabilities)
    citl = calibration_in_the_large(y_true, probabilities)
    both_classes = np.unique(y_true).size == 2
    return {
        "n": int(len(y_true)),
        "events": events,
        "event_rate": float(y_true.mean()),
        "expected_events": expected,
        "oe_ratio": float(events / expected) if expected > 0 else math.nan,
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if both_classes else math.nan,
        "average_precision": float(average_precision_score(y_true, probabilities)) if events else math.nan,
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "calibration_in_the_large": citl,
        "calibration_intercept_joint": intercept_joint,
        "calibration_slope": slope,
        "calibration_inference_status": (
            "estimable" if events >= event_limit and both_classes else "exploratory_low_events"
        ),
    }


def _bootstrap_strata(y_true: np.ndarray, groups: np.ndarray | None = None) -> list[np.ndarray]:
    """Return analytic-record bootstrap strata, optionally preserving center size.

    Outcomes are deliberately not used to define strata: fixing the number of
    events would understate uncertainty for prevalence-dependent quantities
    such as O/E and calibration-in-the-large.
    """
    y_true = np.asarray(y_true, dtype=int)
    if groups is None:
        return [np.arange(len(y_true), dtype=int)] if len(y_true) else []
    labels = np.asarray(groups)
    if len(labels) != len(y_true):
        raise ValueError("groups must have the same length as y_true")
    strata = [np.flatnonzero(labels == group) for group in pd.unique(labels)]
    return [indices for indices in strata if len(indices)]


def bootstrap_metric_ci(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    """Bootstrap aggregate metric 95% CIs without retaining analytic-record draws.

    Centre summaries resample analytic records without outcome stratification;
    pooled summaries preserve center sample sizes while allowing event counts
    to vary. These are sampling intervals conditional on the locked predictions
    and do not include model-selection or refitting uncertainty.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities):
        raise ValueError("Outcome and probability arrays must have equal length")
    strata = _bootstrap_strata(y_true, groups)
    if not strata:
        return {metric: (math.nan, math.nan) for metric in SUMMARY_METRICS}
    rng = np.random.default_rng(seed)
    sampled: dict[str, list[float]] = {metric: [] for metric in SUMMARY_METRICS}
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [rng.choice(stratum, len(stratum), replace=True) for stratum in strata]
        )
        try:
            metrics = probability_metrics(y_true[indices], probabilities[indices])
        except (ValueError, FloatingPointError):
            continue
        for metric in SUMMARY_METRICS:
            value = float(metrics[metric])
            if np.isfinite(value):
                sampled[metric].append(value)
    result: dict[str, tuple[float, float]] = {}
    for metric, values in sampled.items():
        result[metric] = (
            (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
            if values
            else (math.nan, math.nan)
        )
    return result


def calibration_curve_rows(
    y_true: np.ndarray,
    predictions: dict[tuple[str, str], np.ndarray],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create aggregate (not record-level) calibration-curve data."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y_true = np.asarray(y_true, dtype=int)
    frames: list[pd.DataFrame] = []
    for (feature_set, model), probabilities in predictions.items():
        probabilities = np.asarray(probabilities, dtype=float)
        if len(probabilities) != len(y_true):
            raise ValueError("Outcome and probability arrays must have equal length")
        # Equal-frequency bins are more informative for the rare outcome than
        # fixed-width bins, while the bin-level output remains fully aggregate.
        edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
        bins = np.digitize(probabilities, edges[1:-1], right=True)
        rows = []
        for bin_id in range(len(edges) - 1):
            mask = bins == bin_id
            if not np.any(mask):
                continue
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "bin": int(bin_id + 1),
                    "n": int(mask.sum()),
                    "events": int(y_true[mask].sum()),
                    "mean_predicted": float(probabilities[mask].mean()),
                    "observed_event_rate": float(y_true[mask].mean()),
                }
            )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def calibration_in_the_large(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Estimate CITL with the prediction logit as an offset and slope fixed at 1."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.ndim != 1 or len(y_true) != len(probabilities):
        raise ValueError("Outcome and probability arrays must be aligned one-dimensional arrays.")
    if len(y_true) == 0 or not np.isfinite(probabilities).all():
        raise ValueError("Non-empty finite probabilities are required.")
    if not np.isin(y_true, [0, 1]).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("CITL requires binary outcomes and probabilities in [0, 1].")
    if np.unique(y_true).size < 2:
        return math.nan
    offset = logit(np.clip(probabilities, 1e-6, 1 - 1e-6))
    observed_events = float(y_true.sum())

    def score(intercept: float) -> float:
        return float(np.sum(expit(offset + intercept)) - observed_events)

    return float(brentq(score, -50.0, 50.0))


def _paired_auc_difference_with_groups(
    y_true: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Paired analytic-record bootstrap preserving center sizes, not event counts."""
    y_true = np.asarray(y_true, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    groups = np.asarray(groups)
    if not (len(y_true) == len(candidate) == len(reference) == len(groups)):
        raise ValueError("Outcome, predictions, and groups must have equal length.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    observed = roc_auc_score(y_true, candidate) - roc_auc_score(y_true, reference)
    stratum_indices = [np.flatnonzero(groups == group) for group in pd.unique(groups)]
    stratum_indices = [indices for indices in stratum_indices if len(indices)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [rng.choice(stratum, len(stratum), replace=True) for stratum in stratum_indices]
        )
        values.append(
            roc_auc_score(y_true[indices], candidate[indices])
            - roc_auc_score(y_true[indices], reference[indices])
        )
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"auc_difference": float(observed), "ci_lower": float(lower), "ci_upper": float(upper)}


def paired_comparisons(
    y_true: np.ndarray,
    predictions: dict[tuple[str, str], np.ndarray],
    n_bootstrap: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> list[dict]:
    """Prespecified comparisons only; no performance-driven pair construction."""
    pairs: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
    # Feature-set increment is assessed for every prespecified model,
    # including the derived equal-weight soft vote when available.
    for model in ALL_MODELS:
        pairs.append(("PI_minus_P", ("PI", model), ("P", model)))
    feature_sets = sorted({feature_set for feature_set, _ in predictions})
    for feature_set in feature_sets:
        pairs.append(
            (
                "restricted_RF_minus_ridge",
                (feature_set, "restricted_rf"),
                (feature_set, "ridge"),
            )
        )
        pairs.append(
            (
                "gradient_boosting_minus_restricted_RF",
                (feature_set, "gradient_boosting"),
                (feature_set, "restricted_rf"),
            )
        )
        pairs.append(
            (
                "soft_voting_minus_restricted_RF",
                (feature_set, "soft_voting"),
                (feature_set, "restricted_rf"),
            )
        )

    rows = []
    for index, (comparison, candidate_key, reference_key) in enumerate(pairs):
        if candidate_key not in predictions or reference_key not in predictions:
            continue
        if groups is None:
            result = paired_auc_difference(
                y_true,
                predictions[candidate_key],
                predictions[reference_key],
                n_bootstrap=n_bootstrap,
                seed=seed + index,
            )
            bootstrap_scheme = "paired_analytic_record_bootstrap"
        else:
            result = _paired_auc_difference_with_groups(
                y_true,
                predictions[candidate_key],
                predictions[reference_key],
                groups=groups,
                n_bootstrap=n_bootstrap,
                seed=seed + index,
            )
            bootstrap_scheme = "paired_analytic_record_bootstrap_within_center"
        result.update(
            {
                "comparison": comparison,
                "candidate_feature_set": candidate_key[0],
                "candidate_model": candidate_key[1],
                "reference_feature_set": reference_key[0],
                "reference_model": reference_key[1],
                "bootstrap_scheme": bootstrap_scheme,
            }
        )
        rows.append(result)
    return rows


def dca_rows(
    y_true: np.ndarray,
    predictions: dict[tuple[str, str], np.ndarray],
    thresholds: np.ndarray,
    event_limit: int = EXPLORATORY_EVENT_LIMIT,
) -> pd.DataFrame:
    frames = []
    status = "estimable" if int(np.sum(y_true)) >= event_limit else "exploratory_low_events"
    for (feature_set, model), probabilities in predictions.items():
        frame = net_benefit(y_true, probabilities, thresholds)
        frame.insert(0, "net_benefit_inference_status", status)
        frame.insert(0, "model", model)
        frame.insert(0, "feature_set", feature_set)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def historical_geography_evaluation(
    cohort: pd.DataFrame,
    feature_set_names: Iterable[str] = ("P", "PI", "H"),
    train_centers: Iterable[int] = (3, 4, 5),
    test_centers: Iterable[int] = (1, 2),
    fast: bool = False,
) -> pd.DataFrame:
    """Secondary historical geography check: train 3/4/5, test 1/2.

    The returned table is explicitly labelled ``secondary_not_untouched``:
    centers 1/2 are also used as held-out centers in the primary LOCO run, so
    this analysis is descriptive and cannot be presented as a new untouched
    external validation.
    """
    train_centers = tuple(int(center) for center in train_centers)
    test_centers = tuple(int(center) for center in test_centers)
    train_mask = cohort[CENTER].isin(train_centers).to_numpy()
    y_train = cohort.loc[train_mask, TARGET].to_numpy(dtype=int)
    train_groups = cohort.loc[train_mask, CENTER].to_numpy(dtype=int)
    if np.unique(y_train).size != 2 or len(np.unique(train_groups)) < 2:
        raise ValueError("Historical geography training data require both outcomes and two centers")

    test_masks = {
        f"center_{center}": (cohort[CENTER].to_numpy() == center) for center in test_centers
    }
    test_masks["centers_1_2_combined"] = cohort[CENTER].isin(test_centers).to_numpy()
    rows: list[dict] = []
    for feature_set in feature_set_names:
        spec = FEATURE_SET_SPECS[feature_set]
        X_train = cohort.loc[train_mask, list(spec.features)]
        fitted: dict[str, object] = {}
        for model in BASE_MODELS:
            search = build_loco_search(
                spec,
                model,
                n_inner_centers=len(np.unique(train_groups)),
                fast=fast,
            )
            search.fit(X_train, y_train, groups=train_groups)
            fitted[model] = search
        for test_label, test_mask in test_masks.items():
            if not np.any(test_mask):
                continue
            y_test = cohort.loc[test_mask, TARGET].to_numpy(dtype=int)
            predictions = {
                (feature_set, model): fitted[model]
                .predict_proba(cohort.loc[test_mask, list(spec.features)])[:, 1]
                for model in BASE_MODELS
            }
            predictions[(feature_set, "soft_voting")] = np.mean(
                [predictions[(feature_set, model)] for model in BASE_MODELS], axis=0
            )
            for (feature_name, model), probabilities in predictions.items():
                row = probability_metrics(y_test, probabilities)
                row.update(
                    {
                        "evaluation": test_label,
                        "train_centers": ",".join(map(str, train_centers)),
                        "test_centers": ",".join(map(str, test_centers)),
                        "feature_set": feature_name,
                        "model": model,
                        "validation_status": "secondary_not_untouched",
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)
