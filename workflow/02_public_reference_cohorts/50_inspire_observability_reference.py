#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
"""Build the INSPIRE longitudinal creatinine reference and observability analyses."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260826
DAY_MINUTES = 1440.0
GAMMAS = (1 / 3, 1 / 2, 2 / 3, 1.0, 1.5, 2.0, 3.0)
THRESHOLDS = tuple(np.round(np.arange(0.02, 0.151, 0.01), 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inspire", required=True, type=Path)
    parser.add_argument("--builder", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("inspire_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load builder {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def make_preprocessor(continuous: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "continuous",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                continuous,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def crossfit_nuisance(
    x: pd.DataFrame,
    observed: np.ndarray,
    y: np.ndarray,
    continuous: list[str],
    categorical: list[str],
    model_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    propensity = np.full(len(x), np.nan)
    outcome = np.full(len(x), np.nan)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preprocessor = make_preprocessor(continuous, categorical)
    if model_kind == "logistic":
        observation_model = LogisticRegression(C=0.2, max_iter=3000, solver="lbfgs")
        outcome_model = LogisticRegression(C=0.2, max_iter=3000, solver="lbfgs")
    elif model_kind == "gradient_boosting":
        observation_model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=7,
            min_samples_leaf=50, l2_regularization=2.0, random_state=SEED,
        )
        outcome_model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=7,
            min_samples_leaf=30, l2_regularization=2.0, random_state=SEED + 1,
        )
    else:
        raise ValueError(model_kind)
    for train, test in folds.split(x, observed):
        obs_pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(observation_model))])
        obs_pipeline.fit(x.iloc[train], observed[train])
        propensity[test] = obs_pipeline.predict_proba(x.iloc[test])[:, 1]
        observed_train = train[observed[train] == 1]
        if np.unique(y[observed_train]).size < 2:
            outcome[test] = float(np.nanmean(y[observed_train]))
        else:
            out_pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(outcome_model))])
            out_pipeline.fit(x.iloc[observed_train], y[observed_train])
            outcome[test] = out_pipeline.predict_proba(x.iloc[test])[:, 1]
    return np.clip(propensity, 0.005, 0.995), np.clip(outcome, 1e-5, 1 - 1e-5)


def weighted_calibration(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))

    def objective(parameters: np.ndarray) -> float:
        fitted = np.clip(expit(parameters[0] + parameters[1] * lp), 1e-10, 1 - 1e-10)
        return -float(np.sum(weight * (y * np.log(fitted) + (1 - y) * np.log(1 - fitted))))

    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        return np.nan, np.nan
    return float(result.x[0]), float(result.x[1])


def weighted_metrics(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    weight = np.asarray(weight, dtype=float)
    weight = weight / weight.sum()
    event_rate = float(np.sum(weight * y))
    expected_rate = float(np.sum(weight * p))
    intercept, slope = weighted_calibration(y, p, weight)
    return {
        "n": int(len(y)),
        "events": float(np.sum(y)),
        "event_rate": event_rate,
        "expected_rate": expected_rate,
        "oe_ratio": event_rate / expected_rate,
        "brier": float(np.sum(weight * (y - p) ** 2)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=weight)) if np.unique(y).size == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p, sample_weight=weight)) if np.unique(y).size == 2 else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def aipw_metrics(
    observed: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    propensity: np.ndarray,
    outcome_probability: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    residual = np.zeros(len(y), dtype=float)
    residual[observed == 1] = (y[observed == 1] - outcome_probability[observed == 1]) / propensity[observed == 1]
    pseudo_y = outcome_probability + observed * residual
    event_rate = float(np.mean(pseudo_y))
    expected_rate = float(np.mean(p))
    expected_brier = outcome_probability * (1 - p) ** 2 + (1 - outcome_probability) * p**2
    observed_brier = np.zeros(len(y), dtype=float)
    observed_brier[observed == 1] = (y[observed == 1] - p[observed == 1]) ** 2
    aipw_brier = expected_brier + observed / propensity * (observed_brier - expected_brier)
    rows = []
    for threshold in THRESHOLDS:
        treat = p >= threshold
        cost = threshold / (1 - threshold)
        modeled = treat * (outcome_probability - cost * (1 - outcome_probability))
        observed_value = np.zeros(len(y), dtype=float)
        observed_value[observed == 1] = treat[observed == 1] * (
            y[observed == 1] - cost * (1 - y[observed == 1])
        )
        estimate = modeled + observed / propensity * (observed_value - modeled)
        rows.append({"threshold": threshold, "aipw_net_benefit": float(np.mean(estimate))})
    return {
        "event_rate": event_rate,
        "expected_rate": expected_rate,
        "oe_ratio": event_rate / expected_rate,
        "brier": float(np.mean(aipw_brier)),
    }, pd.DataFrame(rows)


def smd_rows(frame: pd.DataFrame, observed: np.ndarray, continuous: list[str], categorical: list[str], weight: np.ndarray | None = None) -> list[dict]:
    """Compare the observed sample with the full candidate population before/after IPW."""
    rows: list[dict] = []
    obs = observed == 1
    obs_weight = np.ones(int(obs.sum())) if weight is None else np.asarray(weight, dtype=float)
    for variable in continuous:
        values = pd.to_numeric(frame[variable], errors="coerce").to_numpy(dtype=float)
        target_mean, target_sd = np.nanmean(values), np.nanstd(values, ddof=1)
        x = values[obs]; valid = np.isfinite(x)
        before = np.nanmean(x)
        after = np.average(x[valid], weights=obs_weight[valid])
        rows.append({"variable": variable, "level": "", "type": "continuous",
                     "full_candidate_value": target_mean, "observed_unweighted_value": before,
                     "observed_ipw_value": after,
                     "smd_before_vs_full": (before-target_mean)/target_sd if target_sd>0 else np.nan,
                     "smd_after_vs_full": (after-target_mean)/target_sd if target_sd>0 else np.nan})
    for variable in categorical:
        values = frame[variable].astype("string")
        for level in sorted(values.dropna().unique()):
            indicator = values.eq(level).to_numpy(dtype=float)
            target = indicator.mean(); before = indicator[obs].mean()
            after = np.average(indicator[obs], weights=obs_weight)
            denominator = np.sqrt(target*(1-target)) if 0 < target < 1 else np.nan
            rows.append({"variable": variable, "level": str(level), "type": "categorical",
                         "full_candidate_value": target, "observed_unweighted_value": before,
                         "observed_ipw_value": after,
                         "smd_before_vs_full": (before-target)/denominator if denominator>0 else np.nan,
                         "smd_after_vs_full": (after-target)/denominator if denominator>0 else np.nan})
    return rows


def reference_from_serial(baseline: pd.Series, serial: pd.DataFrame) -> pd.DataFrame:
    records = []
    for reference_id, group in serial.groupby("reference_id", sort=False):
        base = float(baseline.loc[reference_id])
        hours = group["hours_after_surgery"].to_numpy(dtype=float)
        values = group["creatinine_mg_dl"].to_numpy(dtype=float)
        maximum_ratio = float(np.max(values / base))
        max48 = float(np.max(values[hours <= 48])) if np.any(hours <= 48) else np.nan
        # Match the frozen longitudinal builder's threshold expression exactly;
        # subtraction at the 0.30 boundary can otherwise lose events to binary
        # floating-point representation (for example 0.98 - 0.68).
        event = bool((np.isfinite(max48) and max48 >= base + 0.3) or maximum_ratio >= 1.5)
        if maximum_ratio >= 3:
            stage = 3
        elif maximum_ratio >= 2:
            stage = 2
        elif event:
            stage = 1
        else:
            stage = 0
        records.append(
            {
                "reference_id": int(reference_id),
                "full168_creatinine_aki": int(event),
                "full168_creatinine_stage": stage,
                "maximum_ratio_168h": maximum_ratio,
                "n_creatinine_0_168h": int(len(values)),
                "n_creatinine_0_48h": int(np.sum((hours > 0) & (hours <= 48))),
                "n_creatinine_48_96h": int(np.sum((hours > 48) & (hours <= 96))),
                "n_creatinine_96_168h": int(np.sum((hours > 96) & (hours <= 168))),
                "first_creatinine_hour": float(np.min(hours)),
                "last_creatinine_hour": float(np.max(hours)),
                "measurement_span_hours": float(np.max(hours) - np.min(hours)),
            }
        )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> None:
    root = args.root
    for folder in ["secure_work", "outputs", "tables"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    builder = load_module(args.builder)
    operations = builder.read_operations(args.inspire)
    diagnoses = builder.read_diagnoses(args.inspire)
    cohort, _ = builder.select_index_operations(operations, diagnoses)
    labs = builder.collect_labs(args.inspire, cohort, args.chunksize)
    cohort = builder.derive_laboratory_features(cohort, labs).reset_index(drop=True)
    cohort["reference_id"] = np.arange(len(cohort), dtype=int)
    key = cohort[["subject_id", "opend_time", "reference_id"]]
    serial = key.merge(
        labs.loc[labs["item_name"].eq("creatinine"), ["subject_id", "chart_time", "value"]],
        on="subject_id", how="left",
    )
    serial["hours_after_surgery"] = (serial["chart_time"] - serial["opend_time"]) / DAY_MINUTES * 24
    serial["creatinine_mg_dl"] = pd.to_numeric(serial["value"], errors="coerce")
    serial = serial.loc[
        serial["hours_after_surgery"].gt(0)
        & serial["hours_after_surgery"].le(168)
        & serial["creatinine_mg_dl"].gt(0),
        ["reference_id", "hours_after_surgery", "creatinine_mg_dl"],
    ].sort_values(["reference_id", "hours_after_surgery"]).reset_index(drop=True)
    baseline = pd.to_numeric(cohort["PreopCr"], errors="coerce").where(lambda value: value.gt(0))
    baseline.index = cohort["reference_id"]
    longitudinal = reference_from_serial(baseline.dropna(), serial.loc[serial["reference_id"].isin(baseline.dropna().index)])
    longitudinal["dense_reference"] = (
        longitudinal["n_creatinine_0_168h"].ge(3)
        & longitudinal["n_creatinine_0_48h"].ge(1)
        & longitudinal["n_creatinine_48_96h"].ge(1)
        & longitudinal["measurement_span_hours"].ge(72)
    )
    longitudinal.to_csv(root / "secure_work" / "INSPIRE_LONGITUDINAL_REFERENCE_SECURE.csv.gz", index=False, compression="gzip")
    serial.to_csv(root / "secure_work" / "INSPIRE_CREATININE_SERIAL_SECURE.csv.gz", index=False, compression="gzip")

    features = pd.read_csv(args.prior_root / "secure_work" / "INSPIRE_CANDIDATE_FEATURES_SECURE.csv", low_memory=False)
    mappings = pd.read_csv(args.prior_root / "secure_work" / "INSPIRE_MAPPING_LABELS_SECURE.csv", low_memory=False)
    predictions = pd.read_csv(args.prior_root / "secure_work" / "INSPIRE_HARMONIZED_PREDICTIONS_SECURE.csv", low_memory=False)
    analysis = features.merge(mappings, on="reference_id", validate="one_to_one").merge(predictions, on="reference_id", validate="one_to_one").merge(longitudinal, on="reference_id", how="left", validate="one_to_one")
    if len(analysis) != 7135:
        raise RuntimeError(f"Expected 7135 candidate operations, found {len(analysis)}")
    baseline_check = pd.DataFrame(
        {
            "feature_mg_dl": pd.to_numeric(features["PreopCr"], errors="coerce") / 88.4,
            "rebuilt_mg_dl": baseline.reindex(features["reference_id"]).to_numpy(),
        }
    ).dropna()
    if baseline_check.empty or not np.allclose(
        baseline_check["feature_mg_dl"], baseline_check["rebuilt_mg_dl"], rtol=1e-8, atol=1e-8
    ):
        raise RuntimeError("Rebuilt INSPIRE reference_id order does not match frozen candidate features")
    analysis["full168_observed"] = analysis["full168_creatinine_aki"].notna().astype(int)
    analysis["two_slot_observed"] = analysis["latest_in_slot_aki"].notna().astype(int)
    analysis.to_csv(root / "secure_work" / "INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz", index=False, compression="gzip")

    flow = pd.DataFrame(
        [
            {"step": "candidate_operations", "n": len(analysis), "events": np.nan},
            {"step": "longitudinal_0_168h_creatinine_observed", "n": int(analysis["full168_observed"].sum()), "events": int(analysis["full168_creatinine_aki"].sum())},
            {"step": "two_slot_observed", "n": int(analysis["two_slot_observed"].sum()), "events": int(analysis["latest_in_slot_aki"].sum())},
            {"step": "dense_longitudinal_reference", "n": int(analysis["dense_reference"].eq(True).sum()), "events": int(analysis.loc[analysis["dense_reference"].eq(True), "full168_creatinine_aki"].sum())},
        ]
    )
    flow.to_csv(root / "tables" / "Table_inspire_longitudinal_reference_flow.csv", index=False)

    overlap = analysis["full168_creatinine_aki"].notna() & analysis["latest_in_slot_aki"].notna()
    cross = pd.crosstab(
        index=analysis.loc[overlap, "full168_creatinine_aki"].astype(int),
        columns=analysis.loc[overlap, "latest_in_slot_aki"].astype(int),
        rownames=["full168_creatinine_aki"], colnames=["latest_in_slot_aki"],
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    cross.stack(future_stack=True).rename("n").reset_index().to_csv(root / "tables" / "Table_two_slot_vs_longitudinal_concordance.csv", index=False)
    tn, fp, fn, tp = int(cross.loc[0, 0]), int(cross.loc[0, 1]), int(cross.loc[1, 0]), int(cross.loc[1, 1])
    diagnostic = pd.DataFrame(
        [{
            "overlap_n": int(overlap.sum()), "longitudinal_events": tp + fn,
            "two_slot_events": tp + fp, "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp), "positive_predictive_value": tp / (tp + fp),
            "negative_predictive_value": tn / (tn + fn),
            "reference_status": "operational_longitudinal_creatinine_not_clinician_adjudicated",
        }]
    )
    diagnostic.to_csv(root / "tables" / "Table_two_slot_diagnostic_performance.csv", index=False)

    density_rows = []
    counts = pd.to_numeric(analysis["n_creatinine_0_168h"], errors="coerce").fillna(0)
    y_long = pd.to_numeric(analysis["full168_creatinine_aki"], errors="coerce")
    for minimum in [1, 2, 3, 4, 5, 7, 10]:
        mask = counts.ge(minimum) & y_long.notna()
        density_rows.append({"minimum_postoperative_creatinine_count": minimum, "n": int(mask.sum()), "events": int(y_long[mask].sum()), "event_rate": float(y_long[mask].mean()) if mask.any() else np.nan})
    pd.DataFrame(density_rows).to_csv(root / "tables" / "Table_monitoring_density_event_gradient.csv", index=False)

    continuous = ["Age", "PreopHb", "PreopAlb", "PreopCr", "n_preop_creatinine_7d", "ridge_probability", "restricted_rf_probability", "gradient_boosting_probability"]
    categorical = ["Gender", "Diabetes", "Gastrocolorectal", "SurgicalApproach", "AnyIntraopTransfusion"]
    x = analysis[continuous + categorical].copy()
    all_adjustment_rows = []
    all_weight_rows = []
    all_smd_rows = []
    all_net_benefit = []
    for target_name, observed_column, outcome_column in [
        ("two_slot", "two_slot_observed", "latest_in_slot_aki"),
        ("longitudinal_168h", "full168_observed", "full168_creatinine_aki"),
    ]:
        observed = analysis[observed_column].to_numpy(dtype=int)
        y = pd.to_numeric(analysis[outcome_column], errors="coerce").fillna(0).to_numpy(dtype=int)
        p = analysis["restricted_rf_probability"].to_numpy(dtype=float)
        for nuisance_kind in ["logistic", "gradient_boosting"]:
            propensity, outcome_probability = crossfit_nuisance(x, observed, y, continuous, categorical, nuisance_kind)
            analysis[f"{target_name}_{nuisance_kind}_propensity"] = propensity
            analysis[f"{target_name}_{nuisance_kind}_outcome_probability"] = outcome_probability
            observed_mask = observed == 1
            raw_weight = 1 / propensity[observed_mask]
            complete_case = weighted_metrics(
                y[observed_mask], p[observed_mask], np.ones(int(observed_mask.sum()))
            )
            all_adjustment_rows.append(
                {
                    "target": target_name,
                    "nuisance_model": nuisance_kind,
                    "method": "complete_case",
                    "weight_truncation": "not_applicable",
                    **complete_case,
                }
            )
            for lower, upper, label in [(0, 1, "none"), (0.005, 0.995, "0.5_99.5"), (0.01, 0.99, "1_99"), (0.025, 0.975, "2.5_97.5")]:
                if label == "none":
                    weight = raw_weight.copy()
                else:
                    lo, hi = np.quantile(raw_weight, [lower, upper])
                    weight = np.clip(raw_weight, lo, hi)
                metrics = weighted_metrics(y[observed_mask], p[observed_mask], weight)
                all_adjustment_rows.append({"target": target_name, "nuisance_model": nuisance_kind, "method": "ipw_hajek", "weight_truncation": label, **metrics})
                all_weight_rows.append({
                    "target": target_name, "nuisance_model": nuisance_kind, "weight_truncation": label,
                    "observed_n": int(observed_mask.sum()), "observed_fraction": float(observed.mean()),
                    "weight_min": float(weight.min()), "weight_p01": float(np.quantile(weight, 0.01)),
                    "weight_median": float(np.median(weight)), "weight_p99": float(np.quantile(weight, 0.99)),
                    "weight_max": float(weight.max()), "effective_sample_size": float(weight.sum() ** 2 / np.sum(weight**2)),
                    "propensity_min": float(propensity.min()), "propensity_p01": float(np.quantile(propensity, 0.01)),
                    "propensity_p99": float(np.quantile(propensity, 0.99)), "propensity_max": float(propensity.max()),
                })
            aipw, net_benefit = aipw_metrics(observed, y, p, propensity, outcome_probability)
            all_adjustment_rows.append({"target": target_name, "nuisance_model": nuisance_kind, "method": "aipw", "weight_truncation": "none", "n": len(y), "events": np.nan, "roc_auc": np.nan, "average_precision": np.nan, "calibration_intercept": np.nan, "calibration_slope": np.nan, **aipw})
            net_benefit.insert(0, "target", target_name)
            net_benefit.insert(1, "nuisance_model", nuisance_kind)
            all_net_benefit.append(net_benefit)
        primary_propensity = analysis[f"{target_name}_logistic_propensity"].to_numpy(dtype=float)
        primary_outcome = analysis[f"{target_name}_logistic_outcome_probability"].to_numpy(dtype=float)
        primary_weight = 1 / primary_propensity[observed_mask]
        primary_weight = np.minimum(primary_weight, np.quantile(primary_weight, .99))
        all_smd_rows.extend({"target": target_name, "nuisance_model": "logistic", "weight_truncation": "upper_99",
                             **row} for row in smd_rows(analysis, observed, continuous, categorical, primary_weight))
        mnar_rows = []
        for gamma in GAMMAS:
            shifted = expit(logit(primary_outcome) + np.log(gamma))
            expected_y = np.where(observed == 1, y, shifted)
            event_rate = float(expected_y.mean())
            expected_rate = float(p.mean())
            brier = float(np.mean(expected_y * (1 - p) ** 2 + (1 - expected_y) * p**2))
            for threshold in THRESHOLDS:
                treat = p >= threshold
                cost = threshold / (1 - threshold)
                net_benefit = float(np.mean(treat * (expected_y - cost * (1 - expected_y))))
                mnar_rows.append({"target": target_name, "gamma_unobserved_vs_observed_outcome_odds": gamma, "event_rate": event_rate, "oe_ratio": event_rate / expected_rate, "brier": brier, "threshold": threshold, "net_benefit": net_benefit})
        pd.DataFrame(mnar_rows).to_csv(root / "tables" / f"Table_{target_name}_MNAR_sensitivity.csv", index=False)

    pd.DataFrame(all_adjustment_rows).to_csv(root / "tables" / "Table_observability_adjusted_performance.csv", index=False)
    pd.DataFrame(all_weight_rows).to_csv(root / "tables" / "Table_observability_weight_diagnostics.csv", index=False)
    pd.DataFrame(all_smd_rows).to_csv(root / "tables" / "Table_observability_predictor_imbalance.csv", index=False)
    pd.concat(all_net_benefit, ignore_index=True).to_csv(root / "tables" / "Table_observability_AIPW_net_benefit.csv", index=False)
    analysis.to_csv(root / "secure_work" / "INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz", index=False, compression="gzip")

    audit = {
        "candidate_operations": int(len(analysis)),
        "longitudinal_reference_n": int(analysis["full168_observed"].sum()),
        "longitudinal_reference_events": int(analysis["full168_creatinine_aki"].sum()),
        "two_slot_observed_n": int(analysis["two_slot_observed"].sum()),
        "two_slot_events": int(analysis["latest_in_slot_aki"].sum()),
        "dense_reference_n": int(analysis["dense_reference"].eq(True).sum()),
        "dense_reference_events": int(analysis.loc[analysis["dense_reference"].eq(True), "full168_creatinine_aki"].sum()),
        "two_slot_sensitivity_vs_longitudinal": float(diagnostic.iloc[0]["sensitivity"]),
        "two_slot_specificity_vs_longitudinal": float(diagnostic.iloc[0]["specificity"]),
        "reference_is_clinician_adjudicated": False,
        "reference_is_full_kdigo": False,
        "urine_output_included": False,
        "rrt_included": False,
        "nuisance_models_cross_fitted": True,
        "mnar_identified_without_assumptions": False,
        "raw_inspire_path": str(args.inspire),
        "prior_predictions_sha256": sha256(args.prior_root / "secure_work" / "INSPIRE_HARMONIZED_PREDICTIONS_SECURE.csv"),
    }
    (root / "outputs" / "INSPIRE_OBSERVABILITY_REFERENCE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    run(parse_args())
