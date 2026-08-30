# %% [markdown]
# # VitalDB measurement-transport stress test
#
# A fixed ridge risk engine is trained in one patient-disjoint subset and
# evaluated in a held-out subset. Complete 0-168 h creatinine trajectories are
# then subjected to prespecified deletion mechanisms. The aim is mechanism
# replication, not selection of a best clinical prediction model.

# %%
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260830
MECHANISMS = ["MCAR", "stratum_MAR", "risk_MAR", "history_MAR", "outcome_MNAR", "mixed_MNAR"]
RETENTIONS = [0.35, 0.55, 0.75]
STRENGTHS = ["weak", "strong"]
REFERENCE_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
RECALIBRATION_METHODS = {
    "recalibration_intercept_apparent",
    "recalibration_intercept_truth",
    "recalibration_intercept_slope_apparent",
    "recalibration_intercept_slope_truth",
    "reference_05pct_recalibration",
    "reference_10pct_recalibration",
    "reference_20pct_recalibration",
    "reference_30pct_recalibration",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ascertainment_stress", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def z(values: pd.Series) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    mean = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros(len(x))
    return np.nan_to_num((x - mean) / sd, nan=0.0)


def make_risk_engine(numeric: list[str], categorical: list[str]) -> Pipeline:
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
            ("model", LogisticRegression(C=0.25, solver="liblinear", max_iter=2000, random_state=SEED)),
        ]
    )


def add_event_rate_inference(metrics: dict[str, float], truth_event_rate: float) -> dict[str, float]:
    result = dict(metrics)
    standard_error = result.get("event_rate_se", np.nan)
    estimate = result.get("event_rate", np.nan)
    if np.isfinite(standard_error) and np.isfinite(estimate):
        result["event_rate_ci_lower"] = max(0.0, estimate - 1.96 * standard_error)
        result["event_rate_ci_upper"] = min(1.0, estimate + 1.96 * standard_error)
        result["event_rate_coverage"] = int(
            result["event_rate_ci_lower"] <= truth_event_rate <= result["event_rate_ci_upper"]
        )
    else:
        result["event_rate_ci_lower"] = np.nan
        result["event_rate_ci_upper"] = np.nan
        result["event_rate_coverage"] = np.nan
    return result


def crossfit_recalibration(frame: pd.DataFrame, rng, stress, intercept_only: bool) -> tuple[np.ndarray, bool]:
    fold = rng.integers(0, 2, size=len(frame))
    prediction = np.full(len(frame), np.nan, dtype=float)
    successful = 0
    for held_out in (0, 1):
        train = (
            (fold != held_out)
            & frame.R.eq(1).to_numpy()
            & frame.y_reconstructed.notna().to_numpy()
        )
        test = fold == held_out
        if train.sum() < 20 or np.unique(frame.loc[train, "y_reconstructed"]).size < 2:
            continue
        try:
            _, intercept, slope = stress.recalibrate(
                frame.loc[train, "risk"],
                frame.loc[train, "y_reconstructed"],
                intercept_only=intercept_only,
            )
        except Exception:
            continue
        prediction[test] = expit(
            intercept + slope * logit(frame.loc[test, "risk"].clip(1e-6, 1 - 1e-6))
        )
        successful += 1
    return prediction, successful == 2


def prepare_analysis(case_path: Path, serial_path: Path, stress) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cases = pd.read_csv(case_path, low_memory=False)
    serial = pd.read_csv(serial_path, low_memory=False)

    # Primary independence rule: exclude repeat-operation patients because
    # absolute chronology is unavailable and random case IDs cannot identify a
    # temporal first operation.
    cases_per_patient = cases.groupby("subjectid")["caseid"].nunique()
    single_operation_patients = cases_per_patient.index[cases_per_patient.eq(1)]
    eligible = cases.loc[
        cases["adult"] & cases["dense_reference"] & cases["subjectid"].isin(single_operation_patients)
    ].copy()
    numeric_candidates = [
        "age", "height", "weight", "bmi", "asa", "preop_hb", "preop_plt", "preop_na", "preop_k",
        "preop_gluc", "preop_alb", "preop_bun", "preop_cr", "baseline_cr", "intraop_ebl", "intraop_uo",
        "intraop_rbc", "intraop_ffp", "intraop_crystalloid", "intraop_colloid", "intraop_eph",
        "intraop_phe", "intraop_epi",
    ]
    categorical_candidates = [
        "sex", "emop", "department", "optype", "approach", "ane_type", "preop_htn", "preop_dm"
    ]
    numeric = [name for name in numeric_candidates if name in eligible.columns and eligible[name].notna().any()]
    categorical = [name for name in categorical_candidates if name in eligible.columns and eligible[name].notna().any()]
    features = numeric + categorical
    y = eligible["creatinine_event_168h"].astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, test_idx = next(splitter.split(eligible, y, groups=eligible["subjectid"]))
    train = eligible.iloc[train_idx].copy()
    test = eligible.iloc[test_idx].copy()
    if train["creatinine_event_168h"].nunique() < 2 or test["creatinine_event_168h"].nunique() < 2:
        raise RuntimeError("Patient-disjoint split does not contain both outcome classes")

    model = make_risk_engine(numeric, categorical)
    model.fit(train[features], train["creatinine_event_168h"].astype(int))
    test["risk"] = model.predict_proba(test[features])[:, 1]
    patient = pd.DataFrame(
        {
            "reference_id": test["caseid"].astype(int),
            "baseline_creatinine": test["baseline_cr"].astype(float),
            "y_full": test["creatinine_event_168h"].astype(int),
            "risk": test["risk"].clip(1e-6, 1 - 1e-6),
            "age_z": z(test["age"]),
            "sex_z": z(test["sex"].astype(str).str.upper().map({"M": 1, "F": 0})),
            "stratum_z": z(pd.Series(pd.factorize(test["optype"].astype(str))[0], index=test.index)),
        }
    ).reset_index(drop=True)
    serial = serial.rename(columns={"caseid": "reference_id", "hours_from_opend": "hour", "result": "creatinine"})
    serial = serial.loc[serial["reference_id"].isin(patient["reference_id"]), ["reference_id", "hour", "creatinine"]]
    serial = serial.sort_values(["reference_id", "hour"], kind="stable").reset_index(drop=True)
    audit = {
        "eligible_dense_adult_unique_patients": int(len(eligible)),
        "train_n": int(len(train)),
        "train_events": int(train["creatinine_event_168h"].sum()),
        "test_n": int(len(test)),
        "test_events": int(test["creatinine_event_168h"].sum()),
        "test_gi_n": int(test["gi_stomach_colorectal"].sum()),
        "numeric_predictors": numeric,
        "categorical_predictors": categorical,
        "full_reference_metrics": stress.weighted_metrics(patient["y_full"], patient["risk"]),
        "split": "70/30 patient-disjoint random group split; fixed seed; no tuning against test performance",
        "primary_operation_rule": "patients with exactly one VitalDB operation; repeat-operation patients excluded",
        "model_role": "database-native ridge risk engine for outcome-measurement mechanism replication",
    }
    return patient, serial, audit


def run_simulation(patient: pd.DataFrame, serial: pd.DataFrame, stress, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = stress.weighted_metrics(patient.y_full, patient.risk)
    rows: list[dict[str, object]] = []
    for retention in RETENTIONS:
        for mechanism in MECHANISMS:
            for strength in STRENGTHS:
                for replicate in range(reps):
                    condition = f"VitalDB|{retention}|{mechanism}|{strength}|{replicate}"
                    seed = SEED + int(hashlib.sha256(condition.encode()).hexdigest()[:8], 16) % 2_000_000_000
                    rng = np.random.default_rng(seed)
                    sim = stress.delete_and_reconstruct(patient, serial, mechanism, retention, strength, rng)
                    frame = sim.patient
                    observed = frame.R.eq(1) & frame.y_reconstructed.notna()
                    common = {
                        "database": "VitalDB",
                        "retention_target": retention,
                        "mechanism": mechanism,
                        "strength": strength,
                        "replicate": replicate,
                        "seed": seed,
                        "measurement_retention_realized": sim.mean_measurement_retention,
                        "outcome_observed_fraction": float(observed.mean()),
                        "reconstructed_sensitivity": float(
                            (frame.y_reconstructed.eq(1) & frame.y_full.eq(1)).sum() / max(frame.y_full.sum(), 1)
                        ),
                    }
                    rows.append(
                        {
                            **common,
                            "method": "full_reference",
                            "evaluation_target": "full",
                            **add_event_rate_inference(truth, truth["event_rate"]),
                        }
                    )
                    if observed.sum() < 20 or frame.loc[observed, "y_reconstructed"].nunique() < 2:
                        continue
                    y_obs = frame.loc[observed, "y_reconstructed"]
                    p_obs = frame.loc[observed, "risk"]
                    rows.append(
                        {
                            **common,
                            "method": "naive",
                            "evaluation_target": "reconstructed",
                            **add_event_rate_inference(stress.weighted_metrics(y_obs, p_obs), truth["event_rate"]),
                        }
                    )
                    raw_weight = 1 / frame.loc[observed, "q_observed"].clip(0.005, 1)
                    raw_metrics = stress.weighted_metrics(y_obs, p_obs, raw_weight)
                    raw_metrics.update({"weight_p99": float(raw_weight.quantile(0.99)), "weight_max": float(raw_weight.max())})
                    rows.append(
                        {
                            **common,
                            "method": "IPAW_design_probability_untruncated",
                            "evaluation_target": "reconstructed",
                            **add_event_rate_inference(raw_metrics, truth["event_rate"]),
                        }
                    )
                    truncated_weight = raw_weight.clip(upper=raw_weight.quantile(0.99))
                    trunc_metrics = stress.weighted_metrics(y_obs, p_obs, truncated_weight)
                    trunc_metrics.update(
                        {"weight_p99": float(truncated_weight.quantile(0.99)), "weight_max": float(truncated_weight.max())}
                    )
                    rows.append(
                        {
                            **common,
                            "method": "IPAW_design_probability_truncated99",
                            "evaluation_target": "reconstructed",
                            **add_event_rate_inference(trunc_metrics, truth["event_rate"]),
                        }
                    )
                    aipw, aipw_se = stress.aipw_event_rate(frame)
                    aipw_metrics = {
                        "n": len(frame),
                        "events": aipw * len(frame),
                        "event_rate": aipw,
                        "event_rate_se": aipw_se,
                        "mean_prediction": float(frame.risk.mean()),
                        "oe": aipw / float(frame.risk.mean()),
                        "ess": float((raw_weight.sum() ** 2) / np.square(raw_weight).sum()),
                    }
                    rows.append(
                        {
                            **common,
                            "method": "AIPW_design_probability",
                            "evaluation_target": "reconstructed",
                            **add_event_rate_inference(aipw_metrics, truth["event_rate"]),
                        }
                    )
                    for intercept_only, label in ((True, "intercept"), (False, "intercept_slope")):
                        p_updated, successful = crossfit_recalibration(frame, rng, stress, intercept_only)
                        if not successful:
                            continue
                        apparent = stress.weighted_metrics(frame.loc[observed, "y_reconstructed"], p_updated[observed])
                        rows.append(
                            {
                                **common,
                                "method": f"recalibration_{label}_apparent",
                                "evaluation_target": "reconstructed",
                                **apparent,
                            }
                        )
                        truth_updated = stress.weighted_metrics(frame.y_full, p_updated)
                        rows.append(
                            {
                                **common,
                                "method": f"recalibration_{label}_truth",
                                "evaluation_target": "full",
                                **truth_updated,
                            }
                        )

                    reference_order = rng.permutation(len(frame))
                    for fraction in REFERENCE_FRACTIONS:
                        sample_size = max(30, int(np.ceil(fraction * len(frame))))
                        calibration_index = reference_order[:sample_size]
                        evaluation_index = reference_order[sample_size:]
                        if (
                            np.unique(frame.y_full.iloc[calibration_index]).size < 2
                            or len(evaluation_index) == 0
                        ):
                            continue
                        try:
                            _, intercept, slope = stress.recalibrate(
                                frame.risk.iloc[calibration_index],
                                frame.y_full.iloc[calibration_index],
                                intercept_only=False,
                            )
                        except Exception:
                            continue
                        p_reference = expit(
                            intercept
                            + slope
                            * logit(frame.risk.iloc[evaluation_index].clip(1e-6, 1 - 1e-6))
                        )
                        reference_metrics = stress.weighted_metrics(
                            frame.y_full.iloc[evaluation_index], p_reference
                        )
                        reference_metrics.update(
                            {"reference_sample_n": sample_size, "evaluation_n": len(evaluation_index)}
                        )
                        rows.append(
                            {
                                **common,
                                "method": f"reference_{int(round(fraction * 100)):02d}pct_recalibration",
                                "evaluation_target": "full_heldout",
                                **reference_metrics,
                            }
                        )
                    lower, upper = stress.mnar_event_bounds(frame, gamma=2.0)
                    rows.append(
                        {
                            **common,
                            "method": "Gamma2_prediction_sensitivity_region",
                            "evaluation_target": "full",
                            "event_rate": (lower + upper) / 2,
                            "mnar_lower": lower,
                            "mnar_upper": upper,
                            "mnar_covers_truth": int(lower <= truth["event_rate"] <= upper),
                        }
                    )

    raw = pd.DataFrame(rows)
    keys = ["database", "retention_target", "mechanism", "strength", "method", "evaluation_target"]
    metrics = [
        "event_rate", "oe", "brier", "auc", "calibration_intercept", "calibration_slope",
        "outcome_observed_fraction", "reconstructed_sensitivity", "ess", "event_rate_se", "weight_p99",
        "weight_max", "reference_sample_n", "evaluation_n",
    ]
    summary_rows: list[dict[str, object]] = []
    for values, group in raw.groupby(keys, dropna=False):
        base = dict(zip(keys, values))
        for metric in metrics:
            if metric not in group:
                continue
            x = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(x):
                target = (
                    {"oe": 1.0, "calibration_intercept": 0.0, "calibration_slope": 1.0}.get(metric, np.nan)
                    if base["method"] in RECALIBRATION_METHODS
                    else truth.get(metric, np.nan)
                )
                summary_rows.append(
                    {
                        **base,
                        "metric": metric,
                        "n_replicates": len(x),
                        "mean": float(x.mean()),
                        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                        "q025": float(np.quantile(x, 0.025)),
                        "q975": float(np.quantile(x, 0.975)),
                        "truth": target,
                        "bias": float(x.mean() - target) if np.isfinite(target) else np.nan,
                        "rmse": float(np.sqrt(np.mean((x - target) ** 2))) if np.isfinite(target) else np.nan,
                    }
                )
        if "mnar_covers_truth" in group:
            coverage = pd.to_numeric(group["mnar_covers_truth"], errors="coerce").dropna()
            if len(coverage):
                summary_rows.append(
                    {**base, "metric": "MNAR_event_rate_coverage", "n_replicates": len(coverage), "mean": float(coverage.mean())}
                )
        if "event_rate_coverage" in group:
            coverage = pd.to_numeric(group["event_rate_coverage"], errors="coerce").dropna()
            if len(coverage):
                summary_rows.append(
                    {**base, "metric": "event_rate_interval_coverage", "n_replicates": len(coverage), "mean": float(coverage.mean())}
                )
    return raw, pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-level", required=True, type=Path)
    parser.add_argument("--serial", required=True, type=Path)
    parser.add_argument("--stress-module", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=300)
    args = parser.parse_args()

    stress = load_module(args.stress_module)
    patient, serial, audit = prepare_analysis(args.case_level, args.serial, stress)
    raw, summary = run_simulation(patient, serial, stress, args.reps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "VITALDB_SIMULATION_REPLICATES_INTERNAL.csv.gz", index=False, compression="gzip")
    summary.to_csv(args.output_dir / "Table_vitaldb_simulation_summary.csv", index=False)
    audit.update(
        {
            "replicates_per_condition": args.reps,
            "factorial_conditions": len(MECHANISMS) * len(RETENTIONS) * len(STRENGTHS),
            "simulation_rows": int(len(raw)),
            "serial_rows_test": int(len(serial)),
        }
    )
    (args.output_dir / "VITALDB_SIMULATION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
