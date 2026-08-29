# %% [markdown]
# # Complete-outcome -> deletion -> reconstruction -> correction simulation
# Runs the identical factorial stress test in INSPIRE and MIMIC-IV.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from ascertainment_stress import (
    aipw_event_rate, delete_and_reconstruct, mnar_event_bounds,
    recalibrate, weighted_metrics,
)

ROOT = Path(str(_release_path('analysis')))
SECURE, TABLES, OUTPUTS = ROOT / "secure_work", ROOT / "tables", ROOT / "outputs"
EICU_SECURE = ROOT / "eicu" / "secure"
MECHANISMS = ["MCAR", "stratum_MAR", "risk_MAR", "history_MAR", "outcome_MNAR", "mixed_MNAR"]
RETENTIONS = [0.35, 0.55, 0.75]
STRENGTHS = ["weak", "strong"]
BASE_SEED = 20260826
RECALIBRATION_METHODS = {
    "recalibration_intercept_apparent", "recalibration_intercept_truth",
    "recalibration_intercept_slope_apparent", "recalibration_intercept_slope_truth",
    "reference_05pct_recalibration", "reference_10pct_recalibration",
    "reference_20pct_recalibration", "reference_30pct_recalibration",
}
REFERENCE_FRACTIONS = (0.05, 0.10, 0.20, 0.30)


def estimand_truth(method, metric, truth):
    """Return the method-specific target; updating and performance estimation differ."""
    if method in RECALIBRATION_METHODS:
        return {"oe": 1.0, "calibration_intercept": 0.0, "calibration_slope": 1.0}.get(metric, np.nan)
    if method == "Gamma2_prediction_sensitivity_region":
        return truth["event_rate"] if metric == "event_rate" else np.nan
    return truth.get(metric, np.nan)


def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def prepare_inspire():
    d = pd.read_csv(SECURE / "INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz")
    d = d.loc[d.dense_reference.eq(1)].copy()
    patient = pd.DataFrame({
        "reference_id": d.reference_id,
        # Source-model PreopCr is micromol/L; serial creatinine is mg/dL.
        "baseline_creatinine": d.PreopCr / 88.4,
        "y_full": d.full168_creatinine_aki.astype(int),
        "risk": d.restricted_rf_probability.clip(1e-6, 1 - 1e-6),
        "age_z": z(d.Age), "sex_z": z(d.Gender.astype(str).str.lower().map({"male": 1, "female": 0})),
        "stratum_z": z(pd.to_numeric(d.Gastrocolorectal, errors="coerce").fillna(1.5)),
    })
    serial = pd.read_csv(SECURE / "INSPIRE_CREATININE_SERIAL_SECURE.csv.gz")
    serial = serial.rename(columns={"hours_after_surgery": "hour", "creatinine_mg_dl": "creatinine"})
    serial = serial.loc[serial.reference_id.isin(patient.reference_id), ["reference_id", "hour", "creatinine"]]
    return patient.reset_index(drop=True), serial.reset_index(drop=True)


def prepare_mimic():
    d = pd.read_csv(SECURE / "MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
    d = d.loc[d.R_dense.eq(1)].sort_values(["intime", "reference_id"]).copy()
    split = int(np.floor(0.60 * len(d)))
    train, test = d.iloc[:split].copy(), d.iloc[split:].copy()
    numeric = ["age", "baseline_creatinine", "baseline_albumin", "baseline_bun", "baseline_glucose",
               "baseline_sodium", "baseline_potassium", "baseline_hemoglobin", "baseline_wbc", "baseline_platelet", "diabetes"]
    categorical = ["gender", "active_service", "first_careunit", "admission_type"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("pre", pre), ("model", LogisticRegression(C=0.25, solver="liblinear", max_iter=1000))])
    model.fit(train[numeric + categorical], train.Y_longitudinal.astype(int))
    risk = model.predict_proba(test[numeric + categorical])[:, 1]
    test = test.assign(risk=risk)
    patient = pd.DataFrame({
        "reference_id": test.reference_id,
        "baseline_creatinine": test.baseline_creatinine,
        "y_full": test.Y_longitudinal.astype(int), "risk": test.risk,
        "age_z": z(test.age), "sex_z": z(test.gender.eq("M").astype(int)),
        "stratum_z": z(pd.factorize(test.active_service)[0]),
    })
    serial = pd.read_csv(SECURE / "MIMIC_CREATININE_SERIAL_SECURE.csv.gz")
    serial = serial.loc[serial.reference_id.isin(patient.reference_id), ["reference_id", "hour", "creatinine"]]
    model_audit = {
        "n_train": len(train), "events_train": int(train.Y_longitudinal.sum()),
        "n_temporal_test": len(test), "events_temporal_test": int(test.Y_longitudinal.sum()),
        "split": "first 60% vs last 40% ordered by ICU intime",
        "role": "database-native ridge risk engine for measurement-mechanism replication",
        "full_reference_metrics": weighted_metrics(patient.y_full, patient.risk),
    }
    (OUTPUTS / "MIMIC_TEMPORAL_MODEL_AUDIT.json").write_text(json.dumps(model_audit, indent=2), encoding="utf-8")
    return patient.reset_index(drop=True), serial.reset_index(drop=True)


def prepare_eicu():
    """Build an unseen-hospital test set and a database-native ridge risk engine."""
    d = pd.read_csv(EICU_SECURE / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
    d = d.loc[d.R_dense.eq(1)].sort_values(["hospitaldischargeyear", "hospitalid", "reference_id"]).copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=BASE_SEED)
    train_index, test_index = next(splitter.split(d, groups=d.hospitalid))
    train, test = d.iloc[train_index].copy(), d.iloc[test_index].copy()
    numeric = ["age_num", "baseline_creatinine", "admissionheight", "admissionweight", "hospitaladmitoffset"]
    categorical = ["gender", "ethnicity", "unittype", "hospitaladmitsource", "unitadmitsource",
                   "apacheadmissiondx", "numbedscategory", "teachingstatus", "region"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), categorical),
    ])
    model = Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(C=0.25, solver="liblinear", max_iter=1000)),
    ])
    model.fit(train[numeric + categorical], train.Y_longitudinal.astype(int))
    risk = model.predict_proba(test[numeric + categorical])[:, 1]
    test = test.assign(risk=risk)
    patient = pd.DataFrame({
        "reference_id": test.reference_id,
        "baseline_creatinine": test.baseline_creatinine,
        "y_full": test.Y_longitudinal.astype(int),
        "risk": test.risk,
        "age_z": z(test.age_num),
        "sex_z": z(test.gender.eq("Male").astype(int)),
        "stratum_z": z(pd.factorize(test.hospitalid)[0]),
    })
    serial = pd.read_csv(EICU_SECURE / "EICU_CREATININE_SERIAL_SECURE.csv.gz")
    serial = serial.loc[serial.reference_id.isin(patient.reference_id), ["reference_id", "hour", "creatinine"]]
    model_audit = {
        "n_train": len(train),
        "events_train": int(train.Y_longitudinal.sum()),
        "hospitals_train": int(train.hospitalid.nunique()),
        "n_unseen_hospital_test": len(test),
        "events_unseen_hospital_test": int(test.Y_longitudinal.sum()),
        "hospitals_test": int(test.hospitalid.nunique()),
        "split": "70% vs 30% deterministic group split by hospital; no hospital appears in both sets",
        "role": "database-native ridge risk engine for measurement-mechanism replication",
        "full_reference_metrics": weighted_metrics(patient.y_full, patient.risk),
    }
    (OUTPUTS / "EICU_GROUP_HELDOUT_MODEL_AUDIT.json").write_text(
        json.dumps(model_audit, indent=2) + "\n", encoding="utf-8"
    )
    return patient.reset_index(drop=True), serial.reset_index(drop=True)


def record(method, target, metrics, common):
    return {**common, "method": method, "evaluation_target": target, **metrics}


def add_event_rate_inference(metrics, truth_event_rate):
    """Attach a Wald interval and Monte Carlo coverage indicator when defined."""
    result = dict(metrics)
    se = result.get("event_rate_se", np.nan)
    estimate = result.get("event_rate", np.nan)
    if np.isfinite(se) and np.isfinite(estimate):
        result["event_rate_ci_lower"] = max(0.0, estimate - 1.96 * se)
        result["event_rate_ci_upper"] = min(1.0, estimate + 1.96 * se)
        result["event_rate_coverage"] = int(
            result["event_rate_ci_lower"] <= truth_event_rate <= result["event_rate_ci_upper"]
        )
    else:
        result["event_rate_ci_lower"] = np.nan
        result["event_rate_ci_upper"] = np.nan
        result["event_rate_coverage"] = np.nan
    return result


def crossfit_recalibration(frame, rng, intercept_only):
    """Two-fold local updating with out-of-fold prediction for every patient."""
    fold = rng.integers(0, 2, size=len(frame))
    prediction = np.full(len(frame), np.nan, dtype=float)
    successful_folds = 0
    for held_out in (0, 1):
        train = (fold != held_out) & frame.R.eq(1).to_numpy() & frame.y_reconstructed.notna().to_numpy()
        test = fold == held_out
        if train.sum() < 20 or np.unique(frame.loc[train, "y_reconstructed"]).size < 2:
            continue
        try:
            _, intercept, slope = recalibrate(
                frame.loc[train, "risk"], frame.loc[train, "y_reconstructed"],
                intercept_only=intercept_only,
            )
        except Exception:
            continue
        prediction[test] = expit(
            intercept + slope * logit(frame.loc[test, "risk"].clip(1e-6, 1 - 1e-6))
        )
        successful_folds += 1
    return prediction, successful_folds == 2


def run_database(database, reps):
    preparers = {"INSPIRE": prepare_inspire, "MIMIC": lambda: prepare_mimic()[0:2], "EICU": prepare_eicu}
    patient, serial = preparers[database]()
    truth = weighted_metrics(patient.y_full, patient.risk)
    rows = []
    for retention in RETENTIONS:
        for mechanism in MECHANISMS:
            for strength in STRENGTHS:
                for rep in range(reps):
                    condition_id = f"{database}|{retention}|{mechanism}|{strength}|{rep}"
                    stable_hash = int(hashlib.sha256(condition_id.encode()).hexdigest()[:8], 16)
                    seed = BASE_SEED + (stable_hash % 2_000_000_000)
                    rng = np.random.default_rng(seed)
                    sim = delete_and_reconstruct(patient, serial, mechanism, retention, strength, rng)
                    f = sim.patient
                    obs = f.R.eq(1) & f.y_reconstructed.notna()
                    common = {
                        "database": database, "retention_target": retention, "mechanism": mechanism,
                        "strength": strength, "replicate": rep, "seed": seed,
                        "measurement_retention_realized": sim.mean_measurement_retention,
                        "outcome_observed_fraction": float(obs.mean()),
                        "reconstructed_sensitivity": float(((f.y_reconstructed.eq(1)) & f.y_full.eq(1)).sum() / max(f.y_full.sum(), 1)),
                        "reconstructed_specificity": float(((f.y_reconstructed.eq(0)) & f.y_full.eq(0) & obs).sum() / max((f.y_full.eq(0) & obs).sum(), 1)),
                        "observability_probability_status": (
                            "conditional_realized_history_approximation"
                            if mechanism in {"history_MAR", "mixed_MNAR"}
                            else "exact_under_independent_design_deletion"
                        ),
                    }
                    rows.append(record("full_reference", "full", add_event_rate_inference(truth, truth["event_rate"]), common))
                    if obs.sum() < 20 or f.loc[obs, "y_reconstructed"].nunique() < 2:
                        continue
                    yobs, pobs = f.loc[obs, "y_reconstructed"], f.loc[obs, "risk"]
                    rows.append(record("naive", "reconstructed", add_event_rate_inference(weighted_metrics(yobs, pobs), truth["event_rate"]), common))
                    w_raw = 1 / f.loc[obs, "q_observed"].clip(0.005, 1)
                    raw_metrics = weighted_metrics(yobs, pobs, w_raw)
                    raw_metrics.update({"weight_p99": float(w_raw.quantile(0.99)), "weight_max": float(w_raw.max()), "weight_truncated": 0})
                    rows.append(record("IPAW_design_probability_untruncated", "reconstructed", add_event_rate_inference(raw_metrics, truth["event_rate"]), common))
                    w = w_raw.clip(upper=w_raw.quantile(0.99))
                    truncated_metrics = weighted_metrics(yobs, pobs, w)
                    truncated_metrics.update({"weight_p99": float(w.quantile(0.99)), "weight_max": float(w.max()), "weight_truncated": 1})
                    rows.append(record("IPAW_design_probability_truncated99", "reconstructed", add_event_rate_inference(truncated_metrics, truth["event_rate"]), common))
                    aipw, aipw_se = aipw_event_rate(f)
                    aipw_metrics = {k: np.nan for k in truth}
                    aipw_metrics.update({"n": len(f), "events": aipw * len(f), "event_rate": aipw,
                                         "mean_prediction": float(f.risk.mean()), "oe": aipw / f.risk.mean(),
                                         "ess": float((w_raw.sum() ** 2) / np.square(w_raw).sum()),
                                         "event_rate_se": aipw_se, "aipw_se": aipw_se})
                    rows.append(record("AIPW_design_probability", "reconstructed", add_event_rate_inference(aipw_metrics, truth["event_rate"]), common))

                    p_int_all, ok_int = crossfit_recalibration(f, rng, intercept_only=True)
                    p_slope_all, ok_slope = crossfit_recalibration(f, rng, intercept_only=False)
                    if ok_int:
                        rows.append(record("recalibration_intercept_apparent", "reconstructed", weighted_metrics(f.loc[obs, "y_reconstructed"], p_int_all[obs]), common))
                        rows.append(record("recalibration_intercept_truth", "full", weighted_metrics(f.y_full, p_int_all), common))
                    if ok_slope:
                        rows.append(record("recalibration_intercept_slope_apparent", "reconstructed", weighted_metrics(f.loc[obs, "y_reconstructed"], p_slope_all[obs]), common))
                        rows.append(record("recalibration_intercept_slope_truth", "full", weighted_metrics(f.y_full, p_slope_all), common))

                    reference_order = rng.permutation(len(f))
                    for fraction in REFERENCE_FRACTIONS:
                        sample_size = max(30, int(np.ceil(fraction * len(f))))
                        val, evaluation = reference_order[:sample_size], reference_order[sample_size:]
                        if np.unique(f.y_full.iloc[val]).size < 2 or len(evaluation) == 0:
                            continue
                        try:
                            _, av, bv = recalibrate(f.risk.iloc[val], f.y_full.iloc[val], intercept_only=False)
                        except Exception:
                            continue
                        p_reference = expit(av + bv * logit(f.risk.iloc[evaluation].clip(1e-6, 1 - 1e-6)))
                        label = f"reference_{int(round(fraction * 100)):02d}pct_recalibration"
                        reference_metrics = weighted_metrics(f.y_full.iloc[evaluation], p_reference)
                        reference_metrics.update({"reference_sample_n": sample_size, "evaluation_n": len(evaluation)})
                        rows.append(record(label, "full_heldout", reference_metrics, common))
                    lo, hi = mnar_event_bounds(f, gamma=2.0)
                    bound_metrics = {k: np.nan for k in truth}
                    bound_metrics.update({"n": len(f), "event_rate": (lo + hi) / 2, "mnar_lower": lo,
                                          "mnar_upper": hi, "mnar_covers_truth": int(lo <= truth["event_rate"] <= hi)})
                    rows.append(record("Gamma2_prediction_sensitivity_region", "full", bound_metrics, common))
    raw = pd.DataFrame(rows)
    raw.to_csv(SECURE / f"{database}_SIMULATION_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
    key = ["database", "retention_target", "mechanism", "strength", "method", "evaluation_target"]
    summaries = []
    for keys, g in raw.groupby(key, dropna=False):
        base = dict(zip(key, keys))
        for metric in ["event_rate", "oe", "brier", "auc", "calibration_intercept", "calibration_slope",
                       "outcome_observed_fraction", "reconstructed_sensitivity", "ess", "event_rate_se",
                       "weight_p99", "weight_max", "reference_sample_n", "evaluation_n"]:
            x = g[metric].dropna().to_numpy(float)
            if len(x):
                true_value = estimand_truth(base["method"], metric, truth)
                summaries.append({**base, "metric": metric, "n_replicates": len(x), "mean": x.mean(),
                                  "sd": x.std(ddof=1), "q025": np.quantile(x, .025), "q975": np.quantile(x, .975),
                                  "truth": true_value, "bias": x.mean() - true_value if np.isfinite(true_value) else np.nan,
                                  "rmse": np.sqrt(np.mean((x - true_value) ** 2)) if np.isfinite(true_value) else np.nan})
        if "mnar_covers_truth" in g:
            x = g.mnar_covers_truth.dropna()
            if len(x):
                summaries.append({**base, "metric": "MNAR_event_rate_coverage", "n_replicates": len(x), "mean": x.mean()})
        if "event_rate_coverage" in g:
            x = g.event_rate_coverage.dropna()
            if len(x):
                summaries.append({**base, "metric": "event_rate_interval_coverage", "n_replicates": len(x), "mean": x.mean()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES / f"Table_{database.lower()}_simulation_summary.csv", index=False)
    audit = {"database": database, "replicates_per_condition": reps, "conditions": 36,
             "patient_n": len(patient), "events": int(patient.y_full.sum()), "serial_rows": len(serial),
             "full_reference_metrics": truth, "replicate_rows": len(raw)}
    (OUTPUTS / f"{database}_SIMULATION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=["INSPIRE", "MIMIC", "EICU", "all"], default="all")
    parser.add_argument("--reps", type=int, default=300)
    args = parser.parse_args()
    for name in (["INSPIRE", "MIMIC", "EICU"] if args.database == "all" else [args.database]):
        run_database(name, args.reps)
