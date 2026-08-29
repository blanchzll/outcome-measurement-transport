#!/usr/bin/env python3
# %% [markdown]
# # Non-renal replication using a longitudinal haemoglobin-decline endpoint
#
# The operational endpoint is a decrease of at least 2 g/dL from a harmonized
# peri-landmark baseline through 168 h. The baseline is the last measurement in
# -24 to 0 h, or (if absent) the first measurement in 0 to 6 h. Only measurements
# after that baseline contribute to the endpoint. This is a laboratory-trajectory
# endpoint, not adjudicated postoperative bleeding.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260829
MIMIC_HB_ITEMIDS = (50811, 51222, 51640, 51641)


def stable_seed(*parts: object) -> int:
    return SEED + int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % 2_000_000_000


def load_analysis_modules(base: Path):
    sys.path.insert(0, str(base / "code"))
    path = base / "code/52_measurement_deletion_simulation.py"
    spec = importlib.util.spec_from_file_location("measurement_simulation_hb", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_mimic(reference: pd.DataFrame, raw_root: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    con = duckdb.connect()
    con.register("reference", reference[["reference_id", "hadm_id", "intime"]])
    itemids = ",".join(map(str, MIMIC_HB_ITEMIDS))
    query = f"""
        SELECT r.reference_id,
               date_diff('minute', CAST(r.intime AS TIMESTAMP), CAST(l.charttime AS TIMESTAMP)) / 60.0 AS hour,
               CAST(l.valuenum AS DOUBLE) AS hemoglobin
        FROM read_csv_auto('{raw_root / 'hosp/labevents.csv.gz'}', header=true, sample_size=200000) AS l
        INNER JOIN reference AS r ON CAST(l.hadm_id AS BIGINT) = CAST(r.hadm_id AS BIGINT)
        WHERE CAST(l.itemid AS BIGINT) IN ({itemids})
          AND CAST(l.valuenum AS DOUBLE) BETWEEN 3 AND 25
          AND CAST(l.charttime AS TIMESTAMP) >= CAST(r.intime AS TIMESTAMP) - INTERVAL 24 HOUR
          AND CAST(l.charttime AS TIMESTAMP) <= CAST(r.intime AS TIMESTAMP) + INTERVAL 168 HOUR
    """
    serial = con.execute(query).fetch_df().sort_values(["reference_id", "hour"])
    serial.to_csv(cache, index=False, compression="gzip")
    return serial


def extract_eicu(reference: pd.DataFrame, raw_root: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    con = duckdb.connect()
    con.register("reference", reference[["reference_id", "patientunitstayid"]])
    query = f"""
        SELECT r.reference_id,
               CAST(l.labresultoffset AS DOUBLE) / 60.0 AS hour,
               CAST(l.labresult AS DOUBLE) AS hemoglobin
        FROM read_csv_auto('{raw_root / 'lab.csv.gz'}', header=true, sample_size=200000) AS l
        INNER JOIN reference AS r
          ON CAST(l.patientunitstayid AS BIGINT) = CAST(r.patientunitstayid AS BIGINT)
        WHERE lower(CAST(l.labname AS VARCHAR)) = 'hgb'
          AND CAST(l.labresult AS DOUBLE) BETWEEN 3 AND 25
          AND CAST(l.labresultoffset AS DOUBLE) >= -1440
          AND CAST(l.labresultoffset AS DOUBLE) <= 10080
    """
    serial = con.execute(query).fetch_df().sort_values(["reference_id", "hour"])
    serial.to_csv(cache, index=False, compression="gzip")
    return serial


def harmonize_baseline(reference: pd.DataFrame, serial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = serial.loc[serial.hour.between(-24, 6, inclusive="both")].copy()
    rows = []
    for reference_id, group in candidates.sort_values("hour").groupby("reference_id", sort=False):
        before = group.loc[group.hour.le(0)]
        chosen = before.iloc[-1] if len(before) else group.iloc[0]
        rows.append({
            "reference_id": reference_id,
            "baseline_hour": float(chosen.hour),
            "baseline_hemoglobin": float(chosen.hemoglobin),
        })
    baseline = pd.DataFrame(rows)
    clean_reference = reference.drop(columns=["baseline_hemoglobin"], errors="ignore").merge(
        baseline, on="reference_id", how="left"
    )
    post = serial.merge(baseline[["reference_id", "baseline_hour"]], on="reference_id", how="inner")
    post = post.loc[post.hour.gt(post.baseline_hour) & post.hour.le(168)].drop(columns="baseline_hour")
    return clean_reference, post.reset_index(drop=True)


def build_endpoint(reference: pd.DataFrame, serial: pd.DataFrame) -> pd.DataFrame:
    grouped = serial.groupby("reference_id", sort=False)
    summary = grouped.agg(
        n_hemoglobin_0_168h=("hemoglobin", "size"),
        hb_first_hour=("hour", "min"),
        hb_last_hour=("hour", "max"),
        minimum_hemoglobin=("hemoglobin", "min"),
    ).reset_index()
    early = serial.loc[serial.hour.le(48)].groupby("reference_id").size().rename("n_0_48h")
    late = serial.loc[serial.hour.gt(48) & serial.hour.le(96)].groupby("reference_id").size().rename("n_48_96h")
    summary = summary.merge(early, on="reference_id", how="left").merge(late, on="reference_id", how="left")
    summary[["n_0_48h", "n_48_96h"]] = summary[["n_0_48h", "n_48_96h"]].fillna(0)
    summary["hb_span_hours"] = summary.hb_last_hour - summary.hb_first_hour
    result = reference.merge(summary, on="reference_id", how="left")
    result["R_dense_hb"] = (
        result.n_hemoglobin_0_168h.fillna(0).ge(3)
        & result.n_0_48h.fillna(0).gt(0)
        & result.n_48_96h.fillna(0).gt(0)
        & result.hb_span_hours.fillna(0).ge(72)
    ).astype(int)
    result["Y_hb_decline"] = (
        result.minimum_hemoglobin.le(result.baseline_hemoglobin - 2.0)
    ).astype("Int64")
    result.loc[result.minimum_hemoglobin.isna(), "Y_hb_decline"] = pd.NA
    return result


def make_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(C=0.25, solver="liblinear", max_iter=2000))])


def prepare_mimic(base: Path, raw_root: Path, secure: Path):
    reference = pd.read_csv(base / "secure_work/MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
    serial = extract_mimic(reference, raw_root, secure / "MIMIC_HEMOGLOBIN_SERIAL_SECURE.csv.gz")
    reference, serial = harmonize_baseline(reference, serial)
    endpoint = build_endpoint(reference, serial)
    dense = endpoint.loc[endpoint.R_dense_hb.eq(1)].sort_values(["intime", "reference_id"]).copy()
    split = int(np.floor(0.60 * len(dense)))
    train, test = dense.iloc[:split].copy(), dense.iloc[split:].copy()
    numeric = ["age", "baseline_hemoglobin", "baseline_creatinine", "baseline_albumin", "baseline_bun"]
    categorical = ["gender", "active_service", "first_careunit", "admission_type"]
    model = make_model(numeric, categorical)
    model.fit(train[numeric + categorical], train.Y_hb_decline.astype(int))
    test["risk"] = model.predict_proba(test[numeric + categorical])[:, 1]
    patient = test[["reference_id", "baseline_hemoglobin", "Y_hb_decline", "risk"]].rename(
        columns={"baseline_hemoglobin": "baseline", "Y_hb_decline": "y_full"}
    )
    return endpoint, patient.reset_index(drop=True), serial.loc[serial.reference_id.isin(patient.reference_id)].reset_index(drop=True)


def prepare_eicu(base: Path, raw_root: Path, secure: Path):
    reference = pd.read_csv(base / "eicu/secure/EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
    serial = extract_eicu(reference, raw_root, secure / "EICU_HEMOGLOBIN_SERIAL_SECURE.csv.gz")
    reference, serial = harmonize_baseline(reference, serial)
    endpoint = build_endpoint(reference, serial)
    dense = endpoint.loc[endpoint.R_dense_hb.eq(1)].copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_index, test_index = next(splitter.split(dense, groups=dense.hospitalid))
    train, test = dense.iloc[train_index].copy(), dense.iloc[test_index].copy()
    numeric = ["age_num", "baseline_hemoglobin", "baseline_creatinine", "admissionweight", "hospitaladmitoffset"]
    categorical = ["gender", "ethnicity", "unittype", "hospitaladmitsource", "region"]
    model = make_model(numeric, categorical)
    model.fit(train[numeric + categorical], train.Y_hb_decline.astype(int))
    test["risk"] = model.predict_proba(test[numeric + categorical])[:, 1]
    patient = test[["reference_id", "baseline_hemoglobin", "Y_hb_decline", "risk"]].rename(
        columns={"baseline_hemoglobin": "baseline", "Y_hb_decline": "y_full"}
    )
    return endpoint, patient.reset_index(drop=True), serial.loc[serial.reference_id.isin(patient.reference_id)].reset_index(drop=True)


def trajectory_lookup(serial: pd.DataFrame) -> dict[object, tuple[np.ndarray, np.ndarray]]:
    return {
        reference_id: (group.hour.to_numpy(float), group.hemoglobin.to_numpy(float))
        for reference_id, group in serial.sort_values(["reference_id", "hour"]).groupby("reference_id", sort=False)
    }


def apply_schedule(hours: np.ndarray, values: np.ndarray, schedule: np.ndarray, tolerance: float):
    selected: set[int] = set()
    for planned in schedule:
        index = int(np.argmin(np.abs(hours - planned)))
        if abs(float(hours[index] - planned)) <= tolerance:
            selected.add(index)
    if not selected:
        return np.array([]), np.array([])
    indices = np.array(sorted(selected), dtype=int)
    return hours[indices], values[indices]


def one_replicate(target_name, donor_name, patient, trajectories, donor_schedules, tolerance, replicate, module):
    rng = np.random.default_rng(stable_seed(target_name, donor_name, tolerance, replicate))
    selected_schedules = rng.integers(0, len(donor_schedules), size=len(patient))
    observed = np.zeros(len(patient), dtype=int)
    reconstructed = np.full(len(patient), np.nan)
    retained = np.zeros(len(patient), dtype=int)
    for i, row in enumerate(patient.itertuples(index=False)):
        hours, values = trajectories[row.reference_id]
        kept_hours, kept_values = apply_schedule(hours, values, donor_schedules[int(selected_schedules[i])], tolerance)
        retained[i] = len(kept_values)
        if len(kept_values) and (kept_hours <= 48).any() and ((kept_hours > 48) & (kept_hours <= 96)).any():
            observed[i] = 1
            reconstructed[i] = float(kept_values.min() <= float(row.baseline) - 2.0)
    frame = patient.copy()
    frame["R"] = observed
    frame["y_reconstructed"] = reconstructed
    obs = frame.R.eq(1)
    common = {
        "target_database": target_name, "donor_schedule_database": donor_name,
        "tolerance_hours": tolerance, "replicate": replicate, "n": len(frame),
        "events": int(frame.y_full.sum()), "outcome_observed_fraction": float(obs.mean()),
        "reconstructed_sensitivity": float(((frame.y_reconstructed.eq(1)) & frame.y_full.eq(1)).sum() / max(frame.y_full.sum(), 1)),
        "mean_retained_measurements": float(retained.mean()),
    }
    rows = [{**common, "method": "full_reference", "evaluation_target": "retained_reference", **module.weighted_metrics(frame.y_full, frame.risk)}]
    if obs.sum() >= 20 and frame.loc[obs, "y_reconstructed"].nunique() == 2:
        rows.append({**common, "method": "naive", "evaluation_target": "reconstructed_observed", **module.weighted_metrics(frame.loc[obs, "y_reconstructed"], frame.loc[obs, "risk"])})
        updated, success = module.crossfit_recalibration(frame, rng, intercept_only=False)
        if success:
            rows.append({**common, "method": "local_recalibration", "evaluation_target": "reconstructed_observed", **module.weighted_metrics(frame.loc[obs, "y_reconstructed"], updated[obs])})
            rows.append({**common, "method": "local_recalibration", "evaluation_target": "retained_reference", **module.weighted_metrics(frame.y_full, updated)})
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_database", "donor_schedule_database", "tolerance_hours", "method", "evaluation_target"]
    metrics = ["outcome_observed_fraction", "reconstructed_sensitivity", "mean_retained_measurements", "event_rate", "oe", "brier", "auc", "calibration_intercept", "calibration_slope"]
    rows = []
    for values, group in raw.groupby(keys):
        prefix = dict(zip(keys, values))
        for metric in metrics:
            x = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(x):
                rows.append({**prefix, "metric": metric, "n_replicates": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "q025": float(np.quantile(x, .025)), "q975": float(np.quantile(x, .975))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--mimic-root", required=True, type=Path)
    parser.add_argument("--eicu-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    for name in ("secure_work", "tables", "outputs"):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)
    module = load_analysis_modules(args.base)
    module.OUTPUTS = args.output_root / "outputs"
    endpoints = {}
    targets = {}
    for name, prepared in {
        "MIMIC": prepare_mimic(args.base, args.mimic_root, args.output_root / "secure_work"),
        "EICU": prepare_eicu(args.base, args.eicu_root, args.output_root / "secure_work"),
    }.items():
        endpoint, patient, serial = prepared
        endpoints[name] = endpoint
        targets[name] = (patient, serial)
    donor_schedules = {
        name: [np.sort(group.hour.to_numpy(float)) for _, group in endpoint_serial.groupby("reference_id")]
        for name, (_, endpoint_serial) in targets.items()
    }
    # Use all candidate haemoglobin schedules, not only the held-out target trajectory schedules.
    donor_schedules["MIMIC"] = [np.sort(g.hour.to_numpy(float)) for _, g in pd.read_csv(args.output_root / "secure_work/MIMIC_HEMOGLOBIN_SERIAL_SECURE.csv.gz").groupby("reference_id")]
    donor_schedules["EICU"] = [np.sort(g.hour.to_numpy(float)) for _, g in pd.read_csv(args.output_root / "secure_work/EICU_HEMOGLOBIN_SERIAL_SECURE.csv.gz").groupby("reference_id")]
    tasks = []
    for target_name, (patient, serial) in targets.items():
        trajectories = trajectory_lookup(serial)
        patient = patient.loc[patient.reference_id.isin(trajectories)].reset_index(drop=True)
        for donor_name, schedules in donor_schedules.items():
            for tolerance in (12.0, 24.0):
                for replicate in range(args.replicates):
                    tasks.append((target_name, donor_name, patient, trajectories, schedules, tolerance, replicate, module))
    nested = Parallel(n_jobs=args.jobs, backend="threading", verbose=5)(delayed(one_replicate)(*task) for task in tasks)
    raw = pd.DataFrame([row for group in nested for row in group])
    raw.to_csv(args.output_root / "secure_work/HEMOGLOBIN_ENDPOINT_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
    summary = summarize(raw)
    summary.to_csv(args.output_root / "tables/Table_hemoglobin_endpoint_replication.csv", index=False)
    flow_rows = []
    for database, endpoint in endpoints.items():
        flow_rows.append({
            "database": database,
            "candidate_n": int(len(endpoint)),
            "any_postop_hemoglobin_n": int(endpoint.n_hemoglobin_0_168h.notna().sum()),
            "dense_reference_n": int(endpoint.R_dense_hb.sum()),
            "dense_reference_events": int(endpoint.loc[endpoint.R_dense_hb.eq(1), "Y_hb_decline"].sum()),
        })
    flow = pd.DataFrame(flow_rows)
    flow.to_csv(args.output_root / "tables/Table_hemoglobin_endpoint_flow.csv", index=False)
    audit = {
        "status": "PASS",
        "endpoint": "haemoglobin decrease of at least 2 g/dL after a harmonized -24-to-+6-h peri-landmark baseline through 168 h",
        "clinical_claim_boundary": "Operational laboratory-trajectory endpoint; not adjudicated postoperative bleeding.",
        "flows": flow_rows,
        "replicates_per_condition": args.replicates,
        "raw_rows": int(len(raw)),
        "summary_rows": int(len(summary)),
    }
    (args.output_root / "outputs/HEMOGLOBIN_ENDPOINT_REPLICATION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
