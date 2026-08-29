#!/usr/bin/env python3
# %% [markdown]
# # Empirical cross-database monitoring-schedule transport
#
# This semi-synthetic experiment keeps target patients, retained laboratory
# trajectories and predictions fixed, then applies schedules sampled from each
# public database. Donor schedules are descriptive measurement patterns, not
# causal hospital policies.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


SEED = 20260829


def load_simulation_module(base: Path):
    sys.path.insert(0, str(base / "code"))
    path = base / "code" / "52_measurement_deletion_simulation.py"
    spec = importlib.util.spec_from_file_location("measurement_simulation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_seed(*parts: object) -> int:
    token = "|".join(map(str, parts))
    return SEED + int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 2_000_000_000


def donor_schedules(base: Path) -> dict[str, list[np.ndarray]]:
    paths = {
        "INSPIRE": base / "secure_work/INSPIRE_CREATININE_SERIAL_SECURE.csv.gz",
        "MIMIC": base / "secure_work/MIMIC_CREATININE_SERIAL_SECURE.csv.gz",
        "EICU": base / "eicu/secure/EICU_CREATININE_SERIAL_SECURE.csv.gz",
    }
    result: dict[str, list[np.ndarray]] = {}
    for database, path in paths.items():
        frame = pd.read_csv(path, usecols=lambda c: c in {"reference_id", "hour", "hours_after_surgery"})
        if "hours_after_surgery" in frame:
            frame = frame.rename(columns={"hours_after_surgery": "hour"})
        frame = frame.loc[frame.hour.gt(0) & frame.hour.le(168)].dropna()
        schedules = [
            np.sort(group.hour.to_numpy(float))
            for _, group in frame.groupby("reference_id", sort=False)
            if len(group)
        ]
        result[database] = schedules
    return result


def trajectory_lookup(serial: pd.DataFrame) -> dict[object, tuple[np.ndarray, np.ndarray]]:
    return {
        reference_id: (
            group.hour.to_numpy(float),
            group.creatinine.to_numpy(float),
        )
        for reference_id, group in serial.sort_values(["reference_id", "hour"]).groupby("reference_id", sort=False)
    }


def apply_schedule(
    target_hours: np.ndarray,
    target_values: np.ndarray,
    schedule: np.ndarray,
    tolerance_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(target_hours) or not len(schedule):
        return np.array([], dtype=float), np.array([], dtype=float)
    selected: set[int] = set()
    for planned_hour in schedule:
        index = int(np.argmin(np.abs(target_hours - planned_hour)))
        if abs(float(target_hours[index] - planned_hour)) <= tolerance_hours:
            selected.add(index)
    if not selected:
        return np.array([], dtype=float), np.array([], dtype=float)
    indices = np.array(sorted(selected), dtype=int)
    return target_hours[indices], target_values[indices]


def reconstruct_one(
    target_hours: np.ndarray,
    target_values: np.ndarray,
    baseline: float,
    schedule: np.ndarray,
    tolerance_hours: float,
) -> tuple[int, float, int]:
    hours, values = apply_schedule(target_hours, target_values, schedule, tolerance_hours)
    if not len(hours):
        return 0, np.nan, 0
    early = hours <= 48
    late = (hours > 48) & (hours <= 96)
    observed = int(early.any() and late.any())
    if not observed:
        return 0, np.nan, len(values)
    event = int(values[early].max() >= baseline + 0.3 or values.max() >= 1.5 * baseline)
    return 1, float(event), len(values)


def run_replicate(
    target_name: str,
    donor_name: str,
    patient: pd.DataFrame,
    trajectories: dict[int, tuple[np.ndarray, np.ndarray]],
    schedules: list[np.ndarray],
    tolerance_hours: float,
    replicate: int,
    simulation_module,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(stable_seed(target_name, donor_name, tolerance_hours, replicate))
    donor_index = rng.integers(0, len(schedules), size=len(patient))
    observed = np.zeros(len(patient), dtype=int)
    reconstructed = np.full(len(patient), np.nan)
    retained_counts = np.zeros(len(patient), dtype=int)
    for row_index, row in enumerate(patient.itertuples(index=False)):
        hours, values = trajectories[row.reference_id]
        observed[row_index], reconstructed[row_index], retained_counts[row_index] = reconstruct_one(
            hours,
            values,
            float(row.baseline_creatinine),
            schedules[int(donor_index[row_index])],
            tolerance_hours,
        )
    frame = patient.copy()
    frame["R"] = observed
    frame["y_reconstructed"] = reconstructed
    obs = frame.R.eq(1) & frame.y_reconstructed.notna()
    full_metrics = simulation_module.weighted_metrics(frame.y_full, frame.risk)
    common = {
        "target_database": target_name,
        "donor_schedule_database": donor_name,
        "tolerance_hours": tolerance_hours,
        "replicate": replicate,
        "n": len(frame),
        "full_events": int(frame.y_full.sum()),
        "outcome_observed_fraction": float(obs.mean()),
        "reconstructed_sensitivity": float(
            ((frame.y_reconstructed.eq(1)) & frame.y_full.eq(1)).sum() / max(frame.y_full.sum(), 1)
        ),
        "mean_retained_measurements": float(retained_counts.mean()),
        "donor_schedule_median_measurements": float(np.median([len(schedules[i]) for i in donor_index])),
    }
    rows = [{**common, "method": "full_reference", "evaluation_target": "retained_reference", **full_metrics}]
    if obs.sum() < 20 or frame.loc[obs, "y_reconstructed"].nunique() < 2:
        return rows
    rows.append({
        **common,
        "method": "naive",
        "evaluation_target": "reconstructed_observed",
        **simulation_module.weighted_metrics(frame.loc[obs, "y_reconstructed"], frame.loc[obs, "risk"]),
    })
    updated, successful = simulation_module.crossfit_recalibration(frame, rng, intercept_only=False)
    if successful:
        rows.append({
            **common,
            "method": "local_recalibration",
            "evaluation_target": "reconstructed_observed",
            **simulation_module.weighted_metrics(frame.loc[obs, "y_reconstructed"], updated[obs]),
        })
        rows.append({
            **common,
            "method": "local_recalibration",
            "evaluation_target": "retained_reference",
            **simulation_module.weighted_metrics(frame.y_full, updated),
        })
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "target_database", "donor_schedule_database", "tolerance_hours",
        "method", "evaluation_target",
    ]
    metrics = [
        "outcome_observed_fraction", "reconstructed_sensitivity",
        "mean_retained_measurements", "event_rate", "oe", "brier", "auc",
        "calibration_intercept", "calibration_slope",
    ]
    rows: list[dict[str, object]] = []
    for group_values, group in raw.groupby(keys, dropna=False):
        prefix = dict(zip(keys, group_values))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(values):
                rows.append({
                    **prefix,
                    "metric": metric,
                    "n_replicates": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    for name in ("secure_work", "tables", "outputs"):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)

    module = load_simulation_module(args.base)
    module.OUTPUTS = args.output_root / "outputs"
    targets = {
        "INSPIRE": module.prepare_inspire(),
        "MIMIC": module.prepare_mimic(),
        "EICU": module.prepare_eicu(),
    }
    schedules = donor_schedules(args.base)
    tasks = []
    for target_name, (patient, serial) in targets.items():
        trajectories = trajectory_lookup(serial)
        patient = patient.loc[patient.reference_id.isin(trajectories)].reset_index(drop=True)
        for donor_name, donor_values in schedules.items():
            for tolerance in (12.0, 24.0):
                for replicate in range(args.replicates):
                    tasks.append((target_name, donor_name, patient, trajectories, donor_values, tolerance, replicate))
    nested = Parallel(n_jobs=args.jobs, backend="threading", verbose=5)(
        delayed(run_replicate)(*task, module) for task in tasks
    )
    raw = pd.DataFrame([row for group in nested for row in group])
    raw_path = args.output_root / "secure_work/EMPIRICAL_SCHEDULE_TRANSPORT_REPLICATES_SECURE.csv.gz"
    raw.to_csv(raw_path, index=False, compression="gzip")
    summary = summarize(raw)
    table_path = args.output_root / "tables/Table_empirical_schedule_transport.csv"
    summary.to_csv(table_path, index=False)
    audit = {
        "status": "PASS",
        "replicates_per_condition": args.replicates,
        "target_databases": list(targets),
        "donor_schedule_databases": list(schedules),
        "tolerance_hours": [12, 24],
        "target_n": {name: int(len(value[0])) for name, value in targets.items()},
        "donor_schedule_n": {name: int(len(value)) for name, value in schedules.items()},
        "raw_rows": int(len(raw)),
        "summary_rows": int(len(summary)),
        "interpretation_boundary": (
            "Empirical schedules are sampled observed measurement patterns and are not causal hospital policies. "
            "Mapping uses the nearest retained target measurement within the prespecified tolerance."
        ),
    }
    (args.output_root / "outputs/EMPIRICAL_SCHEDULE_TRANSPORT_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
