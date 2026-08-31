# # Primary measurement-deletion decomposition and nested uncertainty
#
# This targeted audit separates selection from endpoint reconstruction at the
# prespecified 35% retention, strong mixed-MNAR condition. It also repeats the
# experiment after resampling patients (or hospitals for eICU), distinguishing
# Monte Carlo variation from sampling variation.

# %%
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from release_paths import release_path as _release_path


ROOT = Path(str(_release_path("analysis")))
CODE = ROOT / "code"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
SECURE = ROOT / "secure_work"
RETENTION = 0.35
MECHANISM = "mixed_MNAR"
STRENGTH = "strong"
BASE_SEED = 20260831


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


simulation = load_module("measurement_simulation", CODE / "52_measurement_deletion_simulation.py")
stress = load_module("ascertainment_stress_primary", CODE / "ascertainment_stress.py")


def prepare(database: str):
    if database == "INSPIRE":
        patient, serial = simulation.prepare_inspire()
    elif database == "MIMIC":
        patient, serial = simulation.prepare_mimic()[0:2]
    elif database == "EICU":
        patient, serial = simulation.prepare_eicu()
        source = pd.read_csv(
            ROOT / "eicu/secure/EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz",
            usecols=["reference_id", "hospitalid"],
        ).drop_duplicates("reference_id")
        patient = patient.merge(source, on="reference_id", how="left", validate="one_to_one")
        if patient.hospitalid.isna().any():
            raise RuntimeError("eICU held-out patients could not all be mapped to hospital IDs")
    else:
        raise ValueError(database)
    return patient.reset_index(drop=True), serial.reset_index(drop=True)


def decompose(frame: pd.DataFrame, measurement_retention: float) -> dict[str, float]:
    observed = frame.R.eq(1) & frame.y_reconstructed.notna()
    full_rate = float(frame.y_full.mean())
    full_prediction = float(frame.risk.mean())
    selected_reference_rate = float(frame.loc[observed, "y_full"].mean())
    reconstructed_rate = float(frame.loc[observed, "y_reconstructed"].mean())
    selected_prediction = float(frame.loc[observed, "risk"].mean())
    sensitivity = float(
        ((frame.y_reconstructed.eq(1)) & frame.y_full.eq(1)).sum()
        / max(int(frame.y_full.sum()), 1)
    )
    specificity = float(
        ((frame.y_reconstructed.eq(0)) & frame.y_full.eq(0) & observed).sum()
        / max(int((frame.y_full.eq(0) & observed).sum()), 1)
    )
    return {
        "n_full": int(len(frame)),
        "n_evaluable": int(observed.sum()),
        "full_reference_event_rate": full_rate,
        "evaluable_reference_event_rate": selected_reference_rate,
        "evaluable_reconstructed_event_rate": reconstructed_rate,
        "full_mean_prediction": full_prediction,
        "evaluable_mean_prediction": selected_prediction,
        "full_reference_oe": full_rate / full_prediction,
        "evaluable_reference_oe": selected_reference_rate / selected_prediction,
        "evaluable_reconstructed_oe": reconstructed_rate / selected_prediction,
        "outcome_observed_fraction": float(observed.mean()),
        "measurement_retention_realized": float(measurement_retention),
        "reconstructed_sensitivity": sensitivity,
        "reconstructed_specificity": specificity,
        "selection_component": selected_reference_rate - full_rate,
        "reconstruction_component": reconstructed_rate - selected_reference_rate,
        "total_apparent_event_rate_bias": reconstructed_rate - full_rate,
    }


def duplicate_rows(patient: pd.DataFrame, serial: pd.DataFrame, selected: np.ndarray, prefix: str):
    sampled = patient.iloc[selected].copy().reset_index(drop=True)
    sampled["source_reference_id"] = sampled.reference_id.astype(str)
    sampled["reference_id"] = [f"{prefix}_{i}" for i in range(len(sampled))]
    key = sampled[["reference_id", "source_reference_id"]]
    serial_copy = key.merge(
        serial.assign(source_reference_id=serial.reference_id.astype(str)).drop(columns="reference_id"),
        on="source_reference_id", how="left", validate="many_to_many",
    ).drop(columns="source_reference_id")
    return sampled.drop(columns="source_reference_id"), serial_copy


def resample(database: str, patient: pd.DataFrame, serial: pd.DataFrame, rng: np.random.Generator, replicate: int):
    if database != "EICU":
        selected = rng.integers(0, len(patient), size=len(patient))
        return duplicate_rows(patient, serial, selected, f"b{replicate}")
    hospitals = patient.hospitalid.drop_duplicates().to_numpy()
    sampled_hospitals = rng.choice(hospitals, size=len(hospitals), replace=True)
    parts = []
    for draw, hospital in enumerate(sampled_hospitals):
        part = patient.loc[patient.hospitalid.eq(hospital)].copy()
        part["bootstrap_block"] = draw
        parts.append(part)
    sampled = pd.concat(parts, ignore_index=True)
    sampled_serial_keys = sampled[["reference_id"]].copy()
    sampled_serial_keys["source_reference_id"] = sampled_serial_keys.reference_id.astype(str)
    sampled_serial_keys["reference_id"] = [f"b{replicate}_{i}" for i in range(len(sampled))]
    sampled["reference_id"] = sampled_serial_keys.reference_id.to_numpy()
    serial_copy = sampled_serial_keys.merge(
        serial.assign(source_reference_id=serial.reference_id.astype(str)).drop(columns="reference_id"),
        on="source_reference_id", how="left", validate="many_to_many",
    ).drop(columns="source_reference_id")
    return sampled.drop(columns=["hospitalid", "bootstrap_block"]), serial_copy


def summarize(raw: pd.DataFrame, bootstrap: bool) -> pd.DataFrame:
    id_columns = {"database", "replicate", "sampling_layer"}
    rows = []
    for database, group in raw.groupby("database"):
        for metric in [c for c in raw.columns if c not in id_columns]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            rows.append({
                "database": database,
                "sampling_layer": "patient_or_hospital_plus_deletion" if bootstrap else "deletion_given_fixed_cohort",
                "metric": metric,
                "n_replicates": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "mcse_of_mean": float(values.std(ddof=1) / np.sqrt(len(values))),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
            })
    return pd.DataFrame(rows)


def run(reps: int, nested_reps: int):
    fixed_rows, nested_rows, audits = [], [], {}
    for database in ("INSPIRE", "MIMIC", "EICU"):
        patient, serial = prepare(database)
        unique_patients = int(patient.reference_id.nunique())
        if unique_patients != len(patient):
            raise RuntimeError(f"{database}: reference_id is not unique")
        audits[database] = {
            "n": len(patient), "events": int(patient.y_full.sum()),
            "unique_reference_ids": unique_patients,
            "resampling_unit": "hospital" if database == "EICU" else "patient",
            "n_hospitals": int(patient.hospitalid.nunique()) if database == "EICU" else None,
        }
        for replicate in range(reps):
            rng = np.random.default_rng(BASE_SEED + 10_000 * (1 + ("INSPIRE", "MIMIC", "EICU").index(database)) + replicate)
            result = stress.delete_and_reconstruct(patient, serial, MECHANISM, RETENTION, STRENGTH, rng)
            fixed_rows.append({"database": database, "replicate": replicate,
                               "sampling_layer": "deletion_given_fixed_cohort",
                               **decompose(result.patient, result.mean_measurement_retention)})
        for replicate in range(nested_reps):
            rng = np.random.default_rng(BASE_SEED + 1_000_000 + 100_000 * (1 + ("INSPIRE", "MIMIC", "EICU").index(database)) + replicate)
            boot_patient, boot_serial = resample(database, patient, serial, rng, replicate)
            result = stress.delete_and_reconstruct(boot_patient, boot_serial, MECHANISM, RETENTION, STRENGTH, rng)
            nested_rows.append({"database": database, "replicate": replicate,
                                "sampling_layer": "patient_or_hospital_plus_deletion",
                                **decompose(result.patient, result.mean_measurement_retention)})
        # Database-level checkpoints protect the long targeted audit from a
        # later reporting failure. They remain in the secure workspace.
        pd.DataFrame(fixed_rows).to_csv(
            SECURE / "PRIMARY_DECOMPOSITION_FIXED_COHORT_REPLICATES_PARTIAL_SECURE.csv.gz",
            index=False, compression="gzip",
        )
        pd.DataFrame(nested_rows).to_csv(
            SECURE / "PRIMARY_DECOMPOSITION_NESTED_REPLICATES_PARTIAL_SECURE.csv.gz",
            index=False, compression="gzip",
        )
    fixed = pd.DataFrame(fixed_rows)
    nested = pd.DataFrame(nested_rows)
    combined_summary = pd.concat([summarize(fixed, False), summarize(nested, True)], ignore_index=True)
    key_metrics = [
        "full_reference_event_rate", "evaluable_reference_event_rate",
        "evaluable_reconstructed_event_rate", "full_mean_prediction",
        "evaluable_mean_prediction", "full_reference_oe", "evaluable_reference_oe",
        "evaluable_reconstructed_oe", "outcome_observed_fraction",
        "reconstructed_sensitivity", "reconstructed_specificity",
        "selection_component", "reconstruction_component", "total_apparent_event_rate_bias",
    ]
    combined_summary.loc[combined_summary.metric.isin(key_metrics)].to_csv(
        TABLES / "Table_primary_selection_reconstruction_decomposition.csv", index=False
    )
    fixed.to_csv(SECURE / "PRIMARY_DECOMPOSITION_FIXED_COHORT_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
    nested.to_csv(SECURE / "PRIMARY_DECOMPOSITION_NESTED_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
    audit = {
        "status": "PASS",
        "condition": {"retention_target": RETENTION, "mechanism": MECHANISM, "strength": STRENGTH},
        "fixed_cohort_replicates": reps, "nested_replicates": nested_reps,
        "cohorts": audits,
        "interval_interpretation": {
            "deletion_given_fixed_cohort": "2.5th-97.5th percentiles across deletion draws, conditional on the locked cohort and risk scores",
            "patient_or_hospital_plus_deletion": "2.5th-97.5th percentiles after resampling patients (INSPIRE/MIMIC) or hospitals (eICU) and then drawing deletion",
            "mcse_of_mean": "simulation Monte Carlo standard error of each reported replicate mean",
        },
        "limitations": [
            "The nested experiment does not resample source-model development.",
            "eICU resamples hospitals; only the held-out hospital cohort is represented.",
            "Intervals quantify empirical stress-test uncertainty, not causal identification under MNAR.",
        ],
    }
    (OUTPUTS / "PRIMARY_DECOMPOSITION_NESTED_UNCERTAINTY_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--nested-reps", type=int, default=200)
    args = parser.parse_args()
    run(args.reps, args.nested_reps)
