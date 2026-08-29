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
# # Discrimination-strength stress test
#
# This controlled simulation asks whether measurement-induced calibration bias
# persists for fixed scores spanning AUC 0.60-0.80. Scores are synthetic design
# objects generated from the retained endpoint; they are not clinical models.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(str(_release_path('analysis')))
sys.path.insert(0, str(ROOT / "code"))
from ascertainment_stress import delete_and_reconstruct, recalibrate, weighted_metrics  # noqa: E402

SEED = 20260827
TARGET_AUCS = (0.60, 0.70, 0.80)
TARGET_DELTAS = {0.60: 0.35828691, 0.70: 0.74161432, 0.80: 1.19023216}
N_REPLICATES = 100
RETENTION = 0.55
MECHANISM = "mixed_MNAR"
STRENGTH = "strong"


def logistic(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def log_odds(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1.0 - probability))


def z(series):
    values = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
    values = np.where(np.isfinite(values), values, np.nanmedian(values))
    sd = values.std()
    return np.zeros(len(values)) if sd < 1e-10 else (values - values.mean()) / sd


def load_database(database):
    if database == "MIMIC-IV":
        d = pd.read_csv(ROOT / "secure_work" / "MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
        d = d.loc[d.R_dense.eq(1)].copy()
        patient = pd.DataFrame({
            "reference_id": d.reference_id,
            "baseline_creatinine": d.baseline_creatinine,
            "y_full": d.Y_longitudinal.astype(int),
            "age_z": z(d.age),
            "sex_z": z(d.gender.astype("string").str.upper().map({"M": 1, "F": 0})),
            "stratum_z": z(pd.factorize(d.active_service)[0]),
        })
        serial = pd.read_csv(ROOT / "secure_work" / "MIMIC_CREATININE_SERIAL_SECURE.csv.gz")
    elif database == "eICU":
        d = pd.read_csv(ROOT / "eicu" / "secure" / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
        d = d.loc[d.R_dense.eq(1)].copy()
        patient = pd.DataFrame({
            "reference_id": d.reference_id,
            "baseline_creatinine": d.baseline_creatinine,
            "y_full": d.Y_longitudinal.astype(int),
            "age_z": z(d.age_num),
            "sex_z": z(d.gender.astype("string").str.lower().map({"male": 1, "female": 0})),
            "stratum_z": z(pd.factorize(d.hospitalid)[0]),
        })
        serial = pd.read_csv(ROOT / "eicu" / "secure" / "EICU_CREATININE_SERIAL_SECURE.csv.gz")
    else:
        raise ValueError(database)
    serial = serial.loc[serial.reference_id.isin(patient.reference_id), ["reference_id", "hour", "creatinine"]]
    return patient.reset_index(drop=True), serial.reset_index(drop=True)


def calibrated_synthetic_score(y, target_auc, rng):
    # Under equal-variance Gaussian class distributions,
    # AUC = Phi(delta / sqrt(2)). The subsequent logistic calibration makes the
    # retained-reference intercept and slope approximately 0 and 1.
    delta = TARGET_DELTAS[float(target_auc)]
    latent = delta * np.asarray(y, float) + rng.normal(size=len(y))
    initial = logistic(latent)
    calibrated, _, _ = recalibrate(initial, y, intercept_only=False)
    return np.clip(calibrated, 1e-6, 1 - 1e-6)


def one_replicate(database, patient, serial, target_auc, replicate):
    condition = f"{database}|{target_auc}|{replicate}|{SEED}"
    seed = SEED + int(hashlib.sha256(condition.encode()).hexdigest()[:8], 16) % 2_000_000_000
    rng = np.random.default_rng(seed)
    work = patient.copy()
    work["risk"] = calibrated_synthetic_score(work.y_full.to_numpy(), target_auc, rng)
    reference = weighted_metrics(work.y_full, work.risk)
    sim = delete_and_reconstruct(work, serial, MECHANISM, RETENTION, STRENGTH, rng)
    f = sim.patient
    observed = f.R.eq(1) & f.y_reconstructed.notna()
    if observed.sum() < 50 or f.loc[observed, "y_reconstructed"].nunique() < 2:
        return []
    apparent = weighted_metrics(f.loc[observed, "y_reconstructed"], f.loc[observed, "risk"])
    updated, _, _ = recalibrate(
        f.loc[observed, "risk"], f.loc[observed, "y_reconstructed"], intercept_only=False
    )
    # Apply the same fitted apparent calibration map to every retained-reference record.
    _, a, b = recalibrate(f.loc[observed, "risk"], f.loc[observed, "y_reconstructed"], intercept_only=False)
    updated_all = logistic(a + b * log_odds(f.risk.to_numpy()))
    updated_apparent = weighted_metrics(f.loc[observed, "y_reconstructed"], updated)
    updated_truth = weighted_metrics(f.y_full, updated_all)
    common = {
        "database": database, "target_auc": target_auc, "replicate": replicate, "seed": seed,
        "retention_target": RETENTION, "mechanism": MECHANISM, "strength": STRENGTH,
        "outcome_observed_fraction": float(observed.mean()),
        "reconstructed_sensitivity": float((f.y_reconstructed.eq(1) & f.y_full.eq(1)).sum() / max(f.y_full.sum(), 1)),
    }
    rows = []
    for method, metrics, target in (
        ("full_reference_score", reference, "retained_reference"),
        ("naive_apparent", apparent, "reconstructed_observed"),
        ("local_recalibration_apparent", updated_apparent, "reconstructed_observed"),
        ("local_recalibration_truth", updated_truth, "retained_reference"),
    ):
        rows.append({**common, "method": method, "evaluation_target": target, **metrics})
    return rows


# %%
all_rows = []
database_audits = {}
for database in ("MIMIC-IV", "eICU"):
    patient, serial = load_database(database)
    tasks = [(target, replicate) for target in TARGET_AUCS for replicate in range(N_REPLICATES)]
    nested = Parallel(n_jobs=8, prefer="processes", batch_size=1)(
        delayed(one_replicate)(database, patient, serial, target, replicate)
        for target, replicate in tasks
    )
    all_rows.extend(row for group in nested for row in group)
    database_audits[database] = {
        "n": len(patient), "events": int(patient.y_full.sum()), "serial_measurements": len(serial)
    }

raw = pd.DataFrame(all_rows)
raw.to_csv(ROOT / "secure_work" / "DISCRIMINATION_STRENGTH_STRESS_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")

metrics = ["auc", "oe", "brier", "calibration_intercept", "calibration_slope", "outcome_observed_fraction", "reconstructed_sensitivity"]
summary = []
keys = ["database", "target_auc", "method", "evaluation_target"]
for key, group in raw.groupby(keys, dropna=False):
    base = dict(zip(keys, key))
    for metric in metrics:
        values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
        if len(values):
            summary.append({
                **base, "metric": metric, "n_replicates": len(values),
                "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                "q025": float(np.quantile(values, .025)), "q975": float(np.quantile(values, .975)),
                "independent_unit": "Monte Carlo replicate",
            })
summary = pd.DataFrame(summary)
summary.to_csv(ROOT / "tables" / "Table_discrimination_strength_stress_test.csv", index=False)

audit = {
    "analysis": "controlled discrimination-strength stress test",
    "databases": database_audits,
    "target_auc_values": TARGET_AUCS,
    "replicates_per_database_auc": N_REPLICATES,
    "deletion_condition": {"mechanism": MECHANISM, "strength": STRENGTH, "target_per_measurement_retention": RETENTION},
    "score_construction": "synthetic equal-variance Gaussian class score, then logistic calibration to the retained endpoint",
    "independent_unit": "Monte Carlo replicate",
    "interpretation_boundary": "The synthetic scores are controlled design objects, not deployable clinical prediction models.",
    "patient_level_outputs_delivered": False,
}
(ROOT / "outputs" / "DISCRIMINATION_STRENGTH_STRESS_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(summary.loc[summary.metric.isin(["auc", "oe", "calibration_slope"])].to_string(index=False))
