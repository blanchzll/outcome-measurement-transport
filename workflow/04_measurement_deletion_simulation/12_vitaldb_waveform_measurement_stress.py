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
# # Prespecified VitalDB waveform-model measurement stress test
#
# This extension asks whether the apparent-versus-retained calibration
# divergence persists after adding high-resolution intraoperative physiology to
# the frozen clinical-table ridge model. It is a mechanism stress test, not an
# algorithm competition or a claim of clinical impact.

# %%
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SEED = 20260830
PRIMARY_RETENTION = 0.35
PRIMARY_MECHANISM = "mixed_MNAR"
PRIMARY_STRENGTH = "strong"
DEFAULT_REPLICATES = 300
MODEL_COLUMNS = {
    "clinical_table_ridge": "risk_clinical_table_ridge",
    "duration_adjusted_clinical_ridge": "risk_duration_adjusted_clinical_ridge",
    "waveform_enhanced_ridge": "risk_waveform_enhanced_ridge",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-level", required=True, type=Path)
    parser.add_argument("--serial", required=True, type=Path)
    parser.add_argument("--model-predictions", required=True, type=Path)
    parser.add_argument("--stress-module", required=True, type=Path)
    parser.add_argument("--vitaldb-simulation-module", required=True, type=Path)
    parser.add_argument("--secure-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args()

    stress = load_module(args.stress_module, "waveform_measurement_stress_core")
    simulation = load_module(args.vitaldb_simulation_module, "waveform_measurement_vitaldb_sim")
    patient, serial, base_audit = simulation.prepare_analysis(args.case_level, args.serial, stress)
    predictions = pd.read_csv(args.model_predictions, low_memory=False)
    required = {"caseid", "creatinine_event_168h", *MODEL_COLUMNS.values()}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Model prediction file lacks required columns: {missing}")
    if predictions.caseid.duplicated().any():
        raise ValueError("Model prediction file must have one row per caseid")

    joined = patient.merge(
        predictions[["caseid", "creatinine_event_168h", *MODEL_COLUMNS.values()]],
        left_on="reference_id",
        right_on="caseid",
        how="left",
        validate="one_to_one",
    )
    if joined[list(MODEL_COLUMNS.values())].isna().any().any():
        raise RuntimeError("Frozen held-out cases and model-prediction cases do not match")
    if not np.array_equal(
        joined.y_full.astype(int).to_numpy(), joined.creatinine_event_168h.astype(int).to_numpy()
    ):
        raise RuntimeError("Outcome mismatch between reference cases and model predictions")
    max_baseline_difference = float(
        np.max(np.abs(joined.risk - joined[MODEL_COLUMNS["clinical_table_ridge"]]))
    )
    if max_baseline_difference > 1e-12:
        raise RuntimeError(
            f"Clinical-table predictions do not reproduce the frozen risk engine: {max_baseline_difference}"
        )

    # Restrict the existing validated factorial engine to the frozen primary
    # condition. The same deterministic seed schedule is retained.
    simulation.MECHANISMS = [PRIMARY_MECHANISM]
    simulation.RETENTIONS = [PRIMARY_RETENTION]
    simulation.STRENGTHS = [PRIMARY_STRENGTH]

    raw_frames = []
    summary_frames = []
    truth_metrics = {}
    for model, risk_column in MODEL_COLUMNS.items():
        model_patient = joined[patient.columns].copy()
        model_patient["risk"] = joined[risk_column].clip(1e-6, 1 - 1e-6).to_numpy()
        truth_metrics[model] = stress.weighted_metrics(model_patient.y_full, model_patient.risk)
        raw, summary = simulation.run_simulation(model_patient, serial, stress, args.reps)
        raw.insert(0, "model", model)
        summary.insert(0, "model", model)
        raw_frames.append(raw)
        summary_frames.append(summary)

    raw_all = pd.concat(raw_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    args.secure_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw_all.to_csv(args.secure_output, index=False, compression="gzip")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_all.to_csv(args.summary_output, index=False)

    diagnostic = summary_all.loc[
        summary_all.method.isin(
            [
                "naive",
                "IPAW_design_probability_truncated99",
                "AIPW_design_probability",
                "recalibration_intercept_truth",
                "recalibration_intercept_slope_truth",
            ]
        )
        & summary_all.metric.isin(["oe", "calibration_intercept", "calibration_slope", "brier"])
    ].copy()
    audit = {
        "analysis": "prespecified VitalDB waveform-model measurement stress test",
        "seed": SEED,
        "condition": {
            "mechanism": PRIMARY_MECHANISM,
            "target_measurement_retention": PRIMARY_RETENTION,
            "strength": PRIMARY_STRENGTH,
            "replicates": args.reps,
        },
        "heldout_n": int(len(patient)),
        "heldout_events": int(patient.y_full.sum()),
        "baseline_prediction_max_absolute_difference": max_baseline_difference,
        "full_reference_metrics": truth_metrics,
        "simulation_rows": int(len(raw_all)),
        "summary_rows": int(len(summary_all)),
        "diagnostic_summary": diagnostic.to_dict(orient="records"),
        "base_analysis_audit": base_audit,
        "interpretation_boundary": (
            "Tests robustness of the measurement-process conclusion across the historical clinical, "
            "duration-adjusted clinical, and waveform-enhanced real risk engines; "
            "does not establish causal effects of hypotension, transportability, or clinical impact."
        ),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
