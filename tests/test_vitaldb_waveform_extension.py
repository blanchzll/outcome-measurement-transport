from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_waveform_model_table_reports_paired_deltas():
    comparison = load(
        "workflow/04_measurement_deletion_simulation/11_vitaldb_waveform_model_comparison.py",
        "waveform_comparison_test",
    )
    stress = load("ascertainment_stress.py", "waveform_stress_test")
    y = np.array([0, 0, 0, 1, 0, 1, 0, 1], dtype=int)
    predictions = {
        "clinical_table_ridge": np.array([0.05, 0.08, 0.12, 0.40, 0.18, 0.52, 0.25, 0.70]),
        "duration_adjusted_clinical_ridge": np.array(
            [0.05, 0.07, 0.11, 0.43, 0.16, 0.56, 0.23, 0.73]
        ),
        "waveform_enhanced_ridge": np.array([0.04, 0.06, 0.10, 0.48, 0.14, 0.61, 0.20, 0.78]),
    }
    rows = []
    for replicate in range(5):
        for model, probability in predictions.items():
            rows.append({"replicate": replicate, "model": model, **stress.weighted_metrics(y, probability)})
    table, audit = comparison.comparison_table(y, predictions, pd.DataFrame(rows), stress)
    assert len(table) == 25
    assert set(table.comparison) == {
        "model_performance",
        "waveform_minus_duration_adjusted_clinical_paired_delta",
        "waveform_minus_historical_clinical_paired_delta",
    }
    primary = audit["paired_deltas"][
        "waveform_minus_duration_adjusted_clinical_paired_delta"
    ]
    assert primary["auc"]["estimate"] >= 0
    assert primary["brier"]["estimate"] < 0


def test_waveform_measurement_qa_requires_direction_and_magnitude():
    qa = load(
        "workflow/00_provenance_and_estimands/13_vitaldb_waveform_extension_qa.py",
        "waveform_qa_test",
    )
    stress = pd.DataFrame(
        {
            "model": ["waveform_enhanced_ridge"] * 300,
            "method": ["recalibration_intercept_slope_truth"] * 300,
            "evaluation_target": ["full"] * 300,
            "calibration_intercept": np.linspace(0.20, 0.40, 300),
            "calibration_slope": np.linspace(0.60, 0.80, 300),
        }
    )
    result = qa.calibration_robustness(stress, "waveform_enhanced_ridge")
    assert result["passed"] is True
    assert result["metrics"]["calibration_intercept"]["n_replicates"] == 300
    assert result["metrics"]["calibration_intercept"]["directional_consistency"] == 1.0
