from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "workflow/04_measurement_deletion_simulation/99_primary_decomposition_and_nested_uncertainty.py"
)


class DummySimulation:
    def __init__(self) -> None:
        self.folds: list[np.ndarray] = []
        self.observed: list[np.ndarray] = []

    def crossfit_recalibration(self, frame, rng, intercept_only):
        self.folds.append(rng.integers(0, 2, size=len(frame)))
        observed = frame.R.eq(1).to_numpy() & frame.y_reconstructed.notna().to_numpy()
        self.observed.append(observed)
        rate = float(frame.loc[observed, "y_reconstructed"].mean())
        return np.full(len(frame), rate), True

    @staticmethod
    def weighted_metrics(y, prediction):
        return {
            "oe": float(np.asarray(y, dtype=float).mean() / np.asarray(prediction, dtype=float).mean()),
            "calibration_slope": 1.0,
        }


def load_calibration_control(simulation: DummySimulation):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "calibration_control")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np, "pd": pd, "simulation": simulation}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace["calibration_control"]


def test_calibration_control_holds_records_and_folds_fixed():
    simulation = DummySimulation()
    control = load_calibration_control(simulation)

    frame = pd.DataFrame({
        "R": [1] * 20 + [0] * 20,
        "y_full": [1] * 10 + [0] * 30,
        "y_reconstructed": [1] * 5 + [0] * 15 + [np.nan] * 20,
        "risk": np.linspace(0.05, 0.45, 40),
    })
    result = control(frame, seed=731)

    assert len(simulation.folds) == 2
    np.testing.assert_array_equal(simulation.folds[0], simulation.folds[1])
    np.testing.assert_array_equal(simulation.observed[0], simulation.observed[1])
    assert result["retained_fit_selected_retained_oe"] == 1.0
    assert result["retained_fit_full_retained_oe"] == 0.5
    assert result["reconstructed_fit_selected_reconstructed_oe"] == 1.0
    assert result["reconstructed_fit_selected_retained_oe"] == 2.0
    assert result["reconstructed_fit_full_retained_oe"] == 1.0
