import numpy as np
import pandas as pd

from ascertainment_stress import delete_and_reconstruct


def patient_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_id": [1],
            "baseline_creatinine": [1.0],
            "y_full": [1],
            "risk": [0.10],
            "age_z": [0.0],
            "sex_z": [0.0],
            "stratum_z": [0.0],
        }
    )


def simulate(hours: list[float]) -> pd.DataFrame:
    serial = pd.DataFrame(
        {
            "reference_id": 1,
            "hour": hours,
            "creatinine": [1.0] * (len(hours) - 1) + [1.6],
        }
    )
    return delete_and_reconstruct(
        patient_frame(), serial, "MCAR", 0.999999, "weak", np.random.default_rng(17)
    ).patient


def test_second_window_ends_at_96_hours() -> None:
    assert simulate([24.0, 96.0, 168.0]).R.iloc[0] == 1
    assert simulate([24.0, 96.0001, 168.0]).R.iloc[0] == 0


def test_168_hour_measurement_can_define_outcome_but_not_observability() -> None:
    result = simulate([24.0, 72.0, 168.0])
    assert result.R.iloc[0] == 1
    assert result.y_reconstructed.iloc[0] == 1


def test_observed_history_generator_targets_retention() -> None:
    patient = pd.concat(
        [patient_frame().assign(reference_id=i) for i in range(1, 101)], ignore_index=True
    )
    serial = pd.DataFrame(
        [
            (i, hour, 1.0 + 0.01 * position)
            for i in range(1, 101)
            for position, hour in enumerate((12, 36, 60, 84, 120))
        ],
        columns=["reference_id", "hour", "creatinine"],
    )
    result = delete_and_reconstruct(
        patient, serial, "history_MAR", 0.55, "strong", np.random.default_rng(29)
    )
    assert abs(result.mean_measurement_retention - 0.55) < 0.06
