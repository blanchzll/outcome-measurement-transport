import numpy as np
import pandas as pd

from ascertainment_stress import delete_and_reconstruct, weighted_metrics


def synthetic_data():
    patient = pd.DataFrame({
        "reference_id": ["A", "B", "C"],
        "baseline_creatinine": [1.0, 1.0, 2.0],
        "y_full": [1, 0, 1], "risk": [.7, .1, .6],
        "age_z": [0., -1., 1.], "sex_z": [1., -1., 1.], "stratum_z": [0., 1., -1.],
    })
    serial = pd.DataFrame({
        "reference_id": ["A","A","B","B","C","C"],
        "hour": [24,72,24,72,24,72],
        "creatinine": [1.35,1.2,1.0,1.1,2.1,3.1],
    })
    return patient, serial


def test_full_retention_exactly_reconstructs_operational_endpoint():
    patient, serial = synthetic_data()
    result = delete_and_reconstruct(patient, serial, "MCAR", 1.0, "weak", np.random.default_rng(7))
    assert result.patient.R.eq(1).all()
    assert result.patient.y_reconstructed.astype(int).tolist() == patient.y_full.tolist()


def test_seed_reproducibility_and_probability_bounds():
    patient, serial = synthetic_data()
    a = delete_and_reconstruct(patient, serial, "mixed_MNAR", .55, "strong", np.random.default_rng(9))
    b = delete_and_reconstruct(patient, serial, "mixed_MNAR", .55, "strong", np.random.default_rng(9))
    assert a.kept_serial.reference_id.tolist() == b.kept_serial.reference_id.tolist()
    assert a.patient.q_observed.between(0, 1).all()


def test_weighted_metrics_perfect_discrimination():
    metrics = weighted_metrics([0,0,1,1], [.1,.2,.8,.9], [1,2,1,2])
    assert metrics["auc"] == 1.0
    assert 0 < metrics["ess"] <= 4
