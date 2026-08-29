from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_empirical_schedule_reconstructs_two_window_event():
    module = load("workflow/04_measurement_deletion_simulation/94_empirical_schedule_transport.py", "schedule94")
    observed, event, retained = module.reconstruct_one(
        np.array([10.0, 60.0, 120.0]),
        np.array([1.0, 1.6, 1.2]),
        baseline=1.0,
        schedule=np.array([10.0, 60.0]),
        tolerance_hours=0.0,
    )
    assert (observed, event, retained) == (1, 1.0, 2)


def test_schedule_mapping_deduplicates_target_measurements():
    module = load("workflow/04_measurement_deletion_simulation/94_empirical_schedule_transport.py", "schedule94_unique")
    hours, values = module.apply_schedule(
        np.array([10.0, 40.0]), np.array([1.0, 2.0]), np.array([9.0, 11.0]), 2.0
    )
    assert hours.tolist() == [10.0]
    assert values.tolist() == [1.0]


def test_risk_enriched_sampling_has_known_positive_probabilities():
    module = load("workflow/05_reference_sampling_and_correction/95_optimized_reference_sampling.py", "sampling95")
    frame = pd.DataFrame({"risk": np.linspace(0.001, 0.999, 200), "cluster": np.repeat(np.arange(10), 20)})
    selected, inclusion = module.choose_sample(frame, 0.10, "risk_enriched", np.random.default_rng(1))
    assert len(selected) == 30  # prespecified minimum reference size
    assert np.all(inclusion > 0)
    assert np.isclose((1 / inclusion[selected]).sum(), len(frame), rtol=0.25)


def test_harmonized_haemoglobin_baseline_precedes_endpoint_values():
    module = load("workflow/04_measurement_deletion_simulation/97_secondary_hemoglobin_endpoint.py", "hemoglobin97")
    reference = pd.DataFrame({"reference_id": ["pre", "post"], "baseline_hemoglobin": [99.0, 99.0]})
    serial = pd.DataFrame({
        "reference_id": ["pre"] * 4 + ["post"] * 4,
        "hour": [-2.0, 1.0, 50.0, 90.0, 2.0, 3.0, 55.0, 100.0],
        "hemoglobin": [12.0, 11.5, 9.5, 10.0, 13.0, 12.5, 10.5, 11.0],
    })
    harmonized, postoperative = module.harmonize_baseline(reference, serial)
    assert harmonized.set_index("reference_id").loc["pre", "baseline_hour"] == -2.0
    assert harmonized.set_index("reference_id").loc["post", "baseline_hour"] == 2.0
    endpoint = module.build_endpoint(harmonized, postoperative)
    assert endpoint.R_dense_hb.tolist() == [1, 1]
    assert endpoint.Y_hb_decline.astype(int).tolist() == [1, 1]
