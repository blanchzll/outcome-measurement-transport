import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1] / 'workflow/06_transport_external_validation'


sys.path.insert(0, str(CODE))


def load(name: str):
    path = CODE / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inspire_reference_contract_and_kdigo_binary_logic():
    module = load('84_build_inspire_surgical_icu_reference.py')
    cohort = pd.DataFrame({
        'reference_id': [0], 'subject_id': [1], 'hadm_id': [11], 'department': ['GS'],
        'age': [60], 'sex_harmonized': ['Male'], 'icu_delay_min': [10], 'icuin_time': [1000],
    })
    labs = pd.DataFrame({
        'subject_id': [1, 1, 1, 1, 1],
        'chart_time': [900, 1060, 2500, 5000, 7000],
        'item_name': ['creatinine'] * 5,
        'value': [1.0, 1.4, 1.2, 1.1, 1.0],
    })
    result = module.build_reference(cohort, labs)
    assert result.loc[0, 'Y_longitudinal'] == 1
    assert result.loc[0, 'baseline_creatinine'] == 1.0
    assert result.loc[0, 'R_dense'] == 1


def test_locked_public_model_handles_validation_missingness_without_refit():
    module = load('85_inspire_locked_public_icu_transport.py')
    frame = pd.DataFrame({
        'age': [40, 50, 60, 70], 'log_baseline_creatinine': np.log([0.8, 1.0, 1.2, 1.4]),
        'sex': ['Female', 'Male', 'Female', 'Male'],
    })
    model = module.make_model(module.MINIMAL_CONTINUOUS)
    model.fit(frame, [0, 0, 1, 1])
    validation = frame.copy()
    validation.loc[0, 'age'] = np.nan
    probability = model.predict_proba(validation)[:, 1]
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_clinical_bridge_category_contract():
    module = load('86_inspire_gi_model_to_mimic_and_source.py')
    assert module.canonical_binary(pd.Series([0, 1, 'No', 'Yes'])).tolist() == ['0', '1', '0', '1']
    assert module.canonical_site(pd.Series([1, 2, '1', '2'])).tolist() == ['1', '2', '1', '2']
    assert module.PREDICTORS == [
        'Age', 'LogPreopCr', 'PreopHb', 'Gender', 'Diabetes', 'Gastrocolorectal'
    ]
