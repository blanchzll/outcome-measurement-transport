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
# # Locked INSPIRE model transported unchanged to MIMIC-IV and eICU
#
# The prediction landmark, operational endpoint, variables, preprocessing and
# ridge penalty are fixed before either validation database is scored.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(str(_release_path('source')))
ROOT = Path(str(_release_path('analysis')))
STAGE = ROOT / 'cross_database_external_validation'
sys.path.insert(0, str(BASE))
from loco_analysis import SUMMARY_METRICS, probability_metrics  # noqa: E402

SEED = 20260829
N_BOOTSTRAP = 1000
MINIMAL_CONTINUOUS = ['age', 'log_baseline_creatinine']
EXTENDED_CONTINUOUS = MINIMAL_CONTINUOUS + [
    'baseline_albumin', 'baseline_bun', 'baseline_glucose', 'baseline_sodium',
    'baseline_potassium', 'baseline_hemoglobin', 'baseline_wbc', 'baseline_platelet',
]
CATEGORICAL = ['sex']
MODEL_SPECS = {'minimal': MINIMAL_CONTINUOUS, 'extended_common': EXTENDED_CONTINUOUS}
UNITS = {
    'age': 'years', 'log_baseline_creatinine': 'natural log mg/dL',
    'baseline_albumin': 'g/dL', 'baseline_bun': 'mg/dL',
    'baseline_glucose': 'mg/dL', 'baseline_sodium': 'mmol/L',
    'baseline_potassium': 'mmol/L', 'baseline_hemoglobin': 'g/dL',
    'baseline_wbc': 'K/uL', 'baseline_platelet': 'K/uL', 'sex': 'category',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_harmonized(database: str) -> pd.DataFrame:
    if database == 'INSPIRE':
        source = pd.read_csv(STAGE / 'secure_work/INSPIRE_SURGICAL_ICU_REFERENCE_SECURE.csv.gz', low_memory=False)
        result = pd.DataFrame({
            'record_id': source['reference_id'].astype('string'),
            'hospital': 'INSPIRE',
            'cluster': source['subject_id'].astype('string'),
            'age': pd.to_numeric(source['age'], errors='coerce'),
            'sex': source['gender'].astype('string').str.strip().str.title(),
            'outcome': pd.to_numeric(source['Y_longitudinal'], errors='coerce'),
            'dense': pd.to_numeric(source['R_dense'], errors='coerce'),
        })
    elif database == 'MIMIC-IV':
        source = pd.read_csv(ROOT / 'secure_work/MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz', low_memory=False)
        result = pd.DataFrame({
            'record_id': source['reference_id'].astype('string'),
            'hospital': 'MIMIC-IV',
            'cluster': source['subject_id'].astype('string'),
            'age': pd.to_numeric(source['age'], errors='coerce'),
            'sex': source['gender'].astype('string').str.upper().map({'M': 'Male', 'F': 'Female'}),
            'outcome': pd.to_numeric(source['Y_longitudinal'], errors='coerce'),
            'dense': pd.to_numeric(source['R_dense'], errors='coerce'),
        })
    elif database == 'eICU':
        source = pd.read_csv(ROOT / 'eicu/secure/EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz', low_memory=False)
        labs = pd.read_csv(ROOT / 'eicu/secure/EICU_COMMON_PREDICTORS_SECURE.csv.gz', low_memory=False)
        source = source.merge(labs.drop(columns='patientunitstayid'), on='reference_id', how='left', validate='one_to_one')
        result = pd.DataFrame({
            'record_id': source['reference_id'].astype('string'),
            'hospital': source['hospitalid'].astype('string'),
            'cluster': source['hospitalid'].astype('string'),
            'age': pd.to_numeric(source['age_num'], errors='coerce'),
            'sex': source['gender'].astype('string').str.strip().str.title(),
            'outcome': pd.to_numeric(source['Y_longitudinal'], errors='coerce'),
            'dense': pd.to_numeric(source['R_dense'], errors='coerce'),
        })
    else:
        raise ValueError(database)

    baseline = pd.to_numeric(source['baseline_creatinine'], errors='coerce')
    result['log_baseline_creatinine'] = np.log(baseline.where(baseline > 0))
    for column in EXTENDED_CONTINUOUS[2:]:
        result[column] = pd.to_numeric(source[column], errors='coerce')
    result['database'] = database
    result = result.loc[result['dense'].eq(1) & result['outcome'].notna()].copy()
    result['outcome'] = result['outcome'].astype(int)
    if result['record_id'].duplicated().any() or result['outcome'].nunique() != 2:
        raise RuntimeError(f'{database} does not satisfy the locked validation contract')
    return result.reset_index(drop=True)


def make_model(continuous: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer([
        ('continuous', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()),
        ]), continuous),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(C=0.25, solver='lbfgs', max_iter=5000, random_state=SEED)),
    ])


def bootstrap_intervals(y, p, clusters, cluster_bootstrap: bool, seed: int) -> dict[str, tuple[float, float]]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    clusters = np.asarray(clusters)
    unique = pd.unique(clusters)
    index = {label: np.flatnonzero(clusters == label) for label in unique}

    def one_draw(child_seed):
        rng = np.random.default_rng(child_seed)
        if cluster_bootstrap:
            drawn = rng.choice(unique, len(unique), replace=True)
            take = np.concatenate([index[label] for label in drawn])
        else:
            take = rng.choice(len(y), len(y), replace=True)
        if np.unique(y[take]).size < 2:
            return None
        try:
            return probability_metrics(y[take], p[take])
        except (ValueError, FloatingPointError):
            return None

    child_seeds = np.random.SeedSequence(seed).spawn(N_BOOTSTRAP)
    draws = Parallel(n_jobs=8, prefer='processes')(delayed(one_draw)(child) for child in child_seeds)
    samples = {name: [] for name in SUMMARY_METRICS}
    for draw in (item for item in draws if item is not None):
        for name in SUMMARY_METRICS:
            value = float(draw[name])
            if math.isfinite(value):
                samples[name].append(value)
    return {
        name: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ) if values else (math.nan, math.nan)
        for name, values in samples.items()
    }


def calibration_rows(database: str, specification: str, y: np.ndarray, p: np.ndarray) -> list[dict[str, object]]:
    bins = pd.qcut(pd.Series(p), q=10, duplicates='drop')
    frame = pd.DataFrame({'outcome': y, 'probability': p, 'bin': bins})
    rows = []
    for index, (_, group) in enumerate(frame.groupby('bin', observed=True), start=1):
        rows.append({
            'validation_database': database, 'model_specification': specification,
            'bin': index, 'n': int(len(group)), 'events': int(group['outcome'].sum()),
            'mean_predicted_probability': float(group['probability'].mean()),
            'observed_event_fraction': float(group['outcome'].mean()),
        })
    return rows


def decision_rows(database: str, specification: str, y: np.ndarray, p: np.ndarray) -> list[dict[str, object]]:
    rows = []
    prevalence = float(np.mean(y))
    for threshold in np.round(np.arange(0.02, 0.301, 0.01), 2):
        predicted = p >= threshold
        tp = float(np.mean(predicted & (y == 1)))
        fp = float(np.mean(predicted & (y == 0)))
        cost = threshold / (1 - threshold)
        rows.append({
            'validation_database': database, 'model_specification': specification,
            'threshold': float(threshold), 'model_net_benefit': tp - fp * cost,
            'treat_all_net_benefit': prevalence - (1 - prevalence) * cost,
            'treat_none_net_benefit': 0.0,
            'selected_fraction': float(predicted.mean()),
        })
    return rows


def main() -> None:
    datasets = {name: load_harmonized(name) for name in ['INSPIRE', 'MIMIC-IV', 'eICU']}
    development = datasets['INSPIRE']
    availability = []
    for database, frame in datasets.items():
        for variable in EXTENDED_CONTINUOUS + CATEGORICAL:
            availability.append({
                'database': database, 'cohort': 'dense_reference', 'predictor': variable,
                'unit': UNITS[variable], 'n': int(len(frame)),
                'n_observed': int(frame[variable].notna().sum()),
                'missing_fraction': float(frame[variable].isna().mean()),
                'selection_status': 'prespecified_common_predictor',
            })
    pd.DataFrame(availability).to_csv(STAGE / 'tables/Table_inspire_locked_predictor_availability.csv', index=False)

    summary_rows = []
    calibration = []
    decisions = []
    predictions = []
    model_lock = {}
    for model_index, (specification, continuous) in enumerate(MODEL_SPECS.items()):
        predictors = continuous + CATEGORICAL
        model = make_model(continuous)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = cross_val_predict(model, development[predictors], development['outcome'], cv=cv, method='predict_proba', n_jobs=5)[:, 1]
        internal = probability_metrics(development['outcome'].to_numpy(), oof)
        summary_rows.append({
            'transport_direction': 'INSPIRE_internal_5fold', 'training_database': 'INSPIRE',
            'validation_database': 'INSPIRE', 'model_specification': specification,
            'predictors': '|'.join(predictors), 'n_train': len(development),
            'events_train': int(development['outcome'].sum()), 'n_validation': len(development),
            'events_validation': int(development['outcome'].sum()), 'prediction_landmark': 'ICU admission',
            'endpoint': '0-168 h creatinine-defined AKI among dense-reference stays',
            'local_recalibration': False, 'bootstrap_unit': 'not_applied_to_internal_5fold', **internal,
        })
        model.fit(development[predictors], development['outcome'])
        model_path = STAGE / f'secure_work/INSPIRE_{specification}_LOCKED_MODEL.joblib'
        joblib.dump(model, model_path)
        model_lock[specification] = {
            'predictors': predictors, 'model_file': str(model_path.relative_to(STAGE)),
            'model_sha256': sha256(model_path), 'ridge_C': 0.25,
            'training_only_preprocessing': True,
        }

        for validation_index, validation_name in enumerate(['MIMIC-IV', 'eICU']):
            validation = datasets[validation_name]
            probability = model.predict_proba(validation[predictors])[:, 1]
            point = probability_metrics(validation['outcome'].to_numpy(), probability)
            cluster_bootstrap = validation_name == 'eICU'
            intervals = bootstrap_intervals(
                validation['outcome'].to_numpy(), probability, validation['cluster'].to_numpy(),
                cluster_bootstrap, SEED + model_index * 100 + validation_index,
            )
            row = {
                'transport_direction': f'INSPIRE_to_{validation_name}',
                'training_database': 'INSPIRE', 'validation_database': validation_name,
                'model_specification': specification, 'predictors': '|'.join(predictors),
                'n_train': len(development), 'events_train': int(development['outcome'].sum()),
                'n_validation': len(validation), 'events_validation': int(validation['outcome'].sum()),
                'prediction_landmark': 'ICU admission',
                'endpoint': '0-168 h creatinine-defined AKI among dense-reference stays',
                'local_recalibration': False,
                'bootstrap_unit': 'hospital' if cluster_bootstrap else 'patient', **point,
            }
            for metric, (lower, upper) in intervals.items():
                row[f'{metric}_ci_lower'] = lower
                row[f'{metric}_ci_upper'] = upper
            summary_rows.append(row)
            calibration.extend(calibration_rows(validation_name, specification, validation['outcome'].to_numpy(), probability))
            decisions.extend(decision_rows(validation_name, specification, validation['outcome'].to_numpy(), probability))
            secure = validation[['database', 'record_id', 'hospital', 'outcome']].copy()
            secure.insert(0, 'transport_direction', f'INSPIRE_to_{validation_name}')
            secure.insert(1, 'model_specification', specification)
            secure['predicted_probability'] = probability
            predictions.append(secure)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(STAGE / 'tables/Table_inspire_locked_external_validation.csv', index=False)
    pd.DataFrame(calibration).to_csv(STAGE / 'tables/Table_inspire_locked_external_calibration_curve.csv', index=False)
    pd.DataFrame(decisions).to_csv(STAGE / 'tables/Table_inspire_locked_external_decision_curve.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        STAGE / 'secure_work/INSPIRE_LOCKED_EXTERNAL_PREDICTIONS_SECURE.csv.gz', index=False, compression='gzip'
    )

    audit = {
        'analysis': 'true external validation of a frozen INSPIRE surgical-ICU public-data model',
        'development_database': 'INSPIRE 1.4.2',
        'external_validation_databases': ['MIMIC-IV 3.1', 'eICU 2.0'],
        'datasets': {name: {'n': len(frame), 'events': int(frame['outcome'].sum())} for name, frame in datasets.items()},
        'prediction_landmark': 'ICU admission',
        'endpoint': '0-168 h creatinine-defined KDIGO AKI in dense-reference surgical-ICU cohorts',
        'model_lock': model_lock,
        'preprocessing': 'INSPIRE-only median imputation with missingness indicators, scaling and categorical encoding',
        'external_outcomes_used_for_selection_or_tuning': False,
        'local_recalibration_before_primary_evaluation': False,
        'bootstrap': {'replicates': N_BOOTSTRAP, 'MIMIC-IV_unit': 'patient', 'eICU_unit': 'hospital'},
        'limits': [
            'The estimand is conditional on dense post-ICU creatinine measurement.',
            'The endpoint is operational creatinine-only AKI, not full clinician-adjudicated KDIGO.',
            'This validates the public surgical-ICU model, not the five-centre surgery-end model.',
        ],
        'patient_level_outputs_delivered': False,
        'seed': SEED,
        'software': {'python': sys.version, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
    }
    (STAGE / 'outputs/INSPIRE_LOCKED_EXTERNAL_VALIDATION_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(summary.loc[summary['validation_database'].ne('INSPIRE'), [
        'transport_direction', 'model_specification', 'n_validation', 'events_validation',
        'roc_auc', 'oe_ratio', 'calibration_slope',
    ]].to_string(index=False))


if __name__ == '__main__':
    main()
