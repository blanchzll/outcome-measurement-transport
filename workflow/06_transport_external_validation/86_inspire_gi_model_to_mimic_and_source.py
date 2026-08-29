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
# # Public-development clinical bridge to the five-centre cohort
#
# A prespecified six-variable INSPIRE gastric/colorectal model is frozen and
# applied unchanged to MIMIC-IV and the authoritative 4014-operation source
# cohort. This is an endpoint-transport validation because the public outcome is
# creatinine-only whereas the source outcome is expert-adjudicated full KDIGO.

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
INSPIRE_ROOT = Path(str(_release_path('inspire')))
MIMIC_ROOT = Path(str(_release_path('mimic')))
SOURCE_PATH = BASE / 'secure_source/inter3_deidentified_4014_dated.csv'
sys.path.insert(0, str(BASE))

from analysis import harmonize_gender_values  # noqa: E402
from common_feature_external_validation import prepare_development, prepare_external  # noqa: E402
from loco_analysis import SUMMARY_METRICS, probability_metrics  # noqa: E402
from mimic_external_validation import build_mimic_cohort  # noqa: E402

SEED = 20260829
N_BOOTSTRAP = 1000
CONTINUOUS = ['Age', 'LogPreopCr', 'PreopHb']
CATEGORICAL = ['Gender', 'Diabetes', 'Gastrocolorectal']
PREDICTORS = CONTINUOUS + CATEGORICAL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_binary(series: pd.Series) -> pd.Series:
    text = series.astype('string').str.strip().str.casefold()
    result = pd.Series(pd.NA, index=series.index, dtype='string')
    result.loc[text.isin({'0', '0.0', 'no', '否'})] = '0'
    result.loc[text.isin({'1', '1.0', 'yes', '是'})] = '1'
    return result


def canonical_site(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    result = pd.Series(pd.NA, index=series.index, dtype='string')
    result.loc[numeric.eq(1)] = '1'
    result.loc[numeric.eq(2)] = '2'
    return result


def prepare_inspire_development() -> pd.DataFrame:
    source = pd.read_csv(ROOT / 'secure_work/INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz', low_memory=False)
    frame = pd.DataFrame({
        'PostopAKI': pd.to_numeric(source['full168_creatinine_aki'], errors='coerce'),
        'Age': pd.to_numeric(source['Age'], errors='coerce'),
        'PreopCr': pd.to_numeric(source['PreopCr'], errors='coerce'),
        'PreopHb': pd.to_numeric(source['PreopHb'], errors='coerce'),
        'Gender': harmonize_gender_values(source['Gender']),
        'Diabetes': canonical_binary(source['Diabetes']),
        'Gastrocolorectal': canonical_site(source['Gastrocolorectal']),
        'dense_reference': source['dense_reference'].eq(True),
    })
    frame['LogPreopCr'] = np.log(frame['PreopCr'].where(frame['PreopCr'] > 0))
    frame = frame.loc[
        frame['dense_reference'] & frame['PostopAKI'].isin([0, 1]) & frame['Gastrocolorectal'].notna()
    ].copy()
    frame['PostopAKI'] = frame['PostopAKI'].astype(int)
    if frame['PostopAKI'].nunique() != 2:
        raise RuntimeError('INSPIRE development endpoint has fewer than two classes')
    return frame[['PostopAKI'] + PREDICTORS].reset_index(drop=True)


def make_model() -> Pipeline:
    preprocessor = ColumnTransformer([
        ('continuous', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()),
        ]), CONTINUOUS),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(C=0.25, solver='lbfgs', max_iter=5000, random_state=SEED)),
    ])


def bootstrap_intervals(y, p, strata=None, seed: int = SEED) -> dict[str, tuple[float, float]]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    strata_array = None if strata is None else np.asarray(strata)
    if strata_array is not None:
        groups = {label: np.flatnonzero(strata_array == label) for label in pd.unique(strata_array)}

    def one_draw(child_seed):
        rng = np.random.default_rng(child_seed)
        if strata_array is None:
            take = rng.choice(len(y), len(y), replace=True)
        else:
            take = np.concatenate([rng.choice(index, len(index), replace=True) for index in groups.values()])
        if np.unique(y[take]).size < 2:
            return None
        try:
            return probability_metrics(y[take], p[take])
        except (ValueError, FloatingPointError):
            return None

    draws = Parallel(n_jobs=8, prefer='processes')(
        delayed(one_draw)(child) for child in np.random.SeedSequence(seed).spawn(N_BOOTSTRAP)
    )
    samples = {name: [] for name in SUMMARY_METRICS}
    for draw in (item for item in draws if item is not None):
        for name in SUMMARY_METRICS:
            value = float(draw[name])
            if math.isfinite(value):
                samples[name].append(value)
    return {
        name: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) if values else (math.nan, math.nan)
        for name, values in samples.items()
    }


def metric_row(label: str, development: pd.DataFrame, validation: pd.DataFrame, p: np.ndarray, strata=None) -> dict[str, object]:
    point = probability_metrics(validation['PostopAKI'].to_numpy(dtype=int), p)
    intervals = bootstrap_intervals(
        validation['PostopAKI'].to_numpy(dtype=int), p, strata=strata,
        seed=SEED + (0 if label == 'MIMIC-IV' else 100),
    )
    row = {
        'transport_direction': f'INSPIRE_to_{label}', 'training_database': 'INSPIRE',
        'validation_database': label, 'model_specification': 'fixed_C6_ridge',
        'predictors': '|'.join(PREDICTORS), 'n_train': int(len(development)),
        'events_train': int(development['PostopAKI'].sum()), 'n_validation': int(len(validation)),
        'events_validation': int(validation['PostopAKI'].sum()),
        'prediction_landmark': 'surgery end; MIMIC uses first qualifying procedure calendar date',
        'development_endpoint': '0-168 h creatinine-defined AKI in INSPIRE dense-reference cohort',
        'validation_endpoint': (
            '0-168 h creatinine-defined AKI' if label == 'MIMIC-IV'
            else 'site-adjudicated KDIGO 2012 using creatinine, urine output and RRT'
        ),
        'endpoint_equivalent_to_development': label == 'MIMIC-IV',
        'local_recalibration': False,
        'bootstrap_unit': 'analytic record' if label == 'MIMIC-IV' else 'patient stratified by centre',
        **point,
    }
    for name, (lower, upper) in intervals.items():
        row[f'{name}_ci_lower'] = lower
        row[f'{name}_ci_upper'] = upper
    return row


def calibration_rows(label: str, y: np.ndarray, p: np.ndarray) -> list[dict[str, object]]:
    bins = pd.qcut(pd.Series(p), q=10, duplicates='drop')
    frame = pd.DataFrame({'outcome': y, 'probability': p, 'bin': bins})
    rows = []
    for index, (_, group) in enumerate(frame.groupby('bin', observed=True), start=1):
        rows.append({
            'validation_database': label, 'bin': index, 'n': int(len(group)),
            'events': int(group['outcome'].sum()),
            'mean_predicted_probability': float(group['probability'].mean()),
            'observed_event_fraction': float(group['outcome'].mean()),
        })
    return rows


def decision_rows(label: str, y: np.ndarray, p: np.ndarray) -> list[dict[str, object]]:
    prevalence = float(np.mean(y))
    rows = []
    for threshold in np.round(np.arange(0.02, 0.201, 0.01), 2):
        selected = p >= threshold
        cost = threshold / (1 - threshold)
        rows.append({
            'validation_database': label, 'threshold': float(threshold),
            'model_net_benefit': float(np.mean(selected & (y == 1)) - np.mean(selected & (y == 0)) * cost),
            'treat_all_net_benefit': prevalence - (1 - prevalence) * cost,
            'treat_none_net_benefit': 0.0, 'selected_fraction': float(selected.mean()),
        })
    return rows


def main() -> None:
    development = prepare_inspire_development()
    source_validation = prepare_development(SOURCE_PATH)
    source_validation = source_validation[['Center', 'PostopAKI'] + PREDICTORS].copy()

    raw_mimic = build_mimic_cohort(MIMIC_ROOT, preop_window_hours=24.0)
    mimic_validation = prepare_external(raw_mimic)

    model = make_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(
        model, development[PREDICTORS], development['PostopAKI'], cv=cv,
        method='predict_proba', n_jobs=5,
    )[:, 1]
    internal = probability_metrics(development['PostopAKI'].to_numpy(dtype=int), oof)
    model.fit(development[PREDICTORS], development['PostopAKI'])
    model_path = STAGE / 'secure_work/INSPIRE_GI_C6_RIDGE_LOCKED_MODEL.joblib'
    joblib.dump(model, model_path)

    p_mimic = model.predict_proba(mimic_validation[PREDICTORS])[:, 1]
    p_source = model.predict_proba(source_validation[PREDICTORS])[:, 1]
    rows = [
        {
            'transport_direction': 'INSPIRE_internal_5fold', 'training_database': 'INSPIRE',
            'validation_database': 'INSPIRE', 'model_specification': 'fixed_C6_ridge',
            'predictors': '|'.join(PREDICTORS), 'n_train': len(development),
            'events_train': int(development['PostopAKI'].sum()), 'n_validation': len(development),
            'events_validation': int(development['PostopAKI'].sum()),
            'prediction_landmark': 'surgery end',
            'development_endpoint': '0-168 h creatinine-defined AKI in dense-reference cohort',
            'validation_endpoint': 'same internal endpoint', 'endpoint_equivalent_to_development': True,
            'local_recalibration': False, 'bootstrap_unit': 'not_applied_to_internal_5fold', **internal,
        },
        metric_row('MIMIC-IV', development, mimic_validation, p_mimic),
        metric_row('Five-centre-source', development, source_validation, p_source, strata=source_validation['Center']),
    ]
    pd.DataFrame(rows).to_csv(STAGE / 'tables/Table_public_model_to_source_clinical_bridge.csv', index=False)

    calibration = calibration_rows('MIMIC-IV', mimic_validation['PostopAKI'].to_numpy(dtype=int), p_mimic)
    calibration += calibration_rows('Five-centre-source', source_validation['PostopAKI'].to_numpy(dtype=int), p_source)
    pd.DataFrame(calibration).to_csv(STAGE / 'tables/Table_public_model_to_source_calibration_curve.csv', index=False)
    decisions = decision_rows('MIMIC-IV', mimic_validation['PostopAKI'].to_numpy(dtype=int), p_mimic)
    decisions += decision_rows('Five-centre-source', source_validation['PostopAKI'].to_numpy(dtype=int), p_source)
    pd.DataFrame(decisions).to_csv(STAGE / 'tables/Table_public_model_to_source_decision_curve.csv', index=False)

    centre_rows = []
    for centre, indices in source_validation.groupby('Center').groups.items():
        take = np.asarray(list(indices), dtype=int)
        y = source_validation.loc[take, 'PostopAKI'].to_numpy(dtype=int)
        p = p_source[take]
        row = {
            'centre': int(centre), 'n': int(len(take)), 'events': int(y.sum()),
            'event_rate': float(y.mean()), 'mean_predicted_probability': float(p.mean()),
            'oe_ratio': float(y.sum() / p.sum()) if p.sum() > 0 else np.nan,
            'roc_auc': np.nan, 'calibration_slope': np.nan,
            'calibration_inference_status': 'descriptive_only_fewer_than_20_events',
        }
        if np.unique(y).size == 2:
            point = probability_metrics(y, p)
            row['roc_auc'] = point['roc_auc']
            if y.sum() >= 20:
                row['calibration_slope'] = point['calibration_slope']
                row['calibration_inference_status'] = 'estimable_with_caution'
        centre_rows.append(row)
    pd.DataFrame(centre_rows).to_csv(STAGE / 'tables/Table_public_model_to_source_by_centre.csv', index=False)

    secure = pd.DataFrame({
        'centre': source_validation['Center'].to_numpy(dtype=int),
        'outcome': source_validation['PostopAKI'].to_numpy(dtype=int),
        'predicted_probability': p_source,
    })
    secure.to_csv(STAGE / 'secure_work/PUBLIC_MODEL_TO_SOURCE_PREDICTIONS_SECURE.csv.gz', index=False, compression='gzip')

    audit = {
        'analysis': 'public-development model transported unchanged to MIMIC-IV and the five-centre source cohort',
        'development': {'database': 'INSPIRE 1.4.2', 'n': len(development), 'events': int(development['PostopAKI'].sum())},
        'validation': {
            'MIMIC-IV': {'n': len(mimic_validation), 'events': int(mimic_validation['PostopAKI'].sum())},
            'five-centre-source': {'n': len(source_validation), 'events': int(source_validation['PostopAKI'].sum())},
        },
        'model': 'ridge logistic regression with fixed C=0.25',
        'predictors': PREDICTORS,
        'training_only_preprocessing': True,
        'model_file': str(model_path.relative_to(STAGE)), 'model_sha256': sha256(model_path),
        'external_outcomes_used_for_model_selection_or_tuning': False,
        'local_recalibration_before_primary_evaluation': False,
        'interpretation': 'MIMIC is an operational same-creatinine-endpoint transport; the five-centre test is an endpoint-transport clinical bridge, not strict same-endpoint external validation.',
        'limits': [
            'MIMIC uses a procedure calendar-date landmark rather than exact operation end.',
            'The five-centre endpoint is expert full KDIGO and therefore intentionally richer than the public creatinine-only endpoint.',
            'The INSPIRE development sample is conditioned on dense postoperative creatinine measurement.',
        ],
        'patient_level_outputs_delivered': False,
        'input_sha256': {
            'source_4014': sha256(SOURCE_PATH),
            'inspire_analysis': sha256(ROOT / 'secure_work/INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz'),
        },
        'seed': SEED,
    }
    (STAGE / 'outputs/PUBLIC_MODEL_TO_SOURCE_CLINICAL_BRIDGE_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(pd.DataFrame(rows).loc[:, [
        'transport_direction', 'n_validation', 'events_validation', 'roc_auc', 'oe_ratio', 'calibration_slope'
    ]].to_string(index=False))


if __name__ == '__main__':
    main()
