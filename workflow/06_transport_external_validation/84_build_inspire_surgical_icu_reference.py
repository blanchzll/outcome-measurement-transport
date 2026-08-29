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
# # INSPIRE surgical-ICU reference cohort at the ICU-admission landmark
#
# The cohort, prediction landmark, predictor lookback and operational creatinine
# endpoint are prespecified to match the existing MIMIC-IV and eICU testbeds.
# Patient-level output remains in secure_work.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(str(_release_path('analysis')))
STAGE = ROOT / 'cross_database_external_validation'
DATA = Path(str(_release_path('inspire')))
SEED = 20260829
MAX_ICU_DELAY_MIN = 24 * 60
LOOKBACK_MIN = 30 * 24 * 60
FOLLOWUP_MIN = 7 * 24 * 60

LABS = {
    'creatinine': ('baseline_creatinine', 'mg/dL', 0.1, 30.0),
    'albumin': ('baseline_albumin', 'g/dL', 1.0, 6.0),
    'bun': ('baseline_bun', 'mg/dL', 2.0, 200.0),
    'glucose': ('baseline_glucose', 'mg/dL', 20.0, 1000.0),
    'sodium': ('baseline_sodium', 'mmol/L', 100.0, 180.0),
    'potassium': ('baseline_potassium', 'mmol/L', 2.0, 8.0),
    'hb': ('baseline_hemoglobin', 'g/dL', 3.0, 25.0),
    'wbc': ('baseline_wbc', 'K/uL', 0.1, 100.0),
    'platelet': ('baseline_platelet', 'K/uL', 5.0, 2000.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sex(series: pd.Series) -> pd.Series:
    text = series.astype('string').str.strip().str.upper()
    return text.map({'M': 'Male', 'F': 'Female', 'MALE': 'Male', 'FEMALE': 'Female'})


def load_operations() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    operations = pd.read_csv(
        DATA / 'operations.csv.gz',
        usecols=['op_id', 'subject_id', 'hadm_id', 'age', 'sex', 'department', 'opend_time', 'icuin_time'],
        low_memory=False,
    )
    for column in ['op_id', 'subject_id', 'age', 'opend_time', 'icuin_time']:
        operations[column] = pd.to_numeric(operations[column], errors='coerce')
    flow = [{'step': 'all_operations', 'n': int(len(operations)), 'excluded_at_step': None}]
    operations['icu_delay_min'] = operations['icuin_time'] - operations['opend_time']
    eligible = operations.loc[
        operations['age'].ge(18)
        & operations['subject_id'].notna()
        & operations['opend_time'].notna()
        & operations['icuin_time'].notna()
        & operations['icu_delay_min'].between(0, MAX_ICU_DELAY_MIN, inclusive='both')
    ].copy()
    flow.append({
        'step': 'adult_operation_with_icu_admission_within_24h',
        'n': int(len(eligible)),
        'excluded_at_step': int(len(operations) - len(eligible)),
    })
    eligible = eligible.sort_values(['subject_id', 'icuin_time', 'opend_time', 'op_id'])
    before = len(eligible)
    eligible = eligible.drop_duplicates('subject_id', keep='first').reset_index(drop=True)
    flow.append({
        'step': 'first_qualifying_operation_per_patient',
        'n': int(len(eligible)),
        'excluded_at_step': int(before - len(eligible)),
    })
    eligible['reference_id'] = np.arange(len(eligible), dtype=int)
    eligible['sex_harmonized'] = canonical_sex(eligible['sex'])
    return eligible, flow


def collect_labs(subject_ids: set[int], chunksize: int = 1_000_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    wanted = set(LABS)
    for chunk in pd.read_csv(
        DATA / 'labs.csv.gz',
        usecols=['subject_id', 'chart_time', 'item_name', 'value'],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce')
        chunk['item_name'] = chunk['item_name'].astype('string').str.strip().str.lower()
        selected = chunk.loc[chunk['subject_id'].isin(subject_ids) & chunk['item_name'].isin(wanted)].copy()
        if selected.empty:
            continue
        selected['chart_time'] = pd.to_numeric(selected['chart_time'], errors='coerce')
        selected['value'] = pd.to_numeric(selected['value'], errors='coerce')
        selected = selected.dropna(subset=['subject_id', 'chart_time', 'value'])
        parts.append(selected)
    if not parts:
        raise RuntimeError('No INSPIRE laboratory records matched the surgical-ICU cohort')
    return pd.concat(parts, ignore_index=True)


def build_reference(cohort: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    linked = cohort[['reference_id', 'subject_id', 'icuin_time']].merge(
        labs, on='subject_id', how='left', validate='one_to_many'
    )
    linked['delta_min'] = linked['chart_time'] - linked['icuin_time']

    baseline = linked.loc[
        linked['delta_min'].between(-LOOKBACK_MIN, -1, inclusive='both')
    ].copy()
    valid_parts = []
    for raw, (_, _, lower, upper) in LABS.items():
        part = baseline.loc[
            baseline['item_name'].eq(raw) & baseline['value'].between(lower, upper, inclusive='both')
        ]
        if not part.empty:
            valid_parts.append(part)
    baseline = pd.concat(valid_parts, ignore_index=True) if valid_parts else baseline.iloc[0:0]
    latest = baseline.sort_values(['reference_id', 'item_name', 'delta_min']).groupby(
        ['reference_id', 'item_name'], as_index=False
    ).tail(1)
    baseline_wide = latest.pivot(index='reference_id', columns='item_name', values='value')
    baseline_wide = baseline_wide.rename(columns={raw: target for raw, (target, _, _, _) in LABS.items()})

    postoperative = linked.loc[
        linked['item_name'].eq('creatinine')
        & linked['delta_min'].gt(0)
        & linked['delta_min'].le(FOLLOWUP_MIN)
        & linked['value'].between(0.1, 30.0, inclusive='both')
    ].copy()
    base_cr = baseline_wide.get('baseline_creatinine', pd.Series(dtype=float))
    rows = []
    for reference_id in cohort['reference_id']:
        base = float(base_cr.get(reference_id, np.nan))
        serial = postoperative.loc[postoperative['reference_id'].eq(reference_id)]
        hours = serial['delta_min'].to_numpy(dtype=float) / 60.0
        values = serial['value'].to_numpy(dtype=float)
        if not np.isfinite(base) or base <= 0 or len(values) == 0:
            rows.append({
                'reference_id': int(reference_id), 'Y_longitudinal': np.nan,
                'n_creatinine_0_168h': int(len(values)), 'n_creatinine_0_48h': int(np.sum(hours <= 48)),
                'n_creatinine_48_96h': int(np.sum((hours > 48) & (hours <= 96))),
                'first_hour': float(np.min(hours)) if len(hours) else np.nan,
                'last_hour': float(np.max(hours)) if len(hours) else np.nan,
                'span_hours': float(np.max(hours) - np.min(hours)) if len(hours) else np.nan,
            })
            continue
        absolute_48h = bool(np.any((hours <= 48) & ((values - base) >= 0.3)))
        ratio_168h = bool(np.any(values >= 1.5 * base))
        rows.append({
            'reference_id': int(reference_id), 'Y_longitudinal': int(absolute_48h or ratio_168h),
            'n_creatinine_0_168h': int(len(values)), 'n_creatinine_0_48h': int(np.sum(hours <= 48)),
            'n_creatinine_48_96h': int(np.sum((hours > 48) & (hours <= 96))),
            'first_hour': float(np.min(hours)), 'last_hour': float(np.max(hours)),
            'span_hours': float(np.max(hours) - np.min(hours)),
        })
    outcome = pd.DataFrame(rows)

    result = cohort[['reference_id', 'subject_id', 'hadm_id', 'department', 'age', 'sex_harmonized', 'icu_delay_min']].copy()
    result = result.merge(baseline_wide.reset_index(), on='reference_id', how='left', validate='one_to_one')
    result = result.merge(outcome, on='reference_id', how='left', validate='one_to_one')
    result['R_longitudinal'] = result['Y_longitudinal'].notna().astype(int)
    result['R_dense'] = (
        result['Y_longitudinal'].notna()
        & result['n_creatinine_0_168h'].ge(3)
        & result['n_creatinine_0_48h'].ge(1)
        & result['n_creatinine_48_96h'].ge(1)
        & result['span_hours'].ge(72)
    ).astype(int)
    result = result.rename(columns={'age': 'age', 'sex_harmonized': 'gender'})
    return result


def main() -> None:
    for folder in ['secure_work', 'tables', 'outputs', 'logs']:
        (STAGE / folder).mkdir(parents=True, exist_ok=True, mode=0o700)
    cohort, flow = load_operations()
    labs = collect_labs(set(cohort['subject_id'].dropna().astype(int)))
    reference = build_reference(cohort, labs)
    if not reference['reference_id'].is_unique or len(reference) != len(cohort):
        raise RuntimeError('INSPIRE reference linkage is not one-to-one')
    secure_path = STAGE / 'secure_work/INSPIRE_SURGICAL_ICU_REFERENCE_SECURE.csv.gz'
    reference.to_csv(secure_path, index=False, compression='gzip')

    flow.extend([
        {
            'step': 'longitudinal_creatinine_reference_observed',
            'n': int(reference['R_longitudinal'].sum()),
            'events': int(reference.loc[reference['R_longitudinal'].eq(1), 'Y_longitudinal'].sum()),
        },
        {
            'step': 'dense_reference',
            'n': int(reference['R_dense'].sum()),
            'events': int(reference.loc[reference['R_dense'].eq(1), 'Y_longitudinal'].sum()),
        },
    ])
    pd.DataFrame(flow).to_csv(STAGE / 'tables/Table_inspire_surgical_icu_reference_flow.csv', index=False)

    availability = []
    for cohort_label, subset in [
        ('candidate', reference),
        ('dense_reference', reference.loc[reference['R_dense'].eq(1)]),
    ]:
        for raw, (column, unit, lower, upper) in LABS.items():
            availability.append({
                'cohort': cohort_label, 'predictor': column, 'source_item_name': raw,
                'unit': unit, 'valid_range': f'{lower:g}-{upper:g}', 'n': int(len(subset)),
                'n_observed': int(subset[column].notna().sum()),
                'missing_fraction': float(subset[column].isna().mean()),
            })
    pd.DataFrame(availability).to_csv(STAGE / 'tables/Table_inspire_surgical_icu_predictor_availability.csv', index=False)

    audit = {
        'analysis': 'INSPIRE adult postoperative surgical-ICU operational reference cohort',
        'prediction_landmark': 'ICU admission within 24 hours after operation end',
        'patient_unit': 'first qualifying operation per patient',
        'predictor_lookback': '-30 days to immediately before ICU admission',
        'endpoint': '0-168 h creatinine-defined KDIGO AKI',
        'dense_reference_rule': 'at least 3 measurements, at least 1 at 0-48 h, at least 1 at 48-96 h, span at least 72 h',
        'candidate_n': int(len(reference)),
        'candidate_unique_patients': int(reference['subject_id'].nunique()),
        'longitudinal_n': int(reference['R_longitudinal'].sum()),
        'longitudinal_events': int(reference.loc[reference['R_longitudinal'].eq(1), 'Y_longitudinal'].sum()),
        'dense_n': int(reference['R_dense'].sum()),
        'dense_events': int(reference.loc[reference['R_dense'].eq(1), 'Y_longitudinal'].sum()),
        'reference_is_clinician_adjudicated': False,
        'reference_is_full_kdigo': False,
        'urine_output_included': False,
        'rrt_included': False,
        'patient_level_output_delivered': False,
        'seed': SEED,
        'input_sha256': {
            'operations_csv_gz': sha256(DATA / 'operations.csv.gz'),
            'labs_csv_gz': sha256(DATA / 'labs.csv.gz'),
        },
    }
    (STAGE / 'outputs/INSPIRE_SURGICAL_ICU_REFERENCE_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
