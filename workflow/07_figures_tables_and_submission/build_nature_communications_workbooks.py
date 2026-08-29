# %% [markdown]
# # Nature Communications editable tables and source-data workbooks
# Builds release-safe aggregate Excel files from frozen CSV outputs.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(str(_release_path('analysis')))
NC = ROOT / 'nature_communications'
MAIN_SOURCE = ROOT / 'delivery' / 'main' / 'tables'
SUPP_SOURCE = ROOT / 'delivery' / 'supplement' / 'tables'
MAIN_OUT = NC / 'tables' / 'main'
SUPP_OUT = NC / 'tables' / 'supplement'
SOURCE_OUT = NC / 'source_data'
for directory in (MAIN_OUT, SUPP_OUT, SOURCE_OUT):
    directory.mkdir(parents=True, exist_ok=True)

MAIN_TABLES = [
    'Table1_source_model_results.csv',
    'Table2_mixed_MNAR_results.csv',
    'Table3_apparent_vs_reference_recalibration.csv',
    'Table4_reference_sample_design.csv',
]

CROSSDB_SOURCE = ROOT / 'cross_database_external_validation' / 'tables'
CROSSDB_SUPP_TABLES = [
    'Table_inspire_surgical_icu_reference_flow.csv',
    'Table_inspire_surgical_icu_predictor_availability.csv',
    'Table_inspire_locked_external_validation.csv',
    'Table_inspire_locked_external_calibration_curve.csv',
    'Table_inspire_locked_external_decision_curve.csv',
    'Table_inspire_locked_predictor_availability.csv',
    'Table_public_model_to_source_clinical_bridge.csv',
    'Table_public_model_to_source_calibration_curve.csv',
    'Table_public_model_to_source_decision_curve.csv',
    'Table_public_model_to_source_by_centre.csv',
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_sheet(raw: str, used: set[str]) -> str:
    name = re.sub(r'[^A-Za-z0-9_]+', '_', raw).strip('_')[:31] or 'Sheet'
    base, i = name, 1
    while name in used:
        suffix = f'_{i}'
        name = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def style_workbook(path: Path) -> None:
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = 'A2'
        for cell in ws[1]:
            cell.font = Font(name='Arial', size=10, bold=True, color='000000')
            cell.fill = PatternFill(fill_type='solid', fgColor='E7E7E7')
            cell.alignment = Alignment(wrap_text=True, vertical='top')
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name='Arial', size=9, color='000000')
                cell.alignment = Alignment(wrap_text=True, vertical='top')
        for idx, column in enumerate(ws.columns, start=1):
            values = [str(c.value) if c.value is not None else '' for c in list(column)[:200]]
            width = min(45, max(10, max((len(v) for v in values), default=10) + 2))
            ws.column_dimensions[get_column_letter(idx)].width = width
    wb.save(path)


def workbook_from_csvs(csvs: list[Path], output: Path, readme_rows: list[list[str]]) -> dict:
    used: set[str] = set()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(readme_rows[1:], columns=readme_rows[0]).to_excel(writer, sheet_name='README', index=False)
        used.add('README')
        manifest = []
        for csv in csvs:
            frame = pd.read_csv(csv)
            sheet = safe_sheet(csv.stem, used)
            frame.to_excel(writer, sheet_name=sheet, index=False)
            manifest.append({'source': str(csv), 'sheet': sheet, 'rows': len(frame), 'columns': len(frame.columns), 'sha256': sha256(csv)})
    style_workbook(output)
    return {'output': str(output), 'sha256': sha256(output), 'sheets': manifest}


# %% Copy frozen aggregate tables
main_csvs = []
for name in MAIN_TABLES:
    source = MAIN_SOURCE / name
    if not source.exists():
        raise FileNotFoundError(source)
    target = MAIN_OUT / name
    shutil.copy2(source, target)
    main_csvs.append(target)

compatibility = NC / 'tables' / 'Supplementary_Table_1_method_identification_conditions.csv'
if compatibility.exists():
    target = SUPP_OUT / compatibility.name
    shutil.copy2(compatibility, target)

supp_csvs = []
for source in sorted(SUPP_SOURCE.glob('*.csv')):
    target = SUPP_OUT / source.name
    shutil.copy2(source, target)
    supp_csvs.append(target)
if compatibility.exists():
    supp_csvs.insert(0, SUPP_OUT / compatibility.name)
for name in CROSSDB_SUPP_TABLES:
    source = CROSSDB_SOURCE / name
    if not source.exists():
        raise FileNotFoundError(source)
    target = SUPP_OUT / name
    shutil.copy2(source, target)
    supp_csvs.append(target)

# %% Main and supplementary table workbooks
main_audit = workbook_from_csvs(
    main_csvs,
    NC / 'tables' / 'Main_Tables.xlsx',
    [['Field', 'Value'], ['Scope', 'Four editable main tables'], ['Data level', 'Aggregate only; no patient-level records'], ['Formatting', 'Black text, grey header, editable cells']],
)

supp_audit = workbook_from_csvs(
    supp_csvs,
    NC / 'tables' / 'Supplementary_Tables.xlsx',
    [['Field', 'Value'], ['Scope', 'Complete aggregate supplementary table set'], ['Data level', 'Aggregate only; no patient-level records'], ['Primary compatibility table', 'Supplementary Table 1 maps methods to estimands and identification conditions']],
)

# %% Nature Source Data workbook from all submitted panel CSVs
figure_csvs = sorted((NC / 'figures').glob('Figure*/**/*_source_data.csv'))
figure_csvs += sorted((NC / 'figures').glob('SupplementaryFigure*/**/*_source_data.csv'))
# Glob patterns above can overlap; preserve one copy per resolved path.
figure_csvs = list(dict.fromkeys(p.resolve() for p in figure_csvs))
source_audit = workbook_from_csvs(
    figure_csvs,
    SOURCE_OUT / 'Source_Data.xlsx',
    [
        ['Field', 'Value'],
        ['Article title', 'Outcome measurement transport can govern calibration of clinical AI across health systems'],
        ['Scope', 'Aggregate source values for every main and supplementary figure panel'],
        ['Data level', 'Aggregate or Monte Carlo summary only; no stable patient identifiers or row-level predictions'],
        ['Endpoint note', 'Public endpoints are operational creatinine references, not biological truth or expert-adjudicated full KDIGO'],
    ],
)

audit = {
    'status': 'PASS',
    'main_tables': main_audit,
    'supplementary_tables': supp_audit,
    'source_data': source_audit,
    'main_csv_count': len(main_csvs),
    'supplementary_csv_count': len(supp_csvs),
    'figure_source_csv_count': len(figure_csvs),
}
(NC / 'qa' / 'WORKBOOK_BUILD_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({k: audit[k] for k in ['status', 'main_csv_count', 'supplementary_csv_count', 'figure_source_csv_count']}, indent=2))
