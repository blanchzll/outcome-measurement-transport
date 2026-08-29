# %% [markdown]
# # Nature Communications figure assembly
# Combines existing vector panels without altering data or re-rendering the panel content.
# Every original panel and its source CSV remain alongside the composite PDF.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
import shutil
from pathlib import Path

import pymupdf

ROOT = Path(str(_release_path('analysis')))
INPUT = ROOT / 'figures'
OUTPUT = ROOT / 'nature_communications' / 'figures'
OUTPUT.mkdir(parents=True, exist_ok=True)

PAGE_WIDTH = 183 / 25.4 * 72  # Nature Communications double-column width
MARGIN = 10.0
GAP = 8.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def place_panel(page, source: Path, rect, label: str) -> None:
    src = pymupdf.open(source)
    page.show_pdf_page(rect, src, 0, keep_proportion=True, overlay=True)
    page.insert_text(
        pymupdf.Point(rect.x0 + 1.5, rect.y0 + 8.0),
        label,
        fontsize=8,
        fontname='helv',
        color=(0.08, 0.08, 0.08),
        overlay=True,
    )
    src.close()


def compose(name: str, specs: list[dict], height: float) -> dict:
    out_dir = OUTPUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_WIDTH, height=height)
    manifest = {'figure': name, 'page_width_pt': PAGE_WIDTH, 'page_height_pt': height, 'panels': []}
    for spec in specs:
        src = INPUT / spec['folder'] / spec['file']
        if not src.exists():
            raise FileNotFoundError(src)
        rect = pymupdf.Rect(*spec['rect'])
        place_panel(page, src, rect, spec['label'])
        target = out_dir / src.name
        shutil.copy2(src, target)
        csv = src.with_name(src.stem + '_source_data.csv')
        if csv.exists():
            shutil.copy2(csv, out_dir / csv.name)
        svg = src.with_suffix('.svg')
        if svg.exists():
            shutil.copy2(svg, out_dir / svg.name)
        manifest['panels'].append({
            'label': spec['label'],
            'input': str(src),
            'input_sha256': sha256(src),
            'rect_pt': list(spec['rect']),
        })
    composite = out_dir / f'{name}.pdf'
    doc.save(composite, garbage=4, deflate=True)
    doc.close()
    manifest['composite'] = str(composite)
    manifest['composite_sha256'] = sha256(composite)
    (out_dir / f'{name}_assembly_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def two_col_rects(rows: int, row_height: float, top: float = MARGIN):
    width = (PAGE_WIDTH - 2 * MARGIN - GAP) / 2
    result = []
    for row in range(rows):
        y0 = top + row * (row_height + GAP)
        result.append((MARGIN, y0, MARGIN + width, y0 + row_height))
        result.append((MARGIN + width + GAP, y0, PAGE_WIDTH - MARGIN, y0 + row_height))
    return result


# %% Figure 1: 2 x 2
r = two_col_rects(2, 172)
figure1 = [
    {'label': 'a', 'folder': 'Figure1_reference_observability', 'file': 'Figure1a_reference_flow.pdf', 'rect': r[0]},
    {'label': 'b', 'folder': 'Figure1_reference_observability', 'file': 'Figure1b_ipaw_balance.pdf', 'rect': r[1]},
    {'label': 'c', 'folder': 'Figure1_reference_observability', 'file': 'Figure1c_monitoring_gradient.pdf', 'rect': r[2]},
    {'label': 'd', 'folder': 'Figure1_reference_observability', 'file': 'Figure1d_estimand_schematic.pdf', 'rect': r[3]},
]

# %% Figure 2: 2 columns x 4 rows
r = two_col_rects(4, 158)
figure2 = [
    {'label': 'a', 'folder': 'Figure2_deletion_mechanisms', 'file': 'Figure2a_inspire_event_bias.pdf', 'rect': r[0]},
    {'label': 'b', 'folder': 'Figure2_deletion_mechanisms', 'file': 'Figure2a_mimic_event_bias.pdf', 'rect': r[1]},
    {'label': 'c', 'folder': 'Figure2_deletion_mechanisms', 'file': 'Figure2a_eicu_event_bias.pdf', 'rect': r[2]},
    {'label': 'd', 'folder': 'Figure2_deletion_mechanisms', 'file': 'Figure2b_inspire_method_bias.pdf', 'rect': r[3]},
    {'label': 'e', 'folder': 'Figure2_deletion_mechanisms', 'file': 'Figure2b_mimic_method_bias.pdf', 'rect': r[4]},
    {'label': 'f', 'folder': 'Figure2_deletion_mechanisms', 'file': 'Figure2b_eicu_method_bias.pdf', 'rect': r[5]},
    {'label': 'g', 'folder': 'Figure2_deletion_mechanisms', 'file': 'Figure2c_reconstruction_sensitivity.pdf', 'rect': r[6]},
    {'label': 'h', 'folder': 'Figure2_deletion_mechanisms', 'file': 'Figure2d_pure_selection_control.pdf', 'rect': r[7]},
]

# %% Figure 3: 2 columns x 4 rows
r = two_col_rects(4, 158)
figure3 = [
    {'label': 'a', 'folder': 'Figure3_correction_strategies', 'file': 'Figure3a_apparent_vs_reference_recalibration.pdf', 'rect': r[0]},
    {'label': 'b', 'folder': 'Figure3_correction_strategies', 'file': 'Figure3b_strategy_rmse.pdf', 'rect': r[1]},
    {'label': 'c', 'folder': 'Figure3_correction_strategies', 'file': 'Figure3c_cross_database_bias.pdf', 'rect': r[2]},
    {'label': 'd', 'folder': 'Figure3_correction_strategies', 'file': 'Figure3d_inspire_recalibration_fidelity.pdf', 'rect': r[3]},
    {'label': 'e', 'folder': 'Figure3_correction_strategies', 'file': 'Figure3d_mimic_recalibration_fidelity.pdf', 'rect': r[4]},
    {'label': 'f', 'folder': 'Figure3_correction_strategies', 'file': 'Figure3d_eicu_recalibration_fidelity.pdf', 'rect': r[5]},
    {'label': 'g', 'folder': 'Figure3_correction_strategies', 'file': 'Figure3e_reference_sample_design.pdf',
     'rect': ((PAGE_WIDTH - (r[6][2] - r[6][0])) / 2, r[6][1],
              (PAGE_WIDTH + (r[6][2] - r[6][0])) / 2, r[6][3])},
]

# %% Figure 4: one wide panel and two lower panels
half = (PAGE_WIDTH - 2 * MARGIN - GAP) / 2
figure4 = [
    {'label': 'a', 'folder': 'Figure7_robustness_extensions', 'file': 'Figure7b_discrimination_strength_stress.pdf',
     'rect': (MARGIN, MARGIN, PAGE_WIDTH - MARGIN, 190)},
    {'label': 'b', 'folder': 'Figure7_robustness_extensions', 'file': 'Figure7a_extended_common_transport.pdf',
     'rect': (MARGIN, 198, MARGIN + half, 370)},
    {'label': 'c', 'folder': 'Figure7_robustness_extensions', 'file': 'Figure7c_bayesian_hierarchical_calibration.pdf',
     'rect': (MARGIN + half + GAP, 198, PAGE_WIDTH - MARGIN, 370)},
]

# %% Supplementary figures
r_s1 = two_col_rects(2, 170)
supp1 = [
    {'label': 'a', 'folder': 'Figure4_stability_portability', 'file': 'Figure4a_model_stability.pdf', 'rect': r_s1[0]},
    {'label': 'b', 'folder': 'Figure4_stability_portability', 'file': 'Figure4b_perioperative_increment.pdf', 'rect': r_s1[1]},
    {'label': 'c', 'folder': 'Figure4_stability_portability', 'file': 'Figure4c_portability_frontier.pdf',
     'rect': ((PAGE_WIDTH - (r_s1[2][2] - r_s1[2][0])) / 2, r_s1[2][1],
              (PAGE_WIDTH + (r_s1[2][2] - r_s1[2][0])) / 2, r_s1[2][3])},
]

r_s2 = two_col_rects(1, 180)
supp2 = [
    {'label': 'a', 'folder': 'Figure5_clinical_audit', 'file': 'Figure5a_burden_event_capture.pdf', 'rect': r_s2[0]},
    {'label': 'b', 'folder': 'Figure5_clinical_audit', 'file': 'Figure5b_subgroup_auc.pdf', 'rect': r_s2[1]},
]

r_s3 = two_col_rects(2, 170)
supp3 = [
    {'label': 'a', 'folder': 'Figure6_eicu_replication', 'file': 'Figure6a_eicu_hospital_observability.pdf', 'rect': r_s3[0]},
    {'label': 'b', 'folder': 'Figure6_eicu_replication', 'file': 'Figure6b_eicu_endpoint_components.pdf', 'rect': r_s3[1]},
    {'label': 'c', 'folder': 'Figure6_eicu_replication', 'file': 'Figure6c_eicu_target_performance.pdf',
     'rect': ((PAGE_WIDTH - (r_s3[2][2] - r_s3[2][0])) / 2, r_s3[2][1],
              (PAGE_WIDTH + (r_s3[2][2] - r_s3[2][0])) / 2, r_s3[2][3])},
]

supp4 = [
    {'label': 'a', 'folder': 'Figure7_robustness_extensions', 'file': 'Figure7d_measurement_aware_fairness.pdf',
     'rect': (MARGIN, MARGIN, PAGE_WIDTH - MARGIN, 340)},
]

r_s5 = two_col_rects(2, 170)
supp5 = [
    {'label': 'a', 'folder': 'Figure8_source_temporal_audit', 'file': 'Figure8a_recruitment_by_year.pdf', 'rect': r_s5[0]},
    {'label': 'b', 'folder': 'Figure8_source_temporal_audit', 'file': 'Figure8b_within_centre_temporal_auc.pdf', 'rect': r_s5[1]},
    {'label': 'c', 'folder': 'Figure8_source_temporal_audit', 'file': 'Figure8c_within_centre_temporal_oe.pdf', 'rect': r_s5[2]},
    {'label': 'd', 'folder': 'Figure8_source_temporal_audit', 'file': 'Figure8d_inpatient_observation_opportunity.pdf', 'rect': r_s5[3]},
]

r_s6 = two_col_rects(2, 170)
supp6 = [
    {'label': 'a', 'folder': 'Figure9_source_variable_quality', 'file': 'Figure9a_predictor_missingness_by_centre.pdf', 'rect': r_s6[0]},
    {'label': 'b', 'folder': 'Figure9_source_variable_quality', 'file': 'Figure9b_outcome_internal_consistency.pdf', 'rect': r_s6[1]},
    {'label': 'c', 'folder': 'Figure9_source_variable_quality', 'file': 'Figure9c_AKI_downstream_risk_difference.pdf',
     'rect': ((PAGE_WIDTH - (r_s6[2][2] - r_s6[2][0])) / 2, r_s6[2][1],
              (PAGE_WIDTH + (r_s6[2][2] - r_s6[2][0])) / 2, r_s6[2][3])},
]


if __name__ == '__main__':
    manifests = [
        compose('Figure1', figure1, 372),
        compose('Figure2', figure2, 674),
        compose('Figure3', figure3, 674),
        compose('Figure4', figure4, 380),
        compose('SupplementaryFigure1', supp1, 370),
        compose('SupplementaryFigure2', supp2, 200),
        compose('SupplementaryFigure3', supp3, 370),
        compose('SupplementaryFigure4', supp4, 350),
        compose('SupplementaryFigure5', supp5, 370),
        compose('SupplementaryFigure6', supp6, 370),
    ]
    audit = {'status': 'PASS', 'figures': manifests}
    (OUTPUT / 'FIGURE_ASSEMBLY_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'figures': [m['figure'] for m in manifests]}, indent=2))
