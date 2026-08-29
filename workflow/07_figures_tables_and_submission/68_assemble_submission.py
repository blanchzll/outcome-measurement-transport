# %% [markdown]
# # Assemble the aggregate submission package
#
# This script copies only aggregate, publication-facing artifacts into the delivery tree.
# It does not read or export patient-level data.

# %%
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


# %%
ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "delivery"

MAIN_MANUSCRIPT = DELIVERY / "main" / "manuscript"
MAIN_FIGURES = DELIVERY / "main" / "figures"
MAIN_TABLES = DELIVERY / "main" / "tables"
SUPP_MANUSCRIPT = DELIVERY / "supplement" / "manuscript"
SUPP_FIGURES = DELIVERY / "supplement" / "figures"
SUPP_TABLES = DELIVERY / "supplement" / "tables"
REPRO = DELIVERY / "reproducibility"

for directory in (
    MAIN_MANUSCRIPT,
    MAIN_FIGURES,
    MAIN_TABLES,
    SUPP_MANUSCRIPT,
    SUPP_FIGURES,
    SUPP_TABLES,
    REPRO,
):
    directory.mkdir(parents=True, exist_ok=True)


# %%
def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )


# %%
for name in ("MANUSCRIPT_FINAL.md", "FIGURE_LEGENDS.md", "TABLE_LEGENDS.md", "COVER_LETTER_LANCET_DIGITAL_HEALTH.md"):
    copy_file(ROOT / "manuscript" / name, MAIN_MANUSCRIPT / name)

copy_file(
    ROOT / "manuscript" / "SUPPLEMENTARY_APPENDIX.md",
    SUPP_MANUSCRIPT / "SUPPLEMENTARY_APPENDIX.md",
)
copy_file(
    ROOT / "manuscript" / "TOP_JOURNAL_REVIEW.md",
    DELIVERY / "TOP_JOURNAL_REVIEW.md",
)

main_figure_folders = (
    "Figure1_reference_observability",
    "Figure2_deletion_mechanisms",
    "Figure3_correction_strategies",
    "Figure4_stability_portability",
)
for folder in main_figure_folders:
    for source in sorted((ROOT / "figures" / folder).glob("*.pdf")):
        copy_file(source, MAIN_FIGURES / folder / source.name)
    for source in sorted((ROOT / "figures" / folder).glob("*_source_data.csv")):
        copy_file(source, MAIN_FIGURES / folder / source.name)

for source in sorted((ROOT / "figures" / "Figure5_clinical_audit").glob("*.pdf")):
    copy_file(source, SUPP_FIGURES / "Figure5_clinical_audit" / source.name)
for source in sorted((ROOT / "figures" / "Figure5_clinical_audit").glob("*_source_data.csv")):
    copy_file(source, SUPP_FIGURES / "Figure5_clinical_audit" / source.name)

for source in sorted((ROOT / "figures" / "Figure6_eicu_replication").glob("*.pdf")):
    copy_file(source, SUPP_FIGURES / "Figure6_eicu_replication" / source.name)
for source in sorted((ROOT / "figures" / "Figure6_eicu_replication").glob("*_source_data.csv")):
    copy_file(source, SUPP_FIGURES / "Figure6_eicu_replication" / source.name)

for source in sorted((ROOT / "figures" / "Figure7_robustness_extensions").glob("*.pdf")):
    copy_file(source, SUPP_FIGURES / "Figure7_robustness_extensions" / source.name)
for source in sorted((ROOT / "figures" / "Figure7_robustness_extensions").glob("*_source_data.csv")):
    copy_file(source, SUPP_FIGURES / "Figure7_robustness_extensions" / source.name)

for source in sorted((ROOT / "figures" / "Figure8_source_temporal_audit").glob("*.pdf")):
    copy_file(source, SUPP_FIGURES / "Figure8_source_temporal_audit" / source.name)
for source in sorted((ROOT / "figures" / "Figure8_source_temporal_audit").glob("*_source_data.csv")):
    copy_file(source, SUPP_FIGURES / "Figure8_source_temporal_audit" / source.name)

for source in sorted((ROOT / "figures" / "Figure9_source_variable_quality").glob("*.pdf")):
    copy_file(source, SUPP_FIGURES / "Figure9_source_variable_quality" / source.name)
for source in sorted((ROOT / "figures" / "Figure9_source_variable_quality").glob("*_source_data.csv")):
    copy_file(source, SUPP_FIGURES / "Figure9_source_variable_quality" / source.name)

main_tables = {
    "Table_key_source_model_results.csv": "Table1_source_model_results.csv",
    "Table_key_mixed_mnar_results.csv": "Table2_mixed_MNAR_results.csv",
    "Table_key_apparent_vs_reference_recalibration.csv": "Table3_apparent_vs_reference_recalibration.csv",
    "Table_key_reference_sample_design.csv": "Table4_reference_sample_design.csv",
}
for source_name, destination_name in main_tables.items():
    copy_file(ROOT / "tables" / source_name, MAIN_TABLES / destination_name)

for source in sorted((ROOT / "tables").glob("*.csv")):
    if source.name not in main_tables:
        copy_file(source, SUPP_TABLES / source.name)

for source in sorted((ROOT / "eicu" / "tables").glob("*.csv")):
    copy_file(source, SUPP_TABLES / source.name)

copy_tree(ROOT / "package" / "ascertainment-stress-test", REPRO / "ascertainment-stress-test")
copy_tree(ROOT / "code", REPRO / "analysis-code")
copy_tree(ROOT / "eicu" / "code", REPRO / "eicu-code")
for name in (
    "STATISTICAL_ANALYSIS_PLAN.md",
    "RISK_REGISTER.md",
    "REPORTING_AND_BIAS_AUDIT.md",
    "TERMINOLOGY_LEDGER.md",
    "FIGURE_CONTRACT.md",
    "LITERATURE_EVIDENCE_MAP.md",
    "PROJECT_STATIC_CONTEXT.md",
    "ETHICS_PROTOCOL_CONCORDANCE_AUDIT.md",
):
    copy_file(ROOT / "protocol" / name, REPRO / "protocol" / name)

for name in (
    "ANALYSIS_COMPLETION_AUDIT.json",
    "FIGURE_SOURCE_PREFLIGHT.json",
    "PUBLICATION_FACT_BASE.json",
    "SOURCE_OUTCOME_ADJUDICATION_RECORD.json",
    "INSPIRE_SIMULATION_AUDIT.json",
    "MIMIC_SIMULATION_AUDIT.json",
    "EICU_SIMULATION_AUDIT.json",
    "INSPIRE_PURE_LABEL_SELECTION_CONTROL_AUDIT.json",
    "MIMIC_PURE_LABEL_SELECTION_CONTROL_AUDIT.json",
    "EICU_PURE_LABEL_SELECTION_CONTROL_AUDIT.json",
    "EICU_GROUP_HELDOUT_MODEL_AUDIT.json",
    "EICU_KDIGO_COMPONENT_AUDIT.json",
    "EICU_RELEASE_AUDIT.json",
    "PUBLIC_KDIGO_COMPONENT_AUDIT.json",
    "DATASET_ROLE_ESTIMAND_AUDIT.json",
    "SIMULATION_ESTIMAND_RESUMMARY_AUDIT.json",
    "PARALLEL_SIMULATION_PROMOTION_AUDIT.json",
    "SIMULATION_LABEL_NORMALIZATION_AUDIT.json",
    "CORRECTED_SIMULATION_RESULT_DIGEST.json",
    "PAPER_IMPROVEMENT_STATE.json",
    "SOURCE_REPORTING_TABLE_AUDIT.json",
    "DENSE_REFERENCE_SELECTION_AUDIT.json",
    "REFERENCE_EVENT_DESIGN_AUDIT.json",
    "SOURCE_FIXED_GEOGRAPHY_VALIDATION_AUDIT.json",
    "PUBLIC_HARMONIZED_TRANSPORT_AUDIT.json",
    "EICU_COMMON_PREDICTOR_AUDIT.json",
    "PUBLIC_EXTENDED_TRANSPORT_AUDIT.json",
    "DISCRIMINATION_STRENGTH_STRESS_AUDIT.json",
    "BAYESIAN_HIERARCHICAL_CALIBRATION_AUDIT.json",
    "MEASUREMENT_AWARE_FAIRNESS_AUDIT.json",
    "FIGURE7_EXTENSION_AUDIT.json",
    "SOURCE_TEMPORAL_VALIDATION_AUDIT.json",
    "FIGURE8_TEMPORAL_AUDIT.json",
    "SOURCE_VARIABLE_ROLE_AUDIT.json",
    "SOURCE_MODEL_DATA_QUALITY_SENSITIVITY_AUDIT.json",
    "FIGURE9_SOURCE_VARIABLE_AUDIT.json",
    "SOURCE_OBSERVATION_AND_PREOP_AKI_SENSITIVITY_AUDIT.json",
    "WATERMARK_RELEASE_AUDIT.json",
    "MANUSCRIPT_RELEASE_AUDIT.json",
    "REPRODUCIBILITY_AUDIT.json",
):
    source = ROOT / "outputs" / name
    if source.exists():
        copy_file(source, REPRO / "audits" / name)

for name in ("EICU_REFERENCE_AUDIT.json",):
    source = ROOT / "eicu" / "outputs" / name
    if source.exists():
        copy_file(source, REPRO / "audits" / name)

for name in ("PAPER_IMPROVEMENT_LOG.md", "TOP_JOURNAL_FINAL_REVIEW.md", "FINAL_PRE_SUBMISSION_REVIEW.md", "AUTHOR_INPUT_REQUIRED.md"):
    source = ROOT / "manuscript" / name
    if source.exists():
        copy_file(source, REPRO / "review" / name)

reviews_source = ROOT / "manuscript" / "reviews"
if reviews_source.exists():
    copy_tree(reviews_source, REPRO / "review" / "blind_reviews")

# The DOCX builder stages the supplement beside the main manuscript before it is
# copied to the supplement tree. Remove that staging duplicate from the final layout.
staged_supplement = MAIN_MANUSCRIPT / "SUPPLEMENTARY_APPENDIX.docx"
if staged_supplement.exists():
    copy_file(staged_supplement, SUPP_MANUSCRIPT / "SUPPLEMENTARY_APPENDIX.docx")
    staged_supplement.unlink()


# %%
manifest = []
for path in sorted(DELIVERY.rglob("*")):
    if not path.is_file() or path.name == "DELIVERY_MANIFEST.json":
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.append(
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    )

payload = {
    "root": str(ROOT),
    "patient_level_data_read": False,
    "patient_level_data_exported": False,
    "files": len(manifest),
    "manifest": manifest,
}
(DELIVERY / "DELIVERY_MANIFEST.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

print(json.dumps({"files": len(manifest), "delivery": str(DELIVERY)}, indent=2))
