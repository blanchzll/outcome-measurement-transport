# %% [markdown]
# # Cross-database external-validation QA
# Release-safe checks for the locked INSPIRE model and clinical bridge.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image


ROOT = Path(
    '<external-path-redacted>'
    "ascertainment_framework_20260826/cross_database_external_validation"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


required = [
    ROOT / "code/84_build_inspire_surgical_icu_reference.py",
    ROOT / "code/85_inspire_locked_public_icu_transport.py",
    ROOT / "code/86_inspire_gi_model_to_mimic_and_source.py",
    ROOT / "code/87_plot_cross_database_external_validation.py",
    ROOT / "tests/test_cross_database_external_validation.py",
    ROOT / "logs/PYTEST_CROSS_DATABASE.log",
    ROOT / "outputs/INSPIRE_SURGICAL_ICU_REFERENCE_AUDIT.json",
    ROOT / "outputs/INSPIRE_LOCKED_EXTERNAL_VALIDATION_AUDIT.json",
    ROOT / "outputs/PUBLIC_MODEL_TO_SOURCE_CLINICAL_BRIDGE_AUDIT.json",
    ROOT / "tables/Table_inspire_locked_external_validation.csv",
    ROOT / "tables/Table_public_model_to_source_clinical_bridge.csv",
    ROOT / "figures/SupplementaryFigure7/SupplementaryFigure7.pdf",
    ROOT / "figures/SupplementaryFigure8/SupplementaryFigure8.pdf",
]

public_validation = pd.read_csv(ROOT / "tables/Table_inspire_locked_external_validation.csv")
clinical_bridge = pd.read_csv(ROOT / "tables/Table_public_model_to_source_clinical_bridge.csv")

numeric_columns = [
    "n", "events", "auc", "oe", "calibration_intercept", "calibration_slope"
]
finite_metrics = True
for frame in (public_validation, clinical_bridge):
    available = [column for column in numeric_columns if column in frame.columns]
    finite_metrics &= bool(frame[available].map(math.isfinite).all().all())

pdf_checks = []
tiff_checks = []
for number in (7, 8):
    folder = ROOT / f"figures/SupplementaryFigure{number}"
    pdf = folder / f"SupplementaryFigure{number}.pdf"
    with pymupdf.open(pdf) as doc:
        page = doc[0]
        pdf_checks.append(
            {
                "file": str(pdf.relative_to(ROOT)),
                "pages": len(doc),
                "width_pt": page.rect.width,
                "width_183mm": abs(page.rect.width - 183 / 25.4 * 72) < 0.5,
            }
        )
    tiff = folder / f"SupplementaryFigure{number}.tiff"
    with Image.open(tiff) as image:
        dpi = image.info.get("dpi", (0, 0))
        tiff_checks.append(
            {
                "file": str(tiff.relative_to(ROOT)),
                "pixels": list(image.size),
                "dpi": [round(float(dpi[0])), round(float(dpi[1]))],
                "at_least_600_dpi": min(dpi) >= 599,
            }
        )

patient_level_outside_secure = []
patient_markers = ("prediction", "patient", "subject_id", "stay_id", "hadm_id", "caseid")
for path in ROOT.rglob("*"):
    if not path.is_file() or "secure_work" in path.parts:
        continue
    lower = path.name.lower()
    if any(marker in lower for marker in patient_markers) and path.suffix.lower() in {".csv", ".gz", ".parquet", ".xlsx"}:
        patient_level_outside_secure.append(str(path.relative_to(ROOT)))

checks = {
    "required_files_present": all(path.exists() and path.stat().st_size > 0 for path in required),
    "regression_tests_pass": "3 passed" in (ROOT / "logs/PYTEST_CROSS_DATABASE.log").read_text(),
    "metrics_finite": finite_metrics,
    "locked_public_rows_complete": int(
        public_validation["validation_database"].ne("INSPIRE").sum()
    ) == 4,
    "clinical_bridge_rows_complete": len(clinical_bridge) == 3,
    "figures_one_page_183mm": all(item["pages"] == 1 and item["width_183mm"] for item in pdf_checks),
    "figures_600dpi": all(item["at_least_600_dpi"] for item in tiff_checks),
    "panel_source_data_present": all(
        any(folder.glob("*_source_data.csv"))
        for folder in [
            ROOT / "figures/SupplementaryFigure7",
            ROOT / "figures/SupplementaryFigure8",
        ]
    ),
    "patient_level_outputs_confined_to_secure_work": not patient_level_outside_secure,
}

release_files = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or "secure_work" in path.parts or "__pycache__" in path.parts:
        continue
    if path.name in {"CROSS_DATABASE_EXTERNAL_VALIDATION_QA.json", "SHA256SUMS_RELEASE"}:
        continue
    release_files.append(
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )

payload = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "pdf_checks": pdf_checks,
    "tiff_checks": tiff_checks,
    "patient_level_outside_secure_work": patient_level_outside_secure,
    "release_file_count": len(release_files),
    "patient_level_data_read_by_qa": False,
    "scientific_boundaries": {
        "public_locked_validation": "True same-model external validation of an INSPIRE-developed surgical-ICU model at ICU admission; not validation of the source end-of-surgery model.",
        "source_clinical_bridge": "External evaluation of an INSPIRE-developed GI model in the five-centre cohort under a different endpoint reference; not strict same-endpoint validation.",
        "public_outcome": "Operational creatinine-only KDIGO reference, not clinician-adjudicated full KDIGO.",
    },
}
(ROOT / "outputs/CROSS_DATABASE_EXTERNAL_VALIDATION_QA.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
(ROOT / "outputs/SHA256SUMS_RELEASE").write_text(
    "".join(f"{item['sha256']}  {item['path']}\n" for item in release_files)
)
print(json.dumps(payload, indent=2))
if payload["status"] != "PASS":
    raise SystemExit(1)
