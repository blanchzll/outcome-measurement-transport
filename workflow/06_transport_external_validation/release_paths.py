"""Environment-based path configuration for restricted-data workflows."""
from __future__ import annotations
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "analysis": REPOSITORY_ROOT / "analysis_workspace",
    "source": REPOSITORY_ROOT / "restricted_data/source",
    "inspire": REPOSITORY_ROOT / "restricted_data/physionet/inspire/1.4.2",
    "mimic": REPOSITORY_ROOT / "restricted_data/physionet/mimic",
    "eicu": REPOSITORY_ROOT / "restricted_data/physionet/eicu/2.0",
    "vitaldb": REPOSITORY_ROOT / "restricted_data/physionet/vitaldb/1.0.0",
    "mimic_duckdb": REPOSITORY_ROOT / "restricted_data/mimiciv31.duckdb",
}
ENVIRONMENT = {
    "analysis": "AKI_ANALYSIS_ROOT",
    "source": "AKI_SOURCE_ROOT",
    "inspire": "INSPIRE_ROOT",
    "mimic": "MIMIC_ROOT",
    "eicu": "EICU_ROOT",
    "vitaldb": "VITALDB_ROOT",
    "mimic_duckdb": "MIMIC_DUCKDB",
}

def release_path(kind: str, *parts: str) -> Path:
    base = Path(os.environ.get(ENVIRONMENT[kind], str(DEFAULTS[kind])))
    return base.joinpath(*parts)
