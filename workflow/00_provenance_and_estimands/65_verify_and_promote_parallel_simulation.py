# %% [markdown]
# # Verify and promote parallel simulation artifacts
# Atomically promotes only complete 300-replicate outputs.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(str(_release_path('analysis')))
SECURE, TABLES, OUTPUTS = ROOT / "secure_work", ROOT / "tables", ROOT / "outputs"
EXPECTED_METHODS = {
    "full_reference", "naive", "IPAW_design_probability_untruncated",
    "IPAW_design_probability_truncated99", "AIPW_design_probability",
    "recalibration_intercept_apparent", "recalibration_intercept_truth",
    "recalibration_intercept_slope_apparent", "recalibration_intercept_slope_truth",
    "reference_05pct_recalibration", "reference_10pct_recalibration",
    "reference_20pct_recalibration", "reference_30pct_recalibration",
    "Gamma2_prediction_sensitivity_region",
}
CORE_METHODS = EXPECTED_METHODS - {method for method in EXPECTED_METHODS if method.startswith("reference_")}


def atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(target.name + ".promoting")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


promotion = {}
for database in ("INSPIRE", "MIMIC", "EICU"):
    audit_path = OUTPUTS / f"{database}_SIMULATION_PARALLEL_AUDIT.json"
    raw_path = SECURE / f"{database}_SIMULATION_REPLICATES_PARALLEL_SECURE.csv.gz"
    summary_path = TABLES / f"Table_{database.lower()}_simulation_summary_parallel.csv"
    audit = json.loads(audit_path.read_text())
    summary = pd.read_csv(summary_path)
    raw = pd.read_csv(raw_path, usecols=["method"])
    checks = {
        "replicates_per_condition": audit.get("replicates_per_condition") == 300,
        "conditions": audit.get("conditions") == 36,
        "replicate_rows": 108000 <= audit.get("replicate_rows", 0) <= 151200 and len(raw) == audit.get("replicate_rows"),
        "core_summary_replicates": int(summary.loc[summary.method.isin(CORE_METHODS), "n_replicates"].min()) == 300,
        "reference_summary_replicates": int(summary.loc[summary.method.str.startswith("reference_"), "n_replicates"].min()) >= 250,
        "methods": EXPECTED_METHODS.issubset(set(raw.method)),
        "mechanisms": set(summary.mechanism) == {"MCAR", "stratum_MAR", "risk_MAR", "history_MAR", "outcome_MNAR", "mixed_MNAR"},
    }
    if not all(checks.values()):
        raise SystemExit(f"{database} parallel output failed promotion checks: {checks}")
    canonical_audit = dict(audit)
    canonical_audit.update({"runner": "parallel condition runner", "promotion_checks": checks})
    canonical_audit_path = OUTPUTS / f"{database}_SIMULATION_AUDIT.json"
    atomic_copy(raw_path, SECURE / f"{database}_SIMULATION_REPLICATES_SECURE.csv.gz")
    atomic_copy(summary_path, TABLES / f"Table_{database.lower()}_simulation_summary.csv")
    tmp_audit = canonical_audit_path.with_name(canonical_audit_path.name + ".promoting")
    tmp_audit.write_text(json.dumps(canonical_audit, indent=2) + "\n")
    os.replace(tmp_audit, canonical_audit_path)
    promotion[database] = checks

(OUTPUTS / "PARALLEL_SIMULATION_PROMOTION_AUDIT.json").write_text(
    json.dumps({"promoted": True, "databases": promotion}, indent=2) + "\n"
)
print(json.dumps({"promoted": True, "databases": list(promotion)}, indent=2))
