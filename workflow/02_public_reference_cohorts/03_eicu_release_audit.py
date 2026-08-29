# %% [markdown]
# # eICU release audit
# Aggregate-only checks for the frozen operational cohort, exact serum-creatinine
# filter, hospital-group-disjoint split, simulation completeness and endpoint role.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(str(_release_path('analysis')))
EICU = ROOT / "eicu"
OUTPUTS = ROOT / "outputs"
TABLES = ROOT / "tables"
SEED = 20260826

reference = pd.read_csv(EICU / "secure" / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz")
dense = reference.loc[reference.R_dense.eq(1)].sort_values(
    ["hospitaldischargeyear", "hospitalid", "reference_id"]
).copy()
splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, test_idx = next(splitter.split(dense, groups=dense.hospitalid))
train_hospitals = set(dense.iloc[train_idx].hospitalid)
test_hospitals = set(dense.iloc[test_idx].hospitalid)

serial = pd.read_csv(EICU / "secure" / "EICU_CREATININE_SERIAL_SECURE.csv.gz")
build_code = (EICU / "code" / "01_build_eicu_testbed.py").read_text(encoding="utf-8").lower()
simulation = pd.read_csv(TABLES / "Table_eicu_simulation_summary.csv")
model_audit = json.loads((OUTPUTS / "EICU_GROUP_HELDOUT_MODEL_AUDIT.json").read_text())
component_audit = json.loads((OUTPUTS / "EICU_KDIGO_COMPONENT_AUDIT.json").read_text())

checks = {
    "reference_id_unique": bool(reference.reference_id.is_unique),
    "dense_n_is_9689": int(len(dense)) == 9689,
    "serial_hours_are_0_to_168": bool(serial.hour.between(0, 168, inclusive="both").all()),
    "serial_creatinine_positive": bool(serial.creatinine.gt(0).all()),
    "exact_serum_creatinine_filter_in_code": "lower(trim(labname)) = 'creatinine'" in build_code,
    "hospital_groups_are_disjoint": not bool(train_hospitals & test_hospitals),
    "test_hospitals_match_audit": len(test_hospitals) == model_audit["hospitals_test"] == 12,
    "test_size_matches_audit": len(test_idx) == model_audit["n_unseen_hospital_test"] == 3253,
    "all_core_conditions_have_300_replicates": int(simulation.n_replicates.min()) >= 300,
    "primary_endpoint_is_creatinine_only": component_audit["primary_public_endpoint"] == "creatinine-only operational reference",
    "component_union_is_sensitivity_only": component_audit["interpretation"] == "component sensitivity analysis; not replacement of the harmonised creatinine target",
}
payload = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "dense_n": int(len(dense)),
    "train_hospitals": len(train_hospitals),
    "test_hospitals": len(test_hospitals),
    "hospital_overlap": len(train_hospitals & test_hospitals),
    "patient_level_data_exported": False,
}
(OUTPUTS / "EICU_RELEASE_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if payload["status"] != "PASS":
    raise SystemExit(1)
