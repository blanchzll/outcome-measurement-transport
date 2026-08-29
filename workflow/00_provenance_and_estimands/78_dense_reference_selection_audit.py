# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Candidate-to-dense-reference selection audit

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(str(_release_path('analysis')))
SECURE, TABLES, OUTPUTS = ROOT / "secure_work", ROOT / "tables", ROOT / "outputs"
EICU_SECURE = ROOT / "eicu" / "secure"


def continuous_row(database, variable, full, dense):
    a, b = pd.to_numeric(full, errors="coerce").dropna(), pd.to_numeric(dense, errors="coerce").dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {
        "database": database, "variable": variable, "level": "", "type": "continuous",
        "candidate_value": float(a.mean()), "dense_value": float(b.mean()),
        "standardized_difference": float((b.mean() - a.mean()) / pooled) if pooled > 0 else np.nan,
    }


def categorical_rows(database, variable, full, dense):
    levels = sorted(set(full.dropna().astype(str)) | set(dense.dropna().astype(str)))
    rows = []
    for level in levels:
        pa, pb = full.astype(str).eq(level).mean(), dense.astype(str).eq(level).mean()
        pooled = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
        rows.append({
            "database": database, "variable": variable, "level": level, "type": "categorical",
            "candidate_value": float(pa), "dense_value": float(pb),
            "standardized_difference": float((pb - pa) / pooled) if pooled > 0 else np.nan,
        })
    return rows


rows = []
flow = []

ins = pd.read_csv(SECURE / "INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz", low_memory=False)
ins_dense = ins.loc[ins.dense_reference.eq(1)]
for variable in ["Age", "PreopCr", "PreopHb", "PreopAlb", "n_postop_creatinine_7d", "restricted_rf_probability"]:
    rows.append(continuous_row("INSPIRE", variable, ins[variable], ins_dense[variable]))
for variable in ["Gender", "Gastrocolorectal", "SurgicalApproach", "AnyIntraopTransfusion"]:
    rows.extend(categorical_rows("INSPIRE", variable, ins[variable], ins_dense[variable]))
flow.append({"database": "INSPIRE", "candidate_n": len(ins), "dense_n": len(ins_dense), "dense_fraction": len(ins_dense)/len(ins)})

mimic = pd.read_csv(SECURE / "MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
mimic_dense = mimic.loc[mimic.R_dense.eq(1)]
for variable in ["age", "baseline_creatinine", "baseline_albumin", "baseline_bun", "baseline_hemoglobin", "n_postop_creatinine"]:
    rows.append(continuous_row("MIMIC-IV", variable, mimic[variable], mimic_dense[variable]))
for variable in ["gender", "admission_type", "active_service", "diabetes"]:
    rows.extend(categorical_rows("MIMIC-IV", variable, mimic[variable], mimic_dense[variable]))
flow.append({"database": "MIMIC-IV", "candidate_n": len(mimic), "dense_n": len(mimic_dense), "dense_fraction": len(mimic_dense)/len(mimic)})

eicu = pd.read_csv(EICU_SECURE / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
eicu_dense = eicu.loc[eicu.R_dense.eq(1)]
for variable in ["age_num", "baseline_creatinine", "admissionheight", "admissionweight", "n_creatinine_0_168h"]:
    rows.append(continuous_row("eICU", variable, eicu[variable], eicu_dense[variable]))
for variable in ["gender", "unittype", "region", "teachingstatus"]:
    rows.extend(categorical_rows("eICU", variable, eicu[variable], eicu_dense[variable]))
flow.append({"database": "eICU", "candidate_n": len(eicu), "dense_n": len(eicu_dense), "dense_fraction": len(eicu_dense)/len(eicu)})

table = pd.DataFrame(rows)
table["absolute_standardized_difference"] = table.standardized_difference.abs()
table["candidate_n"] = table.database.map({"INSPIRE": len(ins), "MIMIC-IV": len(mimic), "eICU": len(eicu)})
table["dense_n"] = table.database.map({"INSPIRE": len(ins_dense), "MIMIC-IV": len(mimic_dense), "eICU": len(eicu_dense)})
table.to_csv(TABLES / "Table_dense_reference_selection_audit.csv", index=False)
pd.DataFrame(flow).to_csv(TABLES / "Table_dense_reference_selection_flow.csv", index=False)

audit = {
    "INSPIRE": flow[0],
    "MIMIC-IV": flow[1],
    "eICU": flow[2],
    "largest_absolute_standardized_difference": {
        db: float(g.absolute_standardized_difference.max()) for db, g in table.groupby("database")
    },
    "interpretation": "descriptive selection audit; dense-reference cohorts are not random or clinical gold standards",
}
(OUTPUTS / "DENSE_REFERENCE_SELECTION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
