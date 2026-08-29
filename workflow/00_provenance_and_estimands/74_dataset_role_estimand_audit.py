# %% [markdown]
# # Dataset-role and estimand audit
#
# This script writes the submission tables that prevent evidence from one dataset or
# one outcome target being presented as evidence for another. It contains no patient-
# level information and is safe for the aggregate release package.

# %%
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
TABLES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)


# %%
dataset_roles = pd.DataFrame(
    [
        {
            "dataset": "Five-centre source cohort",
            "analysis_population": "4014 gastric or colorectal cancer operations; 155 adjudicated AKI events",
            "prediction_landmark": "End of surgery",
            "model_role": "Five-centre leave-one-centre-out internal-external evaluation of the locked source models",
            "outcome_target": "Site-level nephrologist-adjudicated KDIGO 2012 postoperative AKI within 7 days (creatinine, urine output, or RRT)",
            "reference_status": "Clinical adjudication; component-level case records unavailable in the analytic extract",
            "permitted_inference": "Clinical case study of model performance and centre heterogeneity",
            "prohibited_inference": "Cannot attribute centre differences to under-monitoring or claim independent external validation",
        },
        {
            "dataset": "INSPIRE 1.4.2",
            "analysis_population": "7135 candidate operations; 6333 longitudinal; 1676 dense-reference operations",
            "prediction_landmark": "End of surgery",
            "model_role": "Harmonised-feature transport analysis and primary controlled measurement-deletion experiment",
            "outcome_target": "All-available postoperative creatinine trajectory over 0-168 h; RRT sensitivity only",
            "reference_status": "Retained operational creatinine reference, not full KDIGO or expert adjudication",
            "permitted_inference": "Quantifies evaluation bias under explicit longitudinal deletion and reconstruction mechanisms",
            "prohibited_inference": "Not source-model same-endpoint clinical external validation and not a clinical gold standard",
        },
        {
            "dataset": "MIMIC-IV 3.1",
            "analysis_population": "9253 first surgical ICU admissions; 9164 longitudinal; 7094 dense-reference stays",
            "prediction_landmark": "First surgical ICU admission landmark",
            "model_role": "Database-native temporal model and independent replication of the measurement-deletion experiment",
            "outcome_target": "Primary all-available creatinine trajectory; algorithmic urine-output/RRT-inclusive target as sensitivity analysis",
            "reference_status": "Operational EHR endpoints, neither expert adjudicated nor identical to the source endpoint",
            "permitted_inference": "Replicates the evaluation mechanism and demonstrates endpoint-target dependence",
            "prohibited_inference": "Not transport validation of the source model and not evidence of clinical impact",
        },
        {
            "dataset": "eICU 2.0",
            "analysis_population": "21,755 adult first surgical ICU stays across 40 hospitals; 9689 dense-reference stays",
            "prediction_landmark": "First surgical ICU unit-stay landmark",
            "model_role": "Database-native ridge model with hospital-group-disjoint testing and third replication of the measurement-deletion experiment",
            "outcome_target": "Primary all-available creatinine trajectory; conservative urine-output/RRT available-component union as sensitivity analysis",
            "reference_status": "Operational EHR endpoints, neither duration-certified expert adjudication nor identical to the source endpoint",
            "permitted_inference": "Replicates the evaluation mechanism across unseen hospitals and demonstrates endpoint-target dependence",
            "prohibited_inference": "Not transport validation of the source model, full KDIGO validation, or evidence of clinical impact",
        },
    ]
)


# %%
estimands = pd.DataFrame(
    [
        {
            "estimand": "Recorded-target model performance",
            "target_outcome": "Outcome reconstructed from measurements retained under the simulated local observation process",
            "population": "Dense-reference public-data cohort",
            "methods": "Naive evaluation; feasible IPAW/AIPW; cross-fitted local recalibration",
            "interpretation": "How the model appears when judged against the locally recorded endpoint",
        },
        {
            "estimand": "Retained-reference model performance",
            "target_outcome": "Outcome constructed before deletion from all available longitudinal creatinine measurements",
            "population": "The same patients and unchanged baseline risk predictions",
            "methods": "Direct reference evaluation; oracle weighting benchmark; prediction-based sensitivity region",
            "interpretation": "Evaluation target for quantifying bias introduced by observation and endpoint reconstruction",
        },
        {
            "estimand": "Reference-target local updating performance",
            "target_outcome": "Retained-reference outcome among patients not used to fit the update",
            "population": "Held-out fold or held-out reference-sample patients",
            "methods": "Two-fold cross-fitted apparent-target recalibration; 5%, 10%, 20%, or 30% random reference-sample recalibration",
            "interpretation": "Whether updating repairs risk transport or merely follows the recorded outcome process",
        },
        {
            "estimand": "Pure label-selection performance",
            "target_outcome": "Unchanged retained-reference label observed for a selected subset without trajectory coarsening",
            "population": "Dense-reference public-data cohort",
            "methods": "Naive selected-sample evaluation and untruncated oracle IPW positive control",
            "interpretation": "Separates recoverable label selection from endpoint misclassification caused by measurement deletion",
        },
        {
            "estimand": "Clinical source performance",
            "target_outcome": "Site-level expert-adjudicated KDIGO postoperative AKI",
            "population": "Five-centre cancer-surgery cohort",
            "methods": "Leave-one-centre-out predictions with discrimination, calibration, precision, and stability analyses",
            "interpretation": "Clinical model case study; does not identify the causal source of centre heterogeneity",
        },
    ]
)


# %%
paths = {
    "dataset_roles": TABLES / "Table_dataset_roles_and_inference_boundaries.csv",
    "estimands": TABLES / "Table_estimand_ledger.csv",
}
dataset_roles.to_csv(paths["dataset_roles"], index=False)
estimands.to_csv(paths["estimands"], index=False)

audit = {
    "status": "PASS",
    "dataset_role_rows": int(len(dataset_roles)),
    "estimand_rows": int(len(estimands)),
    "files": {},
}
for key, path in paths.items():
    audit["files"][key] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }

audit_path = OUTPUTS / "DATASET_ROLE_ESTIMAND_AUDIT.json"
audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2))
