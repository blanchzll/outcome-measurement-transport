# %% [markdown]
# # Normalize simulation method labels
# Semantic-only migration from development labels to publication-safe labels.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
from pathlib import Path

import pandas as pd

ROOT = Path(str(_release_path('analysis')))
RENAME = {
    "IPAW_oracle": "IPAW_design_probability_untruncated",
    "IPAW_known_observability": "IPAW_design_probability_truncated99",
    "AIPW_oracle_propensity": "AIPW_design_probability",
    "AIPW_known_observability": "AIPW_design_probability",
    "MNAR_Gamma2_envelope": "Gamma2_prediction_sensitivity_region",
}

changes = {}
for database in ("INSPIRE", "MIMIC", "EICU"):
    database_changes = {}
    paths = (
        ROOT / "tables" / f"Table_{database.lower()}_simulation_summary.csv",
        ROOT / "secure_work" / f"{database}_SIMULATION_REPLICATES_SECURE.csv.gz",
        ROOT / "tables" / f"Table_{database.lower()}_simulation_summary_parallel.csv",
        ROOT / "secure_work" / f"{database}_SIMULATION_REPLICATES_PARALLEL_SECURE.csv.gz",
    )
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        before = frame["method"].value_counts().to_dict()
        frame["method"] = frame["method"].replace(RENAME)
        after = frame["method"].value_counts().to_dict()
        frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)
        database_changes[path.name] = {"before": before, "after": after, "rows": len(frame)}
    changes[database] = database_changes

audit = {
    "semantic_only": True,
    "numeric_values_changed": False,
    "mapping": RENAME,
    "files": changes,
}
(ROOT / "outputs" / "SIMULATION_LABEL_NORMALIZATION_AUDIT.json").write_text(
    json.dumps(audit, indent=2) + "\n"
)
print(json.dumps({"semantic_only": True, "databases": list(changes)}, indent=2))
