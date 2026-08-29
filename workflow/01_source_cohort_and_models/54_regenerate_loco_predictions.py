# %% [markdown]
# # Regenerate locked patient-level LOCO predictions for secondary audits
# No model or feature selection is performed here; fold-specific parameters are read
# from the pre-existing 3710 lock file.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import ast
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(str(_release_path('source')))
ROOT = BASE / "ascertainment_framework_20260826"
sys.path.insert(0, str(BASE))

from analysis import CENTER, TARGET, load_cohort  # noqa: E402
from loco_analysis import BASE_MODELS, FEATURE_SET_SPECS, build_loco_search, engineer_loco_features  # noqa: E402


def parse_value(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["3710", "4014"], default="3710")
    return parser.parse_args()


# %%
args = parse_args()
if args.cohort == "4014":
    data_path = BASE / "secure_source" / "inter3_deidentified_4014.csv"
    lock_path = BASE / "outputs_evidence_closure_20260823" / "loco_4014_corrected_formal" / "model_lock.json"
else:
    data_path = Path(str(_release_path('source', 'clean_clinical_data.csv')))
    lock_path = BASE / "outputs_loco_3710_corrected_v2_20260823" / "model_lock.json"
lock = json.loads(lock_path.read_text())
lookup = {}
for row in lock["fold_locks"]:
    if row["model"] in BASE_MODELS:
        lookup[(int(row["outer_center"]), row["feature_set"], row["model"])] = {
            k: parse_value(v) for k, v in row["best_params"].items()
        }

cohort = engineer_loco_features(load_cohort(data_path)).reset_index(drop=True)
centers = sorted(cohort[CENTER].astype(int).unique())
feature_sets = ["P", "PI", "H"]
pred = {(fs, model): np.full(len(cohort), np.nan) for fs in feature_sets for model in (*BASE_MODELS, "soft_voting")}
fit_audit = []

for center in centers:
    train = cohort[CENTER].astype(int).ne(center).to_numpy()
    test = ~train
    groups = cohort.loc[train, CENTER].astype(int).to_numpy()
    y = cohort.loc[train, TARGET].astype(int).to_numpy()
    for fs in feature_sets:
        spec = FEATURE_SET_SPECS[fs]
        for model in BASE_MODELS:
            params = lookup[(center, fs, model)]
            search = build_loco_search(spec, model, n_inner_centers=len(np.unique(groups)), fast=True)
            search.set_params(param_grid={k: [v] for k, v in params.items()})
            search.fit(cohort.loc[train, list(spec.features)], y, groups=groups)
            pred[(fs, model)][test] = search.predict_proba(cohort.loc[test, list(spec.features)])[:, 1]
            fit_audit.append({"outer_center": center, "feature_set": fs, "model": model,
                              "parameters": params, "n_train": int(train.sum()), "n_test": int(test.sum())})
        pred[(fs, "soft_voting")][test] = np.mean([pred[(fs, m)][test] for m in BASE_MODELS], axis=0)

assert all(np.isfinite(x).all() for x in pred.values())
out = cohort.copy()
out.insert(0, "audit_row_id", [f"SRC-{i:05d}" for i in range(1, len(out) + 1)])
for (fs, model), values in pred.items():
    out[f"pred_{fs}_{model}"] = values
out.to_csv(ROOT / "secure_work" / f"SOURCE_{args.cohort}_LOCKED_LOCO_PREDICTIONS_SECURE.csv.gz", index=False, compression="gzip")
audit = {"n": len(out), "events": int(out[TARGET].sum()), "centers": [int(x) for x in centers],
         "cohort": args.cohort,
         "data_role": "primary observable cohort" if args.cohort == "4014" else "historical sensitivity cohort",
         "selection_note": "4014 retains unresolved sex for fold-internal imputation" if args.cohort == "4014" else "3710 reproduces the legacy Gender 0/1 filter",
         "patient_level_predictions_public": False, "fits": fit_audit}
(ROOT / "outputs" / f"SOURCE_{args.cohort}_LOCO_REGENERATION_AUDIT.json").write_text(
    json.dumps(audit, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8"
)
print(json.dumps({k: v for k, v in audit.items() if k != "fits"}, indent=2))
