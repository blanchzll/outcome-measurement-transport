# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Secondary fixed-geography validation: centres 3/4/5 to centres 1/2
#
# This is a descriptive transport analysis. Centres 1 and 2 have already
# contributed outer folds to the primary LOCO analysis and are therefore not
# described as an untouched external validation cohort.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from release_paths import release_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE = REPOSITORY_ROOT / "workflow/06_transport_external_validation"
SOURCE_COHORT = release_path("source", "clean_clinical_data.csv")
RESULT_ROOT = release_path("analysis", "outputs/source3710_fixed_geography")
sys.path.insert(0, str(BASE))

from analysis import CENTER, STABLE_ID, TARGET, load_cohort  # noqa: E402
from loco_analysis import (  # noqa: E402
    BASE_MODELS,
    FEATURE_SET_SPECS,
    bootstrap_metric_ci,
    build_loco_search,
    engineer_loco_features,
    probability_metrics,
)

SEED = 20260827
N_BOOTSTRAP = 1000
TRAIN_CENTRES = (3, 4, 5)
TEST_CENTRES = (1, 2)
FEATURE_SETS = ("P", "PI", "H")


def metric_rows(y, p, feature_set, model, evaluation, centres=None):
    point = probability_metrics(y, p)
    intervals = bootstrap_metric_ci(
        y,
        p,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED + sum(ord(ch) for ch in f"{feature_set}-{model}-{evaluation}"),
        groups=centres,
    )
    row = {
        "evaluation": evaluation,
        "feature_set": feature_set,
        "model": model,
        "validation_status": "secondary_fixed_geography_not_untouched",
        "bootstrap_unit": "analytic_record_within_test_centre" if centres is not None else "analytic_record",
        **point,
    }
    for metric, (lower, upper) in intervals.items():
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper
    return row


# %%
source = engineer_loco_features(load_cohort(SOURCE_COHORT)).reset_index(drop=True)
if len(source) != 3710 or source[STABLE_ID].nunique() != 3710:
    raise ValueError("The locked primary cohort must contain 3710 unique patients.")
if int(source[TARGET].sum()) != 152:
    raise ValueError("The locked primary cohort must contain 152 recorded events.")
train_mask = source[CENTER].isin(TRAIN_CENTRES).to_numpy()
test_mask = source[CENTER].isin(TEST_CENTRES).to_numpy()
assert train_mask.sum() + test_mask.sum() == len(source)

y_train = source.loc[train_mask, TARGET].to_numpy(dtype=int)
g_train = source.loc[train_mask, CENTER].to_numpy(dtype=int)
y_test = source.loc[test_mask, TARGET].to_numpy(dtype=int)
g_test = source.loc[test_mask, CENTER].to_numpy(dtype=int)

prediction_frame = source.loc[test_mask, [STABLE_ID, CENTER, TARGET]].copy().reset_index(drop=True)
rows = []
fits = []

for feature_set in FEATURE_SETS:
    spec = FEATURE_SET_SPECS[feature_set]
    fitted = {}
    for model in BASE_MODELS:
        search = build_loco_search(spec, model, n_inner_centers=len(np.unique(g_train)), fast=False)
        search.fit(source.loc[train_mask, list(spec.features)], y_train, groups=g_train)
        fitted[model] = search
        fits.append(
            {
                "feature_set": feature_set,
                "model": model,
                "best_params": search.best_params_,
                "inner_selection_metric": "negative_brier_score",
                "n_train": int(train_mask.sum()),
                "events_train": int(y_train.sum()),
            }
        )

    model_predictions = {
        model: fitted[model].predict_proba(source.loc[test_mask, list(spec.features)])[:, 1]
        for model in BASE_MODELS
    }
    model_predictions["soft_voting"] = np.mean(list(model_predictions.values()), axis=0)

    for model, probabilities in model_predictions.items():
        prediction_frame[f"pred_{feature_set}_{model}"] = probabilities
        rows.append(metric_rows(y_test, probabilities, feature_set, model, "centres_1_2_combined", g_test))
        for centre in TEST_CENTRES:
            keep = g_test == centre
            rows.append(metric_rows(y_test[keep], probabilities[keep], feature_set, model, f"centre_{centre}"))

results = pd.DataFrame(rows)
tables = RESULT_ROOT / "tables"
secure = RESULT_ROOT / "secure_work"
outputs = RESULT_ROOT / "outputs"
for directory in (tables, secure, outputs):
    directory.mkdir(parents=True, exist_ok=True)
results.to_csv(tables / "Table_source_fixed_geography_validation_3710.csv", index=False)
prediction_frame.to_csv(
    secure / "SOURCE_3710_FIXED_GEOGRAPHY_PREDICTIONS_SECURE.csv.gz",
    index=False,
    compression="gzip",
)

audit = {
    "analysis": "secondary fixed-geography validation",
    "train_centres": list(TRAIN_CENTRES),
    "test_centres": list(TEST_CENTRES),
    "n_train": int(train_mask.sum()),
    "events_train": int(y_train.sum()),
    "n_test": int(test_mask.sum()),
    "events_test": int(y_test.sum()),
    "interpretation_boundary": (
        "Centres 1 and 2 were outer folds in the primary LOCO analysis; this is a clinically familiar "
        "fixed-geography transport summary, not an untouched external validation."
    ),
    "interval_estimand": (
        "Percentile intervals from unique-patient resampling within test centre, conditional on the fitted "
        "development model; investigator-confirmed one eligible operation per patient."
    ),
    "patient_level_outputs_delivered": False,
    "fits": fits,
}
(outputs / "SOURCE_FIXED_GEOGRAPHY_VALIDATION_3710_AUDIT.json").write_text(
    json.dumps(audit, indent=2), encoding="utf-8"
)
print(json.dumps({key: value for key, value in audit.items() if key != "fits"}, indent=2))
