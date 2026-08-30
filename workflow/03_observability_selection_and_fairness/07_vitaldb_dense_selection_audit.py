# %% [markdown]
# # VitalDB dense-reference selection and correction audit
#
# Within operations that have an observable 0-168 h creatinine reference, this
# analysis treats the full observable cohort as known operational truth and
# tests how well IPAW/AIPW recover its event rate from the selected dense subset.

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260830


def pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), categorical),
        ]
    )
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(C=0.25, solver="liblinear", max_iter=2000, random_state=SEED))])


def weighted_rate(y: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(np.asarray(y, float), weights=np.asarray(weight, float)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-level", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.case_level, low_memory=False)
    counts = data.groupby("subjectid")["caseid"].nunique()
    single = counts.index[counts.eq(1)]
    data = data.loc[data.adult & data.reference_observable & data.subjectid.isin(single)].reset_index(drop=True)
    data["y"] = data.creatinine_event_168h.astype(int)
    data["R"] = data.dense_reference.astype(int)

    numeric_candidates = [
        "age", "height", "weight", "bmi", "asa", "preop_hb", "preop_plt", "preop_na", "preop_k",
        "preop_gluc", "preop_alb", "preop_bun", "preop_cr", "baseline_cr", "intraop_ebl", "intraop_uo",
        "intraop_rbc", "intraop_ffp", "intraop_crystalloid", "intraop_colloid",
    ]
    categorical_candidates = ["sex", "emop", "department", "optype", "approach", "ane_type", "preop_htn", "preop_dm"]
    numeric = [name for name in numeric_candidates if name in data and data[name].notna().any()]
    categorical = [name for name in categorical_candidates if name in data and data[name].notna().any()]
    features = numeric + categorical

    propensity = np.full(len(data), np.nan)
    outcome_probability = np.full(len(data), np.nan)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train, test in folds.split(data, data.R):
        observation_model = pipeline(numeric, categorical)
        observation_model.fit(data.iloc[train][features], data.R.iloc[train])
        propensity[test] = observation_model.predict_proba(data.iloc[test][features])[:, 1]
        dense_train = train[data.R.iloc[train].to_numpy() == 1]
        outcome_model = pipeline(numeric, categorical)
        outcome_model.fit(data.iloc[dense_train][features], data.y.iloc[dense_train])
        outcome_probability[test] = outcome_model.predict_proba(data.iloc[test][features])[:, 1]

    propensity = np.clip(propensity, 0.01, 0.99)
    outcome_probability = np.clip(outcome_probability, 1e-5, 1 - 1e-5)
    observed = data.R.to_numpy(bool)
    y = data.y.to_numpy(float)
    raw_weight = 1 / propensity[observed]
    truncation = np.quantile(raw_weight, 0.99)
    truncated_weight = np.minimum(raw_weight, truncation)
    truth = float(y.mean())
    naive = float(y[observed].mean())
    ipaw_raw = weighted_rate(y[observed], raw_weight)
    ipaw_truncated = weighted_rate(y[observed], truncated_weight)
    aipw_scores = outcome_probability + observed * (y - outcome_probability) / propensity
    aipw = float(aipw_scores.mean())

    rows = [
        {"method": "full observable operational reference", "event_rate": truth, "bias_vs_full_observable": 0.0},
        {"method": "naive dense subset", "event_rate": naive, "bias_vs_full_observable": naive - truth},
        {"method": "IPAW raw", "event_rate": ipaw_raw, "bias_vs_full_observable": ipaw_raw - truth},
        {"method": "IPAW truncated 99th percentile", "event_rate": ipaw_truncated, "bias_vs_full_observable": ipaw_truncated - truth},
        {"method": "AIPW", "event_rate": aipw, "bias_vs_full_observable": aipw - truth},
    ]
    result = pd.DataFrame(rows)

    subgroup_rows = []
    age_group = pd.qcut(data.age, q=4, duplicates="drop").astype(str)
    for variable, series in (("sex", data.sex.astype(str)), ("age_quartile", age_group), ("optype", data.optype.astype(str))):
        for level, index in series.groupby(series).groups.items():
            take = np.asarray(list(index), dtype=int)
            subgroup_rows.append(
                {
                    "variable": variable,
                    "level": level,
                    "n": len(take),
                    "dense_fraction": float(data.R.iloc[take].mean()),
                    "event_rate_full_observable": float(data.y.iloc[take].mean()),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "Table_vitaldb_dense_selection_correction.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.output_dir / "Table_vitaldb_observability_by_subgroup.csv", index=False)
    audit = {
        "n_full_observable_single_operation_adults": int(len(data)),
        "events_full_observable": int(y.sum()),
        "n_dense": int(observed.sum()),
        "events_dense": int(y[observed].sum()),
        "full_observable_event_rate": truth,
        "dense_event_rate": naive,
        "propensity_min": float(propensity.min()),
        "propensity_p01": float(np.quantile(propensity, 0.01)),
        "propensity_p99": float(np.quantile(propensity, 0.99)),
        "propensity_max": float(propensity.max()),
        "raw_weight_p99": float(np.quantile(raw_weight, 0.99)),
        "raw_weight_max": float(raw_weight.max()),
        "truncated_weight_ess": float(truncated_weight.sum() ** 2 / np.square(truncated_weight).sum()),
        "observation_model_auc": float(roc_auc_score(data.R, propensity)),
        "outcome_model_auc_in_full_observable": float(roc_auc_score(data.y, outcome_probability)),
        "outcome_model_brier_in_full_observable": float(brier_score_loss(data.y, outcome_probability)),
        "estimand": "event rate in adults with an observable creatinine operational reference and exactly one recorded operation",
        "limitation": "This positive-control audit targets the observed operational endpoint; it cannot recover latent AKI missed by sparse creatinine measurement.",
        "predictors": features,
    }
    (args.output_dir / "VITALDB_DENSE_SELECTION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
