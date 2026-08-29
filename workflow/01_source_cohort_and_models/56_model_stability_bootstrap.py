# %% [markdown]
# # Two-hundred-refit stability audit of the frozen perioperative models

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

BASE = Path(str(_release_path('source')))
ROOT = BASE / "ascertainment_framework_20260826"
SECURE, TABLES, OUTPUTS = ROOT / "secure_work", ROOT / "tables", ROOT / "outputs"
LOCK = BASE / "outputs_loco_3710_corrected_v2_20260823/model_lock.json"
sys.path.insert(0, str(BASE))
from analysis import CENTER, TARGET  # noqa: E402
from loco_analysis import BASE_MODELS, FEATURE_SET_SPECS, build_loco_search, probability_metrics  # noqa: E402

N_BOOT = 200
SEED = 20260826


def parse_args():
    parser=argparse.ArgumentParser();parser.add_argument("--source-cohort",choices=["3710","4014"],default="4014")
    return parser.parse_args()


def value(x):
    try: return ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError): return x


def modal_params(lock, model):
    items = []
    for row in lock["fold_locks"]:
        if row["feature_set"] == "PI" and row["model"] == model:
            items.append(tuple(sorted((k, value(v)) for k, v in row["best_params"].items())))
    return dict(Counter(items).most_common(1)[0][0])


def strata_bootstrap(frame, rng):
    chosen = []
    for _, idx in frame.groupby([CENTER], observed=True).groups.items():
        arr = idx.to_numpy(); chosen.extend(rng.choice(arr, len(arr), replace=True).tolist())
    return np.asarray(chosen, int)


args=parse_args()
lock_path=(BASE/"outputs_evidence_closure_20260823"/"loco_4014_corrected_formal"/"model_lock.json") if args.source_cohort=="4014" else LOCK
lock = json.loads(lock_path.read_text())
d = pd.read_csv(SECURE / f"SOURCE_{args.source_cohort}_LOCKED_LOCO_PREDICTIONS_SECURE.csv.gz", low_memory=False)
spec = FEATURE_SET_SPECS["PI"]
X, y = d[list(spec.features)], d[TARGET].astype(int).to_numpy()
rng = np.random.default_rng(SEED)
metric_rows, importance_rows = [], []

for model_name in BASE_MODELS:
    params = modal_params(lock, model_name)
    template = build_loco_search(spec, model_name, n_inner_centers=4, fast=True).estimator
    template.set_params(**params)
    full_model = clone(template).fit(X, y)
    full_p = full_model.predict_proba(X)[:, 1]
    full_top = set(np.flatnonzero(full_p >= np.quantile(full_p, .80)))
    for rep in range(N_BOOT):
        train_idx = strata_bootstrap(d, rng)
        present = np.zeros(len(d), bool); present[np.unique(train_idx)] = True
        oob = ~present
        fit = clone(template).fit(X.iloc[train_idx], y[train_idx])
        p_all = fit.predict_proba(X)[:, 1]
        if oob.sum() and np.unique(y[oob]).size == 2:
            metrics = probability_metrics(y[oob], p_all[oob])
        else:
            metrics = {k: np.nan for k in ["roc_auc","brier","oe_ratio","calibration_in_the_large","calibration_slope"]}
        top = set(np.flatnonzero(p_all >= np.quantile(p_all, .80)))
        metric_rows.append({"model": model_name, "replicate": rep, "n_train_draws": len(train_idx),
                            "n_unique_train": int(present.sum()), "n_oob": int(oob.sum()),
                            "risk_spearman_vs_full_fit": spearmanr(full_p, p_all).statistic,
                            "top20_jaccard_vs_full_fit": len(top & full_top) / len(top | full_top), **metrics})
        prep = fit.named_steps["preprocess"]
        names = prep.get_feature_names_out()
        estimator = fit.named_steps["model"]
        if hasattr(estimator, "coef_"):
            values = estimator.coef_[0]
            kind = "coefficient"
        elif hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
            kind = "feature_importance"
        else:
            continue
        for feature, score in zip(names, values):
            importance_rows.append({"model": model_name, "replicate": rep, "feature": feature,
                                    "quantity": kind, "value": float(score), "sign": int(np.sign(score))})

raw = pd.DataFrame(metric_rows)
raw.to_csv(SECURE / "SOURCE_MODEL_STABILITY_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
summary = []
for model, g in raw.groupby("model"):
    for metric in ["roc_auc","brier","oe_ratio","calibration_in_the_large","calibration_slope",
                   "risk_spearman_vs_full_fit","top20_jaccard_vs_full_fit"]:
        x = g[metric].dropna()
        summary.append({"model": model, "metric": metric, "n_replicates": len(x), "mean": x.mean(),
                        "sd": x.std(ddof=1), "q025": x.quantile(.025), "q50": x.quantile(.5), "q975": x.quantile(.975)})
pd.DataFrame(summary).to_csv(TABLES / "Table_source_model_stability_200bootstrap.csv", index=False)

if importance_rows:
    imp = pd.DataFrame(importance_rows)
    imp.to_csv(SECURE / "SOURCE_FEATURE_STABILITY_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
    agg = imp.groupby(["model","feature","quantity"]).agg(
        mean_value=("value","mean"), sd_value=("value","std"),
        positive_fraction=("sign",lambda x: np.mean(x>0)), negative_fraction=("sign",lambda x: np.mean(x<0)),
    ).reset_index()
    agg.to_csv(TABLES / "Table_source_feature_stability_200bootstrap.csv", index=False)

audit = {"source_cohort":args.source_cohort,"bootstrap_refits": N_BOOT, "feature_set": "PI", "models": list(BASE_MODELS),
         "resampling": "analytic-record bootstrap within center; event counts allowed to vary",
         "performance_evaluation": "out-of-bootstrap observations",
         "inference_scope": "refit stability audit, not a full model-development confidence interval; frozen modal hyperparameters",
         "stability_targets": ["OOB performance", "full-cohort risk rank", "top-20% overlap", "ridge/RF feature stability"]}
(OUTPUTS / "SOURCE_MODEL_STABILITY_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
