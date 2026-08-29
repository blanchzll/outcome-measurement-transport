# %% [markdown]
# # Precision, hierarchical calibration, clinical utility, fairness, and portability

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

BASE = Path(str(_release_path('source')))
ROOT = BASE / "ascertainment_framework_20260826"
SECURE, TABLES, OUTPUTS = ROOT / "secure_work", ROOT / "tables", ROOT / "outputs"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "code"))
from analysis import CENTER, TARGET  # noqa: E402
from loco_analysis import bootstrap_metric_ci, probability_metrics  # noqa: E402
from ascertainment_stress import weighted_metrics  # noqa: E402

RNG_SEED = 20260826


def parse_args():
    parser=argparse.ArgumentParser();parser.add_argument("--source-cohort",choices=["3710","4014"],default="4014")
    return parser.parse_args()


def net_benefit(y, p, threshold):
    y, p = np.asarray(y, int), np.asarray(p, float)
    selected = p >= threshold
    tp = np.sum(selected & (y == 1)); fp = np.sum(selected & (y == 0))
    return float(tp / len(y) - fp / len(y) * threshold / (1 - threshold))


def stratified_indices(frame, group_cols, rng):
    return np.concatenate([
        rng.choice(index.to_numpy(), len(index), replace=True)
        for _, index in frame.groupby(group_cols, observed=True).groups.items()
    ])


def calibration_fit_se(y, p):
    y, z = np.asarray(y, float), logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    X = np.column_stack([np.ones(len(z)), z])
    def obj(beta):
        eta = X @ beta
        return -np.sum(y * eta - np.logaddexp(0, eta))
    fit = minimize(obj, np.array([0.0, 1.0]), method="BFGS")
    beta = fit.x
    q = expit(X @ beta)
    information = X.T @ (X * (q * (1 - q))[:, None])
    cov = np.linalg.pinv(information)
    return beta, np.sqrt(np.diag(cov))


def random_effects_meta(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    w = 1 / se**2; fixed = np.sum(w * est) / np.sum(w)
    q = np.sum(w * (est - fixed)**2); df = max(len(est) - 1, 1)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (se**2 + tau2); pooled = np.sum(wr * est) / np.sum(wr)
    pooled_se = np.sqrt(1 / np.sum(wr))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    pred_half = 1.96 * np.sqrt(tau2 + pooled_se**2)
    return {"k": len(est), "pooled": pooled, "se": pooled_se, "ci_lower": pooled - 1.96*pooled_se,
            "ci_upper": pooled + 1.96*pooled_se, "tau2": tau2, "I2": i2,
            "prediction_lower": pooled - pred_half, "prediction_upper": pooled + pred_half,
            "method": "DerSimonian-Laird exploratory; only five source centers"}


def subgroup_rows(database, data, ycol, pcol, group_specs):
    rows = []
    y, p = data[ycol].astype(int), data[pcol].astype(float)
    threshold = float(p.quantile(.80))
    for variable, groups in group_specs.items():
        for label, mask in groups.items():
            sub = data.loc[mask & y.notna() & p.notna()]
            if len(sub) == 0:
                continue
            yy, pp = sub[ycol].astype(int).to_numpy(), sub[pcol].to_numpy(float)
            m = weighted_metrics(yy, pp)
            selected = pp >= threshold
            tpr = float(np.sum(selected & (yy == 1)) / max(np.sum(yy == 1), 1))
            fpr = float(np.sum(selected & (yy == 0)) / max(np.sum(yy == 0), 1))
            rows.append({"database": database, "group_variable": variable, "group": str(label),
                         "inference_status": "estimable" if (len(sub) >= 100 and yy.sum() >= 20) else "descriptive_low_information",
                         "global_top20_threshold": threshold, "tpr": tpr, "fpr": fpr, **m})
    return rows


def utility_rows(database, y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    rows = []
    prevalence = y.mean()
    for t in np.arange(.02, .151, .01):
        selected = p >= t
        rows.append({"database": database, "policy": "risk_threshold", "policy_value": round(float(t), 3),
                     "selected_n": int(selected.sum()), "selected_fraction": float(selected.mean()),
                     "event_capture": float(y[selected].sum() / max(y.sum(), 1)),
                     "ppv": float(y[selected].mean()) if selected.any() else np.nan,
                     "false_alerts": int(np.sum(selected & (y == 0))),
                     "net_benefit": net_benefit(y, p, t),
                     "net_benefit_all": float(prevalence - (1-prevalence)*t/(1-t))})
    for frac in [.10, .20, .30, .40]:
        threshold = np.quantile(p, 1-frac); selected = p >= threshold
        for tests in [1, 2, 3]:
            rows.append({"database": database, "policy": "top_fraction", "policy_value": frac,
                         "additional_tests_per_selected": tests, "selected_n": int(selected.sum()),
                         "selected_fraction": float(selected.mean()),
                         "event_capture": float(y[selected].sum() / max(y.sum(), 1)),
                         "ppv": float(y[selected].mean()), "false_alerts": int(np.sum(selected & (y == 0))),
                         "additional_tests": int(selected.sum()*tests),
                         "tests_per_event_captured": float(selected.sum()*tests / max(y[selected].sum(), 1)),
                         "net_benefit": net_benefit(y, p, threshold), "net_benefit_all": np.nan})
    return rows


# %%
args=parse_args();source_label=f"source_{args.source_cohort}"
source = pd.read_csv(SECURE / f"SOURCE_{args.source_cohort}_LOCKED_LOCO_PREDICTIONS_SECURE.csv.gz", low_memory=False)
y = source[TARGET].astype(int).to_numpy(); centers = source[CENTER].astype(int).to_numpy()

precision = []
center_metrics = []
frontier = []
for fs in ["P", "PI", "H"]:
    feature_count = {"P": 10, "PI": 13, "H": 9}[fs]
    for model in ["ridge", "restricted_rf", "gradient_boosting", "soft_voting"]:
        col = f"pred_{fs}_{model}"; p = source[col].to_numpy(float)
        met = probability_metrics(y, p)
        ci = bootstrap_metric_ci(y, p, n_bootstrap=1000, seed=RNG_SEED + len(precision), groups=centers)
        row = {"database": source_label, "feature_set": fs, "model": model, **met}
        for metric, (lo, hi) in ci.items():
            row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = lo, hi
        precision.append(row)
        cm = []
        for center in sorted(np.unique(centers)):
            mask = centers == center; m = probability_metrics(y[mask], p[mask])
            center_metrics.append({"center": center, "feature_set": fs, "model": model, **m})
            cm.append(m)
        frontier.append({"database": source_label, "feature_set": fs, "model": model,
                         "feature_count": feature_count, "pooled_auc": met["roc_auc"], "pooled_brier": met["brier"],
                         "pooled_abs_citl": abs(met["calibration_in_the_large"]),
                         "worst_center_auc": np.nanmin([m["roc_auc"] for m in cm]),
                         "worst_center_abs_citl": np.nanmax([abs(m["calibration_in_the_large"]) for m in cm])})

pd.DataFrame(precision).to_csv(TABLES / "Table_source_model_precision.csv", index=False)
pd.DataFrame(center_metrics).to_csv(TABLES / "Table_source_center_performance_complete.csv", index=False)

# Hierarchical calibration for the prespecified perioperative restricted RF.
p_primary = source.pred_PI_restricted_rf.to_numpy(float)
cal_rows = []
for center in sorted(np.unique(centers)):
    mask = centers == center
    beta, se = calibration_fit_se(y[mask], p_primary[mask])
    cal_rows.append({"center": center, "n": int(mask.sum()), "events": int(y[mask].sum()),
                     "intercept": beta[0], "intercept_se": se[0], "slope": beta[1], "slope_se": se[1]})
cal = pd.DataFrame(cal_rows)
cal.to_csv(TABLES / "Table_source_hierarchical_calibration_centers.csv", index=False)
hier = []
for metric in ["intercept", "slope"]:
    hier.append({"metric": metric, **random_effects_meta(cal[metric], cal[f"{metric}_se"])})
pd.DataFrame(hier).to_csv(TABLES / "Table_source_hierarchical_calibration_meta.csv", index=False)

# Incremental value: paired analytic-record bootstrap within center, PI vs P.
increment = []
rng = np.random.default_rng(RNG_SEED)
frame_for_boot = pd.DataFrame({"center": centers, "y": y})
for model in ["ridge", "restricted_rf", "gradient_boosting", "soft_voting"]:
    p0 = source[f"pred_P_{model}"].to_numpy(float); p1 = source[f"pred_PI_{model}"].to_numpy(float)
    observed = {"auc_difference": roc_auc_score(y,p1)-roc_auc_score(y,p0),
                "brier_difference": np.mean((y-p1)**2)-np.mean((y-p0)**2),
                "net_benefit_difference_at_5pct": net_benefit(y,p1,.05)-net_benefit(y,p0,.05)}
    draws = {k: [] for k in observed}
    for _ in range(1000):
        idx = stratified_indices(frame_for_boot, ["center"], rng)
        draws["auc_difference"].append(roc_auc_score(y[idx],p1[idx])-roc_auc_score(y[idx],p0[idx]))
        draws["brier_difference"].append(np.mean((y[idx]-p1[idx])**2)-np.mean((y[idx]-p0[idx])**2))
        draws["net_benefit_difference_at_5pct"].append(net_benefit(y[idx],p1[idx],.05)-net_benefit(y[idx],p0[idx],.05))
    for metric, value in observed.items():
        increment.append({"model": model, "metric": metric, "estimate": value,
                          "ci_lower": np.quantile(draws[metric],.025), "ci_upper": np.quantile(draws[metric],.975),
                          "bootstrap": "paired analytic-record bootstrap within center, 1000 replicates; locked predictions"})
pd.DataFrame(increment).to_csv(TABLES / "Table_preop_to_perioperative_increment.csv", index=False)

# INSPIRE and MIMIC performance/test-burden/fairness use their operational full references.
inspire = pd.read_csv(SECURE / "INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz")
inspire = inspire.loc[inspire.dense_reference.eq(1)].copy()
inspire["age_group"] = pd.cut(inspire.Age, [-np.inf,64,74,np.inf], labels=["<65","65-74","75+"])
inspire["sex_group"] = inspire.Gender.astype(str)
inspire["site_group"] = inspire.cancer_site_label.astype(str)
inspire["approach_group"] = inspire.approach_character.astype(str)
inspire_groups = {
    "sex": {x: inspire.sex_group.eq(x) for x in inspire.sex_group.dropna().unique()},
    "age": {x: inspire.age_group.eq(x) for x in inspire.age_group.dropna().unique()},
    "cancer_site": {x: inspire.site_group.eq(x) for x in inspire.site_group.dropna().unique()},
    "surgical_approach": {x: inspire.approach_group.eq(x) for x in inspire.approach_group.dropna().unique()},
}

source["age_group"] = pd.cut(source.Age, [-np.inf,64,74,np.inf], labels=["<65","65-74","75+"])
source_groups = {
    "sex": {x: source.Gender.astype(str).eq(x) for x in source.Gender.astype(str).dropna().unique()},
    "age": {x: source.age_group.eq(x) for x in source.age_group.dropna().unique()},
    "cancer_site": {x: source.Gastrocolorectal.astype(str).eq(x) for x in source.Gastrocolorectal.astype(str).dropna().unique()},
    "surgical_approach": {x: source.SurgicalApproach.astype(str).eq(x) for x in source.SurgicalApproach.astype(str).dropna().unique()},
}
fairness = subgroup_rows(source_label, source, TARGET, "pred_PI_restricted_rf", source_groups)
fairness += subgroup_rows("INSPIRE_dense", inspire, "full168_creatinine_aki", "restricted_rf_probability", inspire_groups)
pd.DataFrame(fairness).to_csv(TABLES / "Table_fairness_representativeness_audit.csv", index=False)

# Load database-native temporal MIMIC test predictions reproducibly.
spec = importlib.util.spec_from_file_location("sim52", ROOT / "code" / "52_measurement_deletion_simulation.py")
sim52 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim52)
mimic, _ = sim52.prepare_mimic()

utility = utility_rows(source_label, y, p_primary)
utility += utility_rows("INSPIRE_dense", inspire.full168_creatinine_aki.astype(int), inspire.restricted_rf_probability)
utility += utility_rows("MIMIC_temporal_test", mimic.y_full.astype(int), mimic.risk)
pd.DataFrame(utility).to_csv(TABLES / "Table_monitoring_threshold_burden_capture.csv", index=False)

for db, yy, pp, role in [
    ("INSPIRE_dense", inspire.full168_creatinine_aki.astype(int), inspire.restricted_rf_probability, "harmonized transport test"),
    ("MIMIC_temporal_test", mimic.y_full.astype(int), mimic.risk, "database-native methodological replication"),
]:
    m = weighted_metrics(yy, pp)
    frontier.append({"database": db, "feature_set": "H" if "INSPIRE" in db else "native",
                     "model": "restricted_rf" if "INSPIRE" in db else "ridge", "feature_count": 9 if "INSPIRE" in db else 15,
                     "pooled_auc": m["auc"], "pooled_brier": m["brier"],
                     "pooled_abs_citl": abs(m["calibration_intercept"]), "worst_center_auc": np.nan,
                     "worst_center_abs_citl": np.nan, "role": role})
pd.DataFrame(frontier).to_csv(TABLES / "Table_portability_performance_frontier.csv", index=False)

audit = {"source_cohort":args.source_cohort,"source_n": len(source), "source_events": int(y.sum()), "bootstrap_precision": 1000,
         "hierarchical_centers": int(len(np.unique(centers))),
         "fairness_claim": "representativeness/performance audit only; no fairness certification",
         "utility_claim": "decision-analytic scenario analysis only; not an observed clinical-impact study",
         "mimic_role": "database-native methodological replication, not same-model clinical external validation"}
(OUTPUTS / "PRECISION_UTILITY_FAIRNESS_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
