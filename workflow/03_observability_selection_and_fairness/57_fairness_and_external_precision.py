# %% [markdown]
# # Bootstrap precision for public references and subgroup disparity audit

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(str(_release_path('source')))
ROOT = BASE / "ascertainment_framework_20260826"
SECURE, TABLES, OUTPUTS = ROOT / "secure_work", ROOT / "tables", ROOT / "outputs"
sys.path.insert(0, str(ROOT / "code"))
from ascertainment_stress import weighted_metrics  # noqa: E402

N_BOOT = 500
SEED = 20260826


def parse_args():
    parser=argparse.ArgumentParser();parser.add_argument("--source-cohort",choices=["3710","4014"],default="4014")
    return parser.parse_args()


def analytic_record_draw(y, rng):
    y = np.asarray(y, int)
    return rng.choice(np.arange(len(y)), len(y), replace=True)


def metric_values(y, p, threshold):
    m = weighted_metrics(y, p)
    selected = np.asarray(p) >= threshold; y = np.asarray(y, int)
    m["tpr_top20"] = np.sum(selected & (y==1)) / max(np.sum(y==1),1)
    m["fpr_top20"] = np.sum(selected & (y==0)) / max(np.sum(y==0),1)
    return m


def audit_dataset(database, d, ycol, pcol, group_columns):
    rows, disparities = [], []
    threshold = float(d[pcol].quantile(.80))
    rng = np.random.default_rng(SEED + len(database))
    for variable, col in group_columns.items():
        usable = d[[ycol,pcol,col]].dropna().copy()
        group_stats = usable.groupby(col, observed=False)[ycol].agg(["size","sum"])
        valid_groups = group_stats.index[(group_stats["size"]>=100)&(group_stats["sum"]>=20)&((group_stats["size"]-group_stats["sum"])>=20)].tolist()
        for group, sub in usable.groupby(col, observed=False):
            y, p = sub[ycol].astype(int).to_numpy(), sub[pcol].to_numpy(float)
            m = metric_values(y,p,threshold)
            draw_metrics = {x: [] for x in ["auc","oe","brier","calibration_slope","tpr_top20","fpr_top20"]}
            if group in valid_groups:
                for _ in range(N_BOOT):
                    idx = analytic_record_draw(y,rng); bm = metric_values(y[idx],p[idx],threshold)
                    for x in draw_metrics: draw_metrics[x].append(bm[x])
            for metric in draw_metrics:
                vals = np.asarray(draw_metrics[metric],float)
                rows.append({"database":database,"group_variable":variable,"group":str(group),
                             "n":len(sub),"events":int(y.sum()),"metric":metric,"estimate":m[metric],
                             "ci_lower":np.nanquantile(vals,.025) if len(vals) else np.nan,
                             "ci_upper":np.nanquantile(vals,.975) if len(vals) else np.nan,
                             "inference_status":"bootstrap_500" if len(vals) else "descriptive_low_information"})
        if len(valid_groups)>=2:
            point = {}
            for group in valid_groups:
                sub=usable.loc[usable[col].eq(group)]; point[group]=metric_values(sub[ycol].astype(int),sub[pcol],threshold)
            boot = {x:[] for x in ["auc","oe","brier","calibration_slope","tpr_top20","fpr_top20"]}
            for _ in range(N_BOOT):
                vals={x:[] for x in boot}
                for group in valid_groups:
                    sub=usable.loc[usable[col].eq(group)]; yy=sub[ycol].astype(int).to_numpy(); pp=sub[pcol].to_numpy(float)
                    idx=analytic_record_draw(yy,rng); bm=metric_values(yy[idx],pp[idx],threshold)
                    for x in vals: vals[x].append(bm[x])
                for x in boot: boot[x].append(np.nanmax(vals[x])-np.nanmin(vals[x]))
            for metric in boot:
                point_gap=np.nanmax([point[g][metric] for g in valid_groups])-np.nanmin([point[g][metric] for g in valid_groups])
                disparities.append({"database":database,"group_variable":variable,"metric":metric,
                                    "groups_included":"|".join(map(str,valid_groups)),"max_minus_min":point_gap,
                                    "ci_lower":np.nanquantile(boot[metric],.025),"ci_upper":np.nanquantile(boot[metric],.975),
                                    "interpretation":"descriptive performance disparity; not a fairness test"})
    return rows, disparities


args=parse_args();source_label=f"source_{args.source_cohort}"
source=pd.read_csv(SECURE/f"SOURCE_{args.source_cohort}_LOCKED_LOCO_PREDICTIONS_SECURE.csv.gz",low_memory=False)
source["age_group"]=pd.cut(source.Age,[-np.inf,64,74,np.inf],labels=["<65","65-74","75+"])
source["sex_group"]=source.Gender.astype(str);source["site_group"]=source.Gastrocolorectal.astype(str);source["approach_group"]=source.SurgicalApproach.astype(str)
inspire=pd.read_csv(SECURE/"INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz")
inspire=inspire.loc[inspire.dense_reference.eq(1)].copy()
inspire["age_group"]=pd.cut(inspire.Age,[-np.inf,64,74,np.inf],labels=["<65","65-74","75+"])
inspire["sex_group"]=inspire.Gender.astype(str);inspire["site_group"]=inspire.cancer_site_label.astype(str);inspire["approach_group"]=inspire.approach_character.astype(str)

groups={"sex":"sex_group","age":"age_group","cancer_site":"site_group","surgical_approach":"approach_group"}
r1,d1=audit_dataset(source_label,source,"PostopAKI","pred_PI_restricted_rf",groups)
r2,d2=audit_dataset("INSPIRE_dense",inspire,"full168_creatinine_aki","restricted_rf_probability",groups)
pd.DataFrame(r1+r2).to_csv(TABLES/"Table_fairness_bootstrap_intervals.csv",index=False)
pd.DataFrame(d1+d2).to_csv(TABLES/"Table_fairness_max_min_disparities.csv",index=False)

# External-reference precision using 1000 ordinary analytic-record bootstrap draws.
spec=importlib.util.spec_from_file_location("sim52",ROOT/"code"/"52_measurement_deletion_simulation.py")
sim52=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim52)
mimic,_=sim52.prepare_mimic()
precision=[];rng=np.random.default_rng(SEED)
for database,yy,pp in [("INSPIRE_dense",inspire.full168_creatinine_aki.astype(int).to_numpy(),inspire.restricted_rf_probability.to_numpy()),
                       ("MIMIC_temporal_test",mimic.y_full.astype(int).to_numpy(),mimic.risk.to_numpy())]:
    point=weighted_metrics(yy,pp); draws={x:[] for x in ["auc","oe","brier","calibration_intercept","calibration_slope"]}
    for _ in range(1000):
        idx=analytic_record_draw(yy,rng); bm=weighted_metrics(yy[idx],pp[idx])
        for x in draws: draws[x].append(bm[x])
    for metric,vals in draws.items():
        precision.append({"database":database,"n":len(yy),"events":int(yy.sum()),"metric":metric,
                          "estimate":point[metric],"ci_lower":np.quantile(vals,.025),"ci_upper":np.quantile(vals,.975),
                          "bootstrap":"ordinary analytic-record bootstrap, 1000 replicates; locked predictions"})
pd.DataFrame(precision).to_csv(TABLES/"Table_public_reference_precision.csv",index=False)

audit={"source_cohort":args.source_cohort,"subgroup_bootstrap":N_BOOT,"external_precision_bootstrap":1000,
       "minimum_inferential_group":"n>=100, events>=20, non-events>=20",
       "multiplicity":"no hypothesis-test p values; descriptive intervals only",
       "claim":"representativeness/performance audit, not algorithmic fairness certification"}
(OUTPUTS/"FAIRNESS_EXTERNAL_PRECISION_AUDIT.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps(audit,indent=2))
