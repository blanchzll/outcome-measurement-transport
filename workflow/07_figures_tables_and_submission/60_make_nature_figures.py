# %% [markdown]
# # Nature-style quantitative figures
# Each panel is exported independently as editable SVG, vector PDF and 600-dpi TIFF.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(str(_release_path('analysis')))
TABLES, FIGURES = ROOT / "tables", ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
mpl.rcParams.update({"svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.labelsize":7,"axes.titlesize":8,
                     "xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6.5,
                     "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":.7,
                     "legend.frameon":False,"savefig.bbox":"tight"})

BLUE="#0072B2"; SKY="#56B4E9"; GREEN="#009E73"; ORANGE="#E69F00"; VERMILLION="#D55E00"
PURPLE="#CC79A7"; GREY="#8A8A8A"; LIGHT="#D9D9D9"; BLACK="#272727"
METHOD_COLORS={"naive":GREY,"IPAW_design_probability_untruncated":BLUE,
               "IPAW_design_probability_truncated99":PURPLE,"AIPW_design_probability":SKY,
               "recalibration_intercept_truth":ORANGE,"recalibration_intercept_slope_truth":VERMILLION,
               "reference_10pct_recalibration":GREEN,"Gamma2_prediction_sensitivity_region":PURPLE}


def save_panel(fig, folder, name, source):
    out=FIGURES/folder;out.mkdir(parents=True,exist_ok=True)
    source.to_csv(out/f"{name}_source_data.csv",index=False)
    fig.savefig(out/f"{name}.svg")
    fig.savefig(out/f"{name}.pdf")
    fig.savefig(out/f"{name}.tiff",dpi=600)
    plt.close(fig)


def heatmap_panel(data,index,columns,value,title,cbar_label,folder,name,vcenter=0):
    data=data.copy()
    if index=="mechanism":
        data[index]=pd.Categorical(data[index],categories=["MCAR","stratum_MAR","risk_MAR","history_MAR","outcome_MNAR","mixed_MNAR"],ordered=True)
    pivot=data.pivot(index=index,columns=columns,values=value)
    fig,ax=plt.subplots(figsize=(3.5,2.6))
    vmax=np.nanmax(np.abs(pivot.to_numpy())); vmax=max(vmax,1e-6)
    norm=mpl.colors.TwoSlopeNorm(vmin=-vmax,vcenter=vcenter,vmax=vmax)
    im=ax.imshow(pivot,cmap="RdBu_r",norm=norm,aspect="auto")
    ax.set_xticks(range(len(pivot.columns)),[str(x) for x in pivot.columns]);ax.set_yticks(range(len(pivot.index)),pivot.index)
    ax.set_xlabel("Target mean per-measurement retention");ax.set_ylabel("")
    ax.set_title(title,loc="left",fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            shown=0.0 if abs(pivot.iloc[i,j])<0.005 else pivot.iloc[i,j]
            ax.text(j,i,f"{shown:.2f}",ha="center",va="center",fontsize=5.5,
                    color="white" if abs(pivot.iloc[i,j])>.55*vmax else BLACK)
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.03);cb.set_label(cbar_label)
    save_panel(fig,folder,name,data)


# %% Figure 1
flow_i=pd.read_csv(TABLES/"Table_inspire_longitudinal_reference_flow.csv")
flow_m=pd.read_csv(TABLES/"Table_mimic_reference_flow.csv")
flow_e=pd.read_csv(ROOT/"eicu"/"tables"/"Table_eicu_reference_flow.csv")
flow=pd.concat([flow_i.assign(database="INSPIRE"),flow_m.assign(database="MIMIC"),
                flow_e.assign(database="EICU")],ignore_index=True)
fig,ax=plt.subplots(figsize=(4.5,4.2))
labels=[];values=[];colors=[]
for db,c in [("INSPIRE",BLUE),("MIMIC",ORANGE),("EICU",GREEN)]:
    g=flow[flow.database.eq(db)]
    for row in g.itertuples():
        labels.append(f"{db}: {row.stage}");values.append(row.n);colors.append(c)
ypos=np.arange(len(labels))[::-1]
ax.barh(ypos,values,color=colors,alpha=.85,height=.65)
ax.set_yticks(ypos,labels);ax.set_xlabel("Patients/operations");ax.set_title("Operational reference cohorts",loc="left",fontweight="bold")
for yv,n in zip(ypos,values):ax.text(n,yv,f" {n:,}",va="center",fontsize=6)
save_panel(fig,"Figure1_reference_observability","Figure1a_reference_flow",flow)

bal=pd.read_csv(TABLES/"Table_observability_predictor_imbalance.csv")
bal=bal[(bal.target=="two_slot")&(bal.level.fillna("").eq(""))].copy()
bal["max_abs"]=bal[["smd_before_vs_full","smd_after_vs_full"]].abs().max(axis=1)
bal=bal.nlargest(12,"max_abs").sort_values("max_abs")
fig,ax=plt.subplots(figsize=(3.5,3.0));yy=np.arange(len(bal))
ax.plot(bal.smd_before_vs_full,yy,"o",color=GREY,label="Before weighting")
ax.plot(bal.smd_after_vs_full,yy,"o",color=BLUE,label="After IPAW")
for yv,a,b in zip(yy,bal.smd_before_vs_full,bal.smd_after_vs_full):ax.plot([a,b],[yv,yv],color=LIGHT,lw=1)
ax.axvline(0,color=BLACK,lw=.7);ax.axvline(.1,color=VERMILLION,lw=.6,ls="--");ax.axvline(-.1,color=VERMILLION,lw=.6,ls="--")
ax.set_yticks(yy,bal.variable);ax.set_xlabel("Standardized difference vs full candidate cohort");ax.legend(loc="lower right")
ax.set_title("Observation-weight balance",loc="left",fontweight="bold")
save_panel(fig,"Figure1_reference_observability","Figure1b_ipaw_balance",bal)

density=pd.read_csv(TABLES/"Table_monitoring_density_event_gradient.csv")
fig,ax1=plt.subplots(figsize=(3.3,2.4));ax2=ax1.twinx()
ax1.plot(density.minimum_postoperative_creatinine_count,density.event_rate*100,"o-",color=VERMILLION,label="Detected event rate")
ax2.plot(density.minimum_postoperative_creatinine_count,density.n,"s--",color=BLUE,label="Eligible n")
ax1.set_xlabel("Minimum postoperative creatinine measurements");ax1.set_ylabel("Detected event rate (%)",color=VERMILLION)
ax2.set_ylabel("Eligible sample",color=BLUE);ax2.spines["right"].set_visible(True)
ax1.set_title("Detection rises with monitoring density",loc="left",fontweight="bold")
save_panel(fig,"Figure1_reference_observability","Figure1c_monitoring_gradient",density)

process = pd.DataFrame(
    [
        {"stage": 1, "object": "Retained longitudinal trajectory", "role": "Operational reference target"},
        {"stage": 2, "object": "Locally retained measurements", "role": "Observation process"},
        {"stage": 3, "object": "Reconstructed binary endpoint", "role": "Apparent evaluation target"},
        {"stage": 4, "object": "Fixed risk predictions", "role": "Evaluated against either target"},
    ]
)
fig, ax = plt.subplots(figsize=(5.8, 2.5))
ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.5); ax.axis("off")
boxes = [
    (0.2, 2.55, 2.8, 1.0, "Retained creatinine\ntrajectory", BLUE),
    (4.1, 2.55, 2.8, 1.0, "Measurements after\nlocal deletion", SKY),
    (8.0, 2.55, 2.8, 1.0, "Reconstructed\nendpoint", ORANGE),
    (4.1, 0.25, 2.8, 1.0, "Fixed risk\npredictions", GREEN),
]
for x, y, w, h, label, color in boxes:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=color,
                           edgecolor="none", alpha=.90)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", color="white",
            fontsize=6.3, fontweight="bold")
ax.annotate("", xy=(4.02, 3.05), xytext=(3.05, 3.05),
            arrowprops=dict(arrowstyle="->", lw=.9, color=BLACK))
ax.text(3.55, 3.42, "measurement\ndeletion", ha="center", va="center", fontsize=5.5)
ax.annotate("", xy=(7.92, 3.05), xytext=(6.95, 3.05),
            arrowprops=dict(arrowstyle="->", lw=.9, color=BLACK))
ax.text(7.45, 3.42, "endpoint\nreconstruction", ha="center", va="center", fontsize=5.5)
ax.annotate("", xy=(1.6, 2.50), xytext=(4.7, 1.28),
            arrowprops=dict(arrowstyle="->", lw=.9, color=BLUE))
ax.text(2.65, 1.55, "retained-reference\nevaluation", ha="center", va="center", color=BLUE, fontsize=5.5)
ax.annotate("", xy=(9.4, 2.50), xytext=(6.3, 1.28),
            arrowprops=dict(arrowstyle="->", lw=.9, color=VERMILLION))
ax.text(8.35, 1.55, "apparent-target\nevaluation", ha="center", va="center", color=VERMILLION, fontsize=5.5)
ax.set_title("One model, two outcome targets", loc="left", fontweight="bold")
save_panel(fig,"Figure1_reference_observability","Figure1d_estimand_schematic",process)

# %% Simulation figures
sim=pd.concat([pd.read_csv(TABLES/"Table_inspire_simulation_summary.csv"),
               pd.read_csv(TABLES/"Table_mimic_simulation_summary.csv"),
               pd.read_csv(TABLES/"Table_eicu_simulation_summary.csv")],ignore_index=True)
if sim.loc[~sim.method.str.startswith("reference_"),"n_replicates"].min()<300:
    raise RuntimeError("Production figures require 300 core replicates per condition")

for db in ["INSPIRE","MIMIC","EICU"]:
    d=sim[(sim.database==db)&(sim.method=="naive")&(sim.strength=="strong")&(sim.metric=="event_rate")]
    heatmap_panel(d,"mechanism","retention_target","bias",f"{db}: naive event-rate bias","Bias", "Figure2_deletion_mechanisms",f"Figure2a_{db.lower()}_event_bias")

method_labels={"naive":"Naive","IPAW_design_probability_untruncated":"Design-probability IPAW",
               "IPAW_design_probability_truncated99":"IPAW, 99th-percentile truncation",
               "AIPW_design_probability":"Augmented IPAW",
               "recalibration_intercept_truth":"Intercept recalibration",
               "recalibration_intercept_slope_truth":"Intercept+slope recalibration",
               "reference_10pct_recalibration":"10% reference recalibration"}
for db in ["INSPIRE","MIMIC","EICU"]:
    estimator_map={"naive":"Naive","IPAW_design_probability_untruncated":"IPAW untruncated",
                   "IPAW_design_probability_truncated99":"IPAW truncated","AIPW_design_probability":"AIPW"}
    d=sim[(sim.database==db)&(sim.mechanism=="mixed_MNAR")&(sim.strength=="strong")&(sim.metric=="oe")&sim.method.isin(estimator_map)].copy()
    d["method_label"]=pd.Categorical(d.method.map(estimator_map),categories=["Naive","IPAW untruncated","IPAW truncated","AIPW"],ordered=True)
    heatmap_panel(d,"method_label","retention_target","bias",f"{db}: original-model O/E recovery","Bias vs retained-reference O/E", "Figure2_deletion_mechanisms",f"Figure2b_{db.lower()}_method_bias")

d=sim[(sim.method=="full_reference")&(sim.metric=="reconstructed_sensitivity")]
agg=d.groupby(["database","retention_target"]).agg(mean=("mean","mean"),q025=("mean",lambda x:np.quantile(x,.025)),q975=("mean",lambda x:np.quantile(x,.975))).reset_index()
fig,ax=plt.subplots(figsize=(3.3,2.4))
for db,c,m in [("INSPIRE",BLUE,"o"),("MIMIC",ORANGE,"s"),("EICU",GREEN,"^")]:
    g=agg[agg.database.eq(db)];ax.plot(g.retention_target,g["mean"],marker=m,color=c,label=db);ax.fill_between(g.retention_target,g.q025,g.q975,color=c,alpha=.15)
ax.set_xlabel("Target mean per-measurement retention");ax.set_ylabel("Endpoint reconstruction sensitivity");ax.set_ylim(0,1);ax.legend()
ax.set_title("Endpoint recovery depends on testing",loc="left",fontweight="bold")
save_panel(fig,"Figure2_deletion_mechanisms","Figure2c_reconstruction_sensitivity",agg)

# %% Figure 3 correction strategies and target divergence
d=sim[(sim.mechanism=="mixed_MNAR")&(sim.strength=="strong")&(sim.metric=="oe")&sim.method.isin([
    "recalibration_intercept_slope_apparent","recalibration_intercept_slope_truth"])].copy()
d["target_label"]=d.method.map({"recalibration_intercept_slope_apparent":"Apparent endpoint","recalibration_intercept_slope_truth":"Retained reference"})
fig,ax=plt.subplots(figsize=(4.2,2.5))
for (db,target),g in d.groupby(["database","target_label"]):
    color={"INSPIRE":BLUE,"MIMIC":ORANGE,"EICU":GREEN}[db];ls="-" if target=="Apparent endpoint" else "--"
    ax.plot(g.retention_target,g["mean"],marker="o",color=color,ls=ls,label=f"{db}, {target}")
ax.axhline(1,color=BLACK,lw=.7);ax.set_xlabel("Target mean per-measurement retention");ax.set_ylabel("O/E after local recalibration");ax.legend(ncol=2)
ax.set_title("Held-out calibration depends on the endpoint target",loc="left",fontweight="bold")
save_panel(fig,"Figure3_correction_strategies","Figure3a_apparent_vs_reference_recalibration",d)

estimators=["naive","IPAW_design_probability_untruncated","IPAW_design_probability_truncated99","AIPW_design_probability","Gamma2_prediction_sensitivity_region"]
d=sim[(sim.strength=="strong")&(sim.metric=="event_rate")&sim.method.isin(estimators)].copy()
rmse=d.groupby(["database","method"]).rmse.mean().reset_index()
estimator_labels={"naive":"Naive","IPAW_design_probability_untruncated":"IPAW untruncated",
                  "IPAW_design_probability_truncated99":"IPAW truncated",
                  "AIPW_design_probability":"AIPW","Gamma2_prediction_sensitivity_region":"Gamma=2 midpoint"}
rmse["label"]=rmse.method.map(estimator_labels)
fig,ax=plt.subplots(figsize=(4.2,2.6));y=np.arange(len(estimators));width=.24
for i,(db,c) in enumerate([("INSPIRE",BLUE),("MIMIC",ORANGE),("EICU",GREEN)]):
    g=rmse[rmse.database.eq(db)].set_index("method").reindex(estimators)
    ax.barh(y+(i-1)*width,g.rmse,width,color=c,label=db)
ax.set_yticks(y,[estimator_labels[m] for m in estimators]);ax.invert_yaxis();ax.set_xlabel("Mean event-rate RMSE across mechanisms");ax.legend()
ax.set_title("Recovery of the retained-reference event rate",loc="left",fontweight="bold")
save_panel(fig,"Figure3_correction_strategies","Figure3b_strategy_rmse",rmse)

methods=["naive","IPAW_design_probability_untruncated","IPAW_design_probability_truncated99","AIPW_design_probability","recalibration_intercept_truth","recalibration_intercept_slope_truth","reference_10pct_recalibration"]
d=sim[(sim.metric=="oe")&sim.method.isin(methods)].copy()
i=d[d.database.eq("INSPIRE")];m=d[d.database.eq("MIMIC")]
keys=["retention_target","mechanism","strength","method"]
paired=i.merge(m,on=keys,suffixes=("_inspire","_mimic"))
fig,ax=plt.subplots(figsize=(3.0,2.7))
for method,g in paired.groupby("method"):
    ax.scatter(g.bias_inspire,g.bias_mimic,s=9,alpha=.6,color=METHOD_COLORS.get(method,GREY),label=method_labels.get(method,method))
ax.axhline(0,color=LIGHT,lw=.7);ax.axvline(0,color=LIGHT,lw=.7);ax.set_xlabel("INSPIRE O/E bias");ax.set_ylabel("MIMIC O/E bias")
ax.set_title("Independent mechanism replication",loc="left",fontweight="bold");ax.legend(bbox_to_anchor=(1.02,1),loc="upper left")
save_panel(fig,"Figure3_correction_strategies","Figure3c_cross_database_bias",paired)

recal_methods=["recalibration_intercept_truth","recalibration_intercept_slope_truth","reference_10pct_recalibration"]
for db,c in [("INSPIRE",BLUE),("MIMIC",ORANGE),("EICU",GREEN)]:
    d=sim[(sim.database==db)&(sim.mechanism=="mixed_MNAR")&(sim.strength=="strong")&(sim.metric=="oe")&sim.method.isin(recal_methods)].copy()
    fig,ax=plt.subplots(figsize=(3.5,2.5))
    for method,marker in zip(recal_methods,["o","s","^"]):
        g=d[d.method.eq(method)].sort_values("retention_target")
        ax.errorbar(g.retention_target,g["mean"],yerr=[g["mean"]-g.q025,g.q975-g["mean"]],marker=marker,capsize=2,label=method_labels[method])
    ax.axhline(1,color=BLACK,lw=.7,ls="--");ax.set_xlabel("Target mean per-measurement retention");ax.set_ylabel("O/E against retained reference");ax.legend()
    ax.set_title(f"{db}: recalibration target fidelity",loc="left",fontweight="bold")
    save_panel(fig,"Figure3_correction_strategies",f"Figure3d_{db.lower()}_recalibration_fidelity",d)

reference_methods=["reference_05pct_recalibration","reference_10pct_recalibration","reference_20pct_recalibration","reference_30pct_recalibration"]
d=sim[(sim.mechanism=="mixed_MNAR")&(sim.strength=="strong")&(sim.retention_target==.35)&
      (sim.metric=="oe")&sim.method.isin(reference_methods)].copy()
d["reference_fraction"]=d.method.str.extract(r"reference_(\d+)pct")[0].astype(float)/100
fig,ax=plt.subplots(figsize=(3.5,2.5))
for db,c,m in [("INSPIRE",BLUE,"o"),("MIMIC",ORANGE,"s"),("EICU",GREEN,"^")]:
    g=d[d.database.eq(db)].sort_values("reference_fraction")
    ax.errorbar(g.reference_fraction*100,g["mean"],yerr=[g["mean"]-g.q025,g.q975-g["mean"]],
                marker=m,color=c,capsize=2,label=db)
ax.axhline(1,color=BLACK,lw=.7,ls="--");ax.set_xlabel("Retained-reference sample (%)");ax.set_ylabel("Held-out O/E")
ax.set_title("Reference-sample size controls precision",loc="left",fontweight="bold");ax.legend()
save_panel(fig,"Figure3_correction_strategies","Figure3e_reference_sample_design",d)

selection=pd.concat([pd.read_csv(TABLES/"Table_inspire_pure_label_selection_control.csv"),
                     pd.read_csv(TABLES/"Table_mimic_pure_label_selection_control.csv"),
                     pd.read_csv(TABLES/"Table_eicu_pure_label_selection_control.csv")],ignore_index=True)
d=selection[(selection.retention_target==.35)&(selection.strength=="strong")&
            selection.mechanism.isin(["risk_MAR","outcome_MNAR"])&(selection.metric=="event_rate")].copy()
fig,ax=plt.subplots(figsize=(5.4,2.5));positions=np.arange(6);width=.34
labels=[]
for database in ["INSPIRE","MIMIC","EICU"]:
    for mechanism in ["risk_MAR","outcome_MNAR"]:labels.append(f"{database}\n{mechanism}")
for offset,(method,label,color) in enumerate([("naive","Naive",GREY),("oracle_IPW_untruncated","Oracle IPW",BLUE)]):
    values=[]
    for database in ["INSPIRE","MIMIC","EICU"]:
        for mechanism in ["risk_MAR","outcome_MNAR"]:
            values.append(d[(d.database==database)&(d.mechanism==mechanism)&(d.method==method)].bias.iloc[0])
    ax.bar(positions+(offset-.5)*width,values,width,color=color,label=label)
ax.axhline(0,color=BLACK,lw=.7);ax.set_xticks(positions,labels);ax.set_ylabel("Event-rate bias")
ax.set_title("Pure label-selection positive control",loc="left",fontweight="bold");ax.legend()
save_panel(fig,"Figure2_deletion_mechanisms","Figure2d_pure_selection_control",d)

# %% Figure 4 stability and portability
stability=pd.read_csv(TABLES/"Table_source_model_stability_200bootstrap.csv")
d=stability[stability.metric.isin(["risk_spearman_vs_full_fit","top20_jaccard_vs_full_fit"])].copy()
model_order=["ridge","restricted_rf","gradient_boosting"]
fig,ax=plt.subplots(figsize=(3.5,2.5));x=np.arange(3);width=.34
for j,(metric,label,c) in enumerate([("risk_spearman_vs_full_fit","Risk-rank Spearman",BLUE),("top20_jaccard_vs_full_fit","Top-20% Jaccard",ORANGE)]):
    g=d[d.metric.eq(metric)].set_index("model").reindex(model_order)
    ax.bar(x+(j-.5)*width,g.q50,width,color=c,label=label)
    ax.errorbar(x+(j-.5)*width,g.q50,yerr=[g.q50-g.q025,g.q975-g.q50],fmt="none",ecolor=BLACK,lw=.7,capsize=2)
ax.set_xticks(x,["Ridge","Restricted RF","Gradient boosting"]);ax.set_ylim(0,1);ax.set_ylabel("Stability")
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2)
ax.set_title("Two-hundred-refit model stability",loc="left",fontweight="bold")
save_panel(fig,"Figure4_stability_portability","Figure4a_model_stability",d)

inc=pd.read_csv(TABLES/"Table_preop_to_perioperative_increment.csv");d=inc[inc.metric.eq("auc_difference")].copy()
fig,ax=plt.subplots(figsize=(3.2,2.3));yy=np.arange(len(d))[::-1]
ax.errorbar(d.estimate,yy,xerr=[d.estimate-d.ci_lower,d.ci_upper-d.estimate],fmt="o",color=BLUE,ecolor=BLUE,capsize=2)
ax.axvline(0,color=BLACK,lw=.7,ls="--");ax.set_yticks(yy,d.model.str.replace("_"," "));ax.set_xlabel("AUC difference: perioperative minus preoperative")
ax.set_title("Minimal incremental discrimination",loc="left",fontweight="bold")
save_panel(fig,"Figure4_stability_portability","Figure4b_perioperative_increment",d)

front=pd.read_csv(TABLES/"Table_portability_performance_frontier.csv");d=front[(front.database.eq("source_4014"))&~front.model.eq("soft_voting")].copy()
center=pd.read_csv(TABLES/"Table_source_center_performance_complete.csv")
center=center[center.events.ge(20)].copy()
port=(center.groupby(["feature_set","model"],as_index=False)
      .agg(worst_estimable_center_abs_citl=("calibration_in_the_large",lambda x:np.abs(x).max()),
           worst_estimable_center_auc=("roc_auc","min")))
d=d.merge(port,on=["feature_set","model"],how="left")
fig,ax=plt.subplots(figsize=(3.5,2.7))
for fs,marker in [("P","o"),("PI","s"),("H","^")]:
    g=d[d.feature_set.eq(fs)];ax.scatter(g.worst_estimable_center_abs_citl,g.pooled_auc,s=28,marker=marker,label=fs)
    for j,row in enumerate(g.itertuples()):
        label=row.model.replace("gradient_boosting","GB").replace("restricted_rf","RF").replace("ridge","Ridge")
        ax.annotate(label,(row.worst_estimable_center_abs_citl,row.pooled_auc),xytext=(3,4 if j%2==0 else -8),textcoords="offset points",fontsize=5)
ax.set_xlabel("Worst |CITL| among centers with ≥20 events");ax.set_ylabel("Pooled LOCO AUC");ax.legend(title="Feature set",loc="lower right")
ax.set_title("Pooled performance vs center calibration",loc="left",fontweight="bold")
save_panel(fig,"Figure4_stability_portability","Figure4c_portability_frontier",d)

# %% Figure 5 clinical utility and subgroup audit
util=pd.read_csv(TABLES/"Table_monitoring_threshold_burden_capture.csv")
d=util[(util.policy=="top_fraction")&(util.additional_tests_per_selected==1)].copy()
fig,ax=plt.subplots(figsize=(3.4,2.5))
for db,c,m in [("source_4014",BLACK,"o"),("INSPIRE_dense",BLUE,"s"),("MIMIC_temporal_test",ORANGE,"^")]:
    g=d[d.database.eq(db)].sort_values("selected_fraction");ax.plot(g.selected_fraction*100,g.event_capture*100,marker=m,color=c,label=db.replace("_"," "))
ax.plot([0,40],[0,40],color=LIGHT,ls="--",label="Random selection")
ax.set_xlabel("Patients allocated monitoring (%)");ax.set_ylabel("Events captured (%)");ax.legend()
ax.set_title("Monitoring allocation efficiency",loc="left",fontweight="bold")
save_panel(fig,"Figure5_clinical_audit","Figure5a_burden_event_capture",d)

fair=pd.read_csv(TABLES/"Table_fairness_bootstrap_intervals.csv")
d=fair[(fair.metric=="auc")&(fair.inference_status=="bootstrap_500")].copy()
def subgroup_label(row):
    db="INSPIRE dense" if row.database=="INSPIRE_dense" else "Source cohort"
    variable={"cancer_site":"cancer site","surgical_approach":"recorded approach code"}.get(row.group_variable,row.group_variable)
    group=str(row.group)
    if row.database=="source_4014" and row.group_variable=="cancer_site": group={"1":"gastric","2":"colorectal"}.get(group,group)
    if row.database=="source_4014" and row.group_variable=="surgical_approach":
        variable="surgical approach"
        group={"1.0":"open","2.0":"laparoscopic","3.0":"converted","4.0":"robotic"}.get(group,group)
    return f"{db}: {variable}={group}"
d["label"]=d.apply(subgroup_label,axis=1)
d=d.sort_values(["database","group_variable","estimate"])
fig,ax=plt.subplots(figsize=(4.0,max(3.0,.18*len(d))));yy=np.arange(len(d))[::-1]
colors=[BLUE if x=="INSPIRE_dense" else BLACK for x in d.database]
for yv,row,c in zip(yy,d.itertuples(),colors):ax.errorbar(row.estimate,yv,xerr=[[row.estimate-row.ci_lower],[row.ci_upper-row.estimate]],fmt="o",color=c,ecolor=c,capsize=1.5)
ax.set_yticks(yy,d.label);ax.set_xlabel("AUC (95% bootstrap interval)");ax.set_xlim(.45,1)
ax.set_title("Subgroup representativeness audit",loc="left",fontweight="bold")
save_panel(fig,"Figure5_clinical_audit","Figure5b_subgroup_auc",d)

# %% Supplementary Figure 6: eICU multicentre replication audit
eicu_hosp=pd.read_csv(TABLES/"Table_eicu_component_observability_by_hospital.csv")
fig,ax=plt.subplots(figsize=(3.6,2.7))
sizes=12+90*np.sqrt(eicu_hosp.n/eicu_hosp.n.max())
sc=ax.scatter(eicu_hosp.dense_creatinine*100,eicu_hosp.creatinine_event_rate*100,
              s=sizes,c=eicu_hosp.urine_output_observed*100,cmap="viridis",alpha=.85,
              edgecolor="white",linewidth=.3)
ax.set_xlabel("Dense creatinine reference available (%)");ax.set_ylabel("Creatinine-event rate (%)")
cb=fig.colorbar(sc,ax=ax,fraction=.04,pad=.03);cb.set_label("Urine-output observability (%)")
ax.set_title("eICU hospitals differ in outcome observability",loc="left",fontweight="bold")
save_panel(fig,"Figure6_eicu_replication","Figure6a_eicu_hospital_observability",eicu_hosp)

eicu_components=pd.read_csv(TABLES/"Table_eicu_kdigo_component_availability.csv")
row=eicu_components[eicu_components.population.eq("dense creatinine reference")].iloc[0]
component_plot=pd.DataFrame({
    "endpoint":["Creatinine","Urine-output proxy","RRT","Available-component union"],
    "events":[row.creatinine_events,row.urine_output_events,row.rrt_events,row.multicomponent_events],
    "n":[row.n]*4,
})
component_plot["event_rate"]=component_plot.events/component_plot.n
fig,ax=plt.subplots(figsize=(3.7,2.5));xx=np.arange(len(component_plot))
ax.bar(xx,component_plot.event_rate*100,color=[BLUE,SKY,PURPLE,GREEN])
ax.set_xticks(xx,["Creatinine","Urine-output\nproxy","RRT","Available-component\nunion"])
ax.set_ylabel("Event prevalence (%)")
for x,row in zip(xx,component_plot.itertuples()):ax.text(x,row.event_rate*100+.35,f"{int(row.events):,}",ha="center",fontsize=6)
ax.set_title("Endpoint components change the apparent target",loc="left",fontweight="bold")
save_panel(fig,"Figure6_eicu_replication","Figure6b_eicu_endpoint_components",component_plot)

with (ROOT/"outputs"/"EICU_KDIGO_COMPONENT_AUDIT.json").open() as handle:
    component_audit=json.load(handle)["test_set_metrics"]
metric_plot=pd.DataFrame([
    {"endpoint":"Creatinine","AUC":component_audit["creatinine_only"]["auc"],"O/E":component_audit["creatinine_only"]["oe"]},
    {"endpoint":"Available-component union","AUC":component_audit["available_component_union"]["auc"],"O/E":component_audit["available_component_union"]["oe"]},
])
fig,axes=plt.subplots(1,2,figsize=(4.5,2.35))
for ax,metric,ideal in [(axes[0],"AUC",.5),(axes[1],"O/E",1.0)]:
    ax.bar([0,1],metric_plot[metric],color=[BLUE,GREEN],width=.65)
    ax.axhline(ideal,color=BLACK,lw=.7,ls="--")
    ax.set_xticks([0,1],["Creatinine","Component\nunion"]);ax.set_ylabel(metric)
    for x,value in enumerate(metric_plot[metric]):ax.text(x,value+.015,f"{value:.3f}",ha="center",fontsize=6)
axes[0].set_ylim(.48,.68);axes[1].set_ylim(0,1.3)
fig.suptitle("The same predictions imply different performance after target change",x=.01,ha="left",fontweight="bold",fontsize=8)
save_panel(fig,"Figure6_eicu_replication","Figure6c_eicu_target_performance",metric_plot)

panel_audit = []
for pdf in sorted(FIGURES.glob("Figure*/*.pdf")):
    stem = pdf.stem
    expected = {
        "pdf": pdf,
        "svg": pdf.with_suffix(".svg"),
        "tiff": pdf.with_suffix(".tiff"),
        "source_data": pdf.with_name(stem + "_source_data.csv"),
    }
    panel_audit.append(
        {
            "panel": str(pdf.relative_to(ROOT)),
            "files": {key: str(path.relative_to(ROOT)) for key, path in expected.items()},
            "ready": all(path.exists() and path.stat().st_size > 0 for path in expected.values()),
        }
    )
audit = {
    "summary": {
        "panels": len(panel_audit),
        "ready_panels": sum(item["ready"] for item in panel_audit),
        "ready": bool(panel_audit) and all(item["ready"] for item in panel_audit),
    },
    "panels": panel_audit,
}
(ROOT / "outputs" / "FIGURE_SOURCE_PREFLIGHT.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit["summary"], indent=2))
