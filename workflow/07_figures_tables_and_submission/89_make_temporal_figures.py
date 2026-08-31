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
# # Figure 8: source-cohort temporal and observation-opportunity audit
#
# Each quantitative panel is saved separately as vector PDF, editable SVG,
# 600-dpi TIFF, and aggregate source data. No patient-level dates are exported.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(str(_release_path('analysis')))
TABLES = ROOT / "tables"
OUT = ROOT / "figures" / "Figure8_source_temporal_audit"
OUT.mkdir(parents=True, exist_ok=True, mode=0o700)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#8A8A8A"
BLACK = "#272727"
CENTRE_COLORS = [BLUE, ORANGE, GREEN, VERMILLION, PURPLE]


def save(fig, name: str, source: pd.DataFrame) -> None:
    source.to_csv(OUT / f"{name}_source_data.csv", index=False)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.tiff", dpi=600)
    plt.close(fig)


# %% Figure 8a: recruitment by calendar year and centre
recruitment = pd.read_csv(TABLES / "Table_source_recruitment_by_center_year.csv")
count_matrix = recruitment.pivot(index="SurgeryYear", columns="Center", values="n").fillna(0)
event_totals = recruitment.groupby("SurgeryYear")["events"].sum()
fig, ax = plt.subplots(figsize=(5.1, 2.7))
bottom = np.zeros(len(count_matrix))
for index, centre in enumerate(count_matrix.columns):
    values = count_matrix[centre].to_numpy()
    ax.bar(
        count_matrix.index.astype(int),
        values,
        bottom=bottom,
        width=0.76,
        color=CENTRE_COLORS[index],
        label=f"Centre {int(centre)}",
    )
    bottom += values
for x, total, events in zip(count_matrix.index.astype(int), bottom, event_totals.reindex(count_matrix.index)):
    ax.text(x, total + max(bottom) * 0.018, f"{int(events)} events", ha="center", va="bottom", fontsize=12)
ax.set_xlabel("Year of surgery")
ax.set_ylabel("Patients")
ax.set_xticks(count_matrix.index.astype(int))
ax.set_ylim(0, max(bottom) * 1.15)
ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.set_title("Centre participation changed across calendar time", loc="left", fontweight="bold")
save(fig, "Figure8a_recruitment_by_year", recruitment)


# %% Figure 8b: primary temporal AUC
performance = pd.read_csv(TABLES / "Table_source_temporal_validation.csv")
primary = performance.loc[performance.split_definition.eq("within_centre_70_30")].copy()
model_labels = {
    "ridge": "Ridge",
    "restricted_rf": "Restricted random forest",
    "gradient_boosting": "Gradient boosting",
    "soft_voting": "Equal-weight voting",
}
primary["label"] = primary.feature_set + ": " + primary.model.map(model_labels)
primary = primary.sort_values(["feature_set", "model"], ascending=[False, True])
y = np.arange(len(primary))[::-1]
colors = primary.feature_set.map({"P": GREY, "PI": BLUE})
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.errorbar(
    primary.roc_auc,
    y,
    xerr=[
        primary.roc_auc - primary.roc_auc_ci_lower,
        primary.roc_auc_ci_upper - primary.roc_auc,
    ],
    fmt="none",
    ecolor=GREY,
    capsize=2,
    lw=0.8,
)
ax.scatter(primary.roc_auc, y, c=colors, s=22, zorder=3)
ax.axvline(0.5, color=BLACK, lw=0.7, ls="--")
ax.set_yticks(y, primary.label)
ax.set_xlabel("Temporal-validation AUC (95% bootstrap interval)")
ax.set_xlim(0.48, 0.88)
ax.set_title("Discrimination persisted in later patients", loc="left", fontweight="bold")
save(fig, "Figure8b_within_centre_temporal_auc", primary)


# %% Figure 8c: primary temporal O/E
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.errorbar(
    primary.oe_ratio,
    y,
    xerr=[
        primary.oe_ratio - primary.oe_ratio_ci_lower,
        primary.oe_ratio_ci_upper - primary.oe_ratio,
    ],
    fmt="none",
    ecolor=GREY,
    capsize=2,
    lw=0.8,
)
ax.scatter(primary.oe_ratio, y, c=colors, s=22, zorder=3)
ax.axvline(1, color=BLACK, lw=0.7, ls="--")
ax.set_yticks(y, primary.label)
ax.set_xlabel("Observed-to-expected ratio (95% bootstrap interval)")
ax.set_xlim(0.4, 1.2)
ax.set_title("Later cohorts had fewer events than predicted", loc="left", fontweight="bold")
save(fig, "Figure8c_within_centre_temporal_oe", primary)


# %% Figure 8d: inpatient observation opportunity
observation = pd.read_csv(TABLES / "Table_source_inpatient_observation_opportunity.csv")
wide = observation.pivot(index="center", columns="observation_group", values="n").fillna(0)
wide["total"] = wide.sum(axis=1)
wide["fraction_at_least_7"] = wide.get(">=7 postoperative inpatient days", 0) / wide["total"]
wide = wide.reset_index()
fig, ax = plt.subplots(figsize=(3.8, 2.6))
bars = ax.bar(
    wide.center.astype(int),
    wide.fraction_at_least_7,
    color=CENTRE_COLORS[: len(wide)],
    width=0.66,
)
for bar, row in zip(bars, wide.itertuples()):
    n7 = int(round(row.fraction_at_least_7 * row.total))
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.012,
        f"{n7}/{int(row.total)}",
        ha="center",
        va="bottom",
        fontsize=12,
    )
ax.set_ylim(0, 1.08)
ax.set_xticks(wide.center.astype(int), [f"Centre {int(x)}" for x in wide.center])
ax.set_ylabel("Patients with ≥7 postoperative inpatient days")
ax.set_title("Most patients remained in hospital through day 7", loc="left", fontweight="bold")
save(fig, "Figure8d_inpatient_observation_opportunity", wide)


files = sorted(path.name for path in OUT.iterdir() if path.is_file())
audit = {
    "figure": "Figure 8 source temporal and observation-opportunity audit",
    "panels": 4,
    "files": len(files),
    "formats_per_panel": ["svg", "pdf", "tiff", "source_data_csv"],
    "palette": "Okabe-Ito colorblind-safe",
    "patient_level_data_read": False,
    "aggregate_source_data_only": True,
}
(ROOT / "outputs" / "FIGURE8_TEMPORAL_AUDIT.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2))
