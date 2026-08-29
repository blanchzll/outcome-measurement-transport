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
# # Figure 9: source-variable availability and internal-consistency audit
#
# Each panel is stored separately as vector PDF, editable SVG, 600-dpi TIFF, and
# aggregate source data. No patient-level identifiers or values are exported.

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
OUT = ROOT / "figures" / "Figure9_source_variable_quality"
OUT.mkdir(parents=True, exist_ok=True, mode=0o700)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams.update({
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#8A8A8A"
BLACK = "#272727"


def save(fig, name: str, source: pd.DataFrame) -> None:
    source.to_csv(OUT / f"{name}_source_data.csv", index=False)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.tiff", dpi=600)
    plt.close(fig)


# %% Figure 9a: missingness heatmap
missing = pd.read_csv(TABLES / "Table_source_missingness_by_variable_center.csv")
selected = [
    "Age", "Gender", "Gastrocolorectal", "Diabetes", "Hypertension",
    "PreopCr", "PreopHb", "PreopAlb", "BMI", "ASAGrade", "OperationTime",
    "IntraopBloodLoss", "IntraopTransfusion", "IntraopFluid", "IntraopVasoactive",
]
plot = missing.loc[missing.variable.isin(selected)].copy()
plot["variable"] = pd.Categorical(plot.variable, categories=selected, ordered=True)
matrix = plot.pivot(index="variable", columns="center", values="missing_rate_effective").reindex(selected)
fig, ax = plt.subplots(figsize=(4.5, 4.2))
image = ax.imshow(matrix.to_numpy() * 100, aspect="auto", cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(np.arange(len(matrix.columns)), [f"Centre {int(x)}" for x in matrix.columns])
ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
        value = matrix.iloc[row, column] * 100
        ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=5.6,
                color="white" if value >= 55 else BLACK)
cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Effective missingness (%)")
ax.set_title("Predictor availability did not transport across centres", loc="left", fontweight="bold")
save(fig, "Figure9a_predictor_missingness_by_centre", plot)


# %% Figure 9b: source-outcome internal consistency
consistency = pd.read_csv(TABLES / "Table_source_outcome_internal_consistency.csv")
categories = [
    ("aki_stage_binary_mismatch", "Binary-stage mismatch", BLUE),
    ("rrt_code_1_among_non_aki", "RRT=1 among non-AKI", VERMILLION),
    ("rrt_invalid_code_2_to_5", "Invalid RRT code", ORANGE),
]
x = np.arange(len(consistency))
width = 0.23
fig, ax = plt.subplots(figsize=(4.4, 2.7))
for index, (column, label, color) in enumerate(categories):
    ax.bar(x + (index - 1) * width, consistency[column], width=width, color=color, label=label)
ax.set_xticks(x, [f"Centre {int(value)}" for value in consistency.center])
ax.set_ylabel("Patients")
ax.set_ylim(0, max(7, consistency[[x[0] for x in categories]].to_numpy().max() + 1))
ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.set_title("Binary outcome, AKI stage, and RRT fields were not fully consistent", loc="left", fontweight="bold")
save(fig, "Figure9b_outcome_internal_consistency", consistency)


# %% Figure 9c: exploratory downstream outcome associations
downstream = pd.read_csv(TABLES / "Table_source_AKI_downstream_outcomes_exploratory.csv")
labels = {
    "Reoperation30d": "30-day reoperation",
    "Readmission30d": "30-day readmission",
    "Mortality90d": "90-day mortality",
}
downstream["label"] = downstream.outcome.map(labels)
y = np.arange(len(downstream))[::-1]
fig, ax = plt.subplots(figsize=(4.2, 2.5))
ax.errorbar(
    downstream.risk_difference * 100, y,
    xerr=[
        (downstream.risk_difference - downstream.risk_difference_ci_lower) * 100,
        (downstream.risk_difference_ci_upper - downstream.risk_difference) * 100,
    ],
    fmt="o", color=BLUE, ecolor=GREY, capsize=2, lw=0.8, markersize=4,
)
ax.axvline(0, color=BLACK, ls="--", lw=0.7)
ax.set_yticks(y, downstream.label)
ax.set_xlabel("Risk difference for AKI vs non-AKI (percentage points)")
ax.set_title("Exploratory downstream associations; no causal interpretation", loc="left", fontweight="bold")
save(fig, "Figure9c_AKI_downstream_risk_difference", downstream)


files = sorted(path.name for path in OUT.iterdir() if path.is_file())
audit = {
    "figure": "Figure 9 source-variable availability and internal consistency",
    "panels": 3, "files": len(files),
    "formats_per_panel": ["svg", "pdf", "tiff", "source_data_csv"],
    "palette": "Okabe-Ito colorblind-safe", "aggregate_source_data_only": True,
    "patient_level_data_read": False,
}
(ROOT / "outputs" / "FIGURE9_SOURCE_VARIABLE_AUDIT.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2))
