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
# # Figure 7: prespecified robustness extensions
#
# Each quantitative panel is saved independently as editable SVG, vector PDF,
# 600-dpi TIFF, and aggregate source data.

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
OUT = ROOT / "figures" / "Figure7_robustness_extensions"
OUT.mkdir(parents=True, exist_ok=True, mode=0o700)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams.update({
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": .7,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": .08,
})

BLUE = "#0072B2"; SKY = "#56B4E9"; GREEN = "#009E73"; ORANGE = "#E69F00"
VERMILLION = "#D55E00"; PURPLE = "#CC79A7"; GREY = "#8A8A8A"; BLACK = "#272727"


def save(fig, name, source):
    source.to_csv(OUT / f"{name}_source_data.csv", index=False)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.tiff", dpi=600)
    plt.close(fig)


# %% Figure 7a: extended transport
transport = pd.read_csv(TABLES / "Table_public_extended_bidirectional_transport.csv")
transport["direction_label"] = transport.transport_direction.map({
    "MIMIC-IV_to_eICU": "MIMIC-IV → eICU", "eICU_to_MIMIC-IV": "eICU → MIMIC-IV"
})
fig, ax = plt.subplots(figsize=(3.7, 2.5))
y_positions = {"MIMIC-IV → eICU": 1, "eICU → MIMIC-IV": 0}
offsets = {"minimal": -.12, "extended_common": .12}
styles = {"minimal": (GREY, "o", "Minimal"), "extended_common": (BLUE, "s", "Extended common")}
for spec, (color, marker, label) in styles.items():
    sub = transport.loc[transport.model_specification.eq(spec)]
    for row in sub.itertuples():
        y = y_positions[row.direction_label] + offsets[spec]
        ax.errorbar(row.roc_auc, y,
                    xerr=[[row.roc_auc - row.roc_auc_ci_lower], [row.roc_auc_ci_upper - row.roc_auc]],
                    fmt=marker, color=color, capsize=2, ms=4, label=label if y_positions[row.direction_label] == 1 else None)
ax.axvline(.5, color=BLACK, lw=.7, ls="--")
ax.set_yticks([0, 1], ["eICU → MIMIC-IV", "MIMIC-IV → eICU"])
ax.set_xlabel("External AUC (95% sampling interval)")
ax.set_xlim(.49, .64)
ax.legend(loc="center left")
ax.set_title("More common variables did not ensure transport", loc="left", fontweight="bold")
save(fig, "Figure7a_extended_common_transport", transport)

# %% Figure 7b: discrimination-strength stress test
stress = pd.read_csv(TABLES / "Table_discrimination_strength_stress_test.csv")
plot = stress.loc[(stress.metric.eq("oe")) & stress.method.isin([
    "full_reference_score", "local_recalibration_apparent", "local_recalibration_truth"
])].copy()
method_style = {
    "full_reference_score": (BLACK, "o", "-", "Original score, retained reference"),
    "local_recalibration_apparent": (GREEN, "s", "-", "Updated score, apparent endpoint"),
    "local_recalibration_truth": (VERMILLION, "^", "--", "Same update, retained reference"),
}
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
for ax, database in zip(axes, ["MIMIC-IV", "eICU"]):
    for method, (color, marker, linestyle, label) in method_style.items():
        sub = plot.loc[(plot.database.eq(database)) & plot.method.eq(method)].sort_values("target_auc")
        markerface = "none" if method == "full_reference_score" else color
        zorder = 5 if method == "full_reference_score" else 3
        ax.errorbar(sub.target_auc, sub["mean"], yerr=[sub["mean"] - sub.q025, sub.q975 - sub["mean"]],
                    color=color, marker=marker, markerfacecolor=markerface, ls=linestyle,
                    capsize=2, label=label, zorder=zorder)
    ax.axhline(1, color=GREY, lw=.7, ls=":")
    ax.set_xlabel("Designed retained-reference AUC")
    ax.set_title(database, loc="left", fontweight="bold")
axes[0].set_ylabel("Observed-to-expected ratio")
axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
save(fig, "Figure7b_discrimination_strength_stress", plot)

# %% Figure 7c: Bayesian hierarchical centre calibration
bayes = pd.read_csv(TABLES / "Table_bayesian_hierarchical_calibration_centres.csv")
bayes["center_numeric"] = pd.to_numeric(bayes.center)
fig, ax = plt.subplots(figsize=(3.7, 3.1))
y = np.arange(len(bayes))[::-1]
ax.errorbar(bayes.intercept_posterior_median, y,
            xerr=[bayes.intercept_posterior_median - bayes.intercept_credible_lower_95,
                  bayes.intercept_credible_upper_95 - bayes.intercept_posterior_median],
            fmt="o", color=BLUE, capsize=2, label="Hierarchical posterior median (95% CrI)")
ax.scatter(bayes.unpooled_intercept_at_common_slope, y, facecolors="none", edgecolors=VERMILLION,
           marker="s", label="Unpooled intercept at common slope")
ax.axvline(0, color=BLACK, lw=.7, ls="--")
ax.set_yticks(y, [f"Centre {int(c)} ({int(n)} records; {int(e)} events)" for c, n, e in zip(bayes.center_numeric, bayes.n, bayes.events)])
ax.set_xlabel("Calibration intercept")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=1)
ax.set_title("Partial pooling limits sparse-centre overinterpretation", loc="left", fontweight="bold")
save(fig, "Figure7c_bayesian_hierarchical_calibration", bayes)

# %% Figure 7d: measurement-aware subgroup decomposition
fair = pd.read_csv(TABLES / "Table_measurement_aware_fairness_calibration_gap.csv")
fair = fair.loc[fair.metric.eq("oe")].copy()
fair["variable_label"] = fair.group_variable.map({"age": "age", "sex": "sex", "race_or_ethnicity": "race/ethnicity"}).fillna(fair.group_variable)
fair["label"] = fair.database + ": " + fair.variable_label + "=" + fair.group.astype(str)
fair = fair.sort_values(["database", "group_variable", "group"])
fig, ax = plt.subplots(figsize=(5.0, max(3.0, .18 * len(fair) + .7)))
y = np.arange(len(fair))[::-1]
colors = fair.database.map({"MIMIC-IV": BLUE, "eICU": ORANGE})
ax.errorbar(fair.measurement_induced_gap_mean, y,
            xerr=[fair.measurement_induced_gap_mean - fair.measurement_induced_gap_q025,
                  fair.measurement_induced_gap_q975 - fair.measurement_induced_gap_mean],
            fmt="none", ecolor=GREY, capsize=2, lw=.8)
ax.scatter(fair.measurement_induced_gap_mean, y, c=colors, s=18)
ax.axvline(0, color=BLACK, lw=.7, ls="--")
ax.set_yticks(y, fair.label)
ax.set_xlabel("Measurement-induced O/E gap\n(apparent reconstructed − retained reference)")
ax.set_title("Subgroup disparities include a measurement component", loc="left", fontweight="bold")
save(fig, "Figure7d_measurement_aware_fairness", fair)

files = sorted(path.name for path in OUT.iterdir() if path.is_file())
audit = {
    "figure": "Figure 7 robustness extensions",
    "panels": 4, "files": len(files), "formats_per_panel": ["svg", "pdf", "tiff", "source_data_csv"],
    "palette": "Okabe-Ito colorblind-safe", "minimum_font_points": 6.5,
    "patient_level_data_read": False, "aggregate_source_data_only": True,
}
(ROOT / "outputs" / "FIGURE7_EXTENSION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2))
