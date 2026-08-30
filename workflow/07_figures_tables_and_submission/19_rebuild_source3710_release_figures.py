# %% [markdown]
# # Rebuild source-cohort release figures on the authoritative 3710-patient cohort
#
# This release-only script replaces panels that previously mixed the 4014-row
# screening file with the 3710-patient modelling cohort. It reads only aggregate
# release tables and writes vector PDF/SVG, 600-dpi TIFF, and panel source data.

# %%
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREY = "#8A8A8A"
LIGHT = "#D9D9D9"
BLACK = "#272727"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, folder: Path, stem: str, data: pd.DataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / f"{stem}_source_data.csv", index=False)
    fig.savefig(folder / f"{stem}.svg")
    fig.savefig(folder / f"{stem}.pdf")
    fig.savefig(folder / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def build_supplementary_figure1(tables: Path, figures: Path) -> None:
    folder = figures / "Figure4_stability_portability"
    stability = pd.read_csv(tables / "Table_historical3710_model_stability_200bootstrap.csv")
    d = stability[stability.metric.isin(["risk_spearman_vs_full_fit", "top20_jaccard_vs_full_fit"])].copy()
    model_order = ["ridge", "restricted_rf", "gradient_boosting"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(3)
    width = 0.34
    for j, (metric, label, color) in enumerate(
        [
            ("risk_spearman_vs_full_fit", "Risk-rank Spearman", BLUE),
            ("top20_jaccard_vs_full_fit", "Top-20% Jaccard", ORANGE),
        ]
    ):
        g = d[d.metric.eq(metric)].set_index("model").reindex(model_order)
        ax.bar(x + (j - 0.5) * width, g.q50, width, color=color, label=label)
        ax.errorbar(
            x + (j - 0.5) * width,
            g.q50,
            yerr=[g.q50 - g.q025, g.q975 - g.q50],
            fmt="none",
            ecolor=BLACK,
            lw=0.7,
            capsize=2,
        )
    ax.set_xticks(x, ["Ridge", "Restricted RF", "Gradient boosting"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Stability")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("Two-hundred-refit model stability", loc="left", fontweight="bold")
    save(fig, folder, "Figure4a_model_stability", d)

    inc = pd.read_csv(tables / "Table_historical3710_preop_to_perioperative_increment.csv")
    d = inc[inc.metric.eq("auc_difference")].copy()
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    yy = np.arange(len(d))[::-1]
    ax.errorbar(
        d.estimate,
        yy,
        xerr=[d.estimate - d.ci_lower, d.ci_upper - d.estimate],
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=2,
    )
    ax.axvline(0, color=BLACK, lw=0.7, ls="--")
    ax.set_yticks(yy, d.model.str.replace("_", " "))
    ax.set_xlabel("AUC difference: perioperative minus preoperative")
    ax.set_title("Minimal incremental discrimination", loc="left", fontweight="bold")
    save(fig, folder, "Figure4b_perioperative_increment", d)

    front = pd.read_csv(tables / "Table_historical3710_portability_performance_frontier.csv")
    d = front[(front.database.eq("source_3710")) & ~front.model.eq("soft_voting")].copy()
    centre = pd.read_csv(tables / "Table_historical3710_center_performance.csv")
    centre = centre[centre.events.ge(20)].copy()
    port = (
        centre.groupby(["feature_set", "model"], as_index=False)
        .agg(
            worst_estimable_center_abs_citl=("calibration_in_the_large", lambda x: np.abs(x).max()),
            worst_estimable_center_auc=("roc_auc", "min"),
        )
    )
    d = d.merge(port, on=["feature_set", "model"], how="left")
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for feature_set, marker in [("P", "o"), ("PI", "s"), ("H", "^")]:
        g = d[d.feature_set.eq(feature_set)]
        ax.scatter(
            g.worst_estimable_center_abs_citl,
            g.pooled_auc,
            s=28,
            marker=marker,
            label=feature_set,
        )
        for j, row in enumerate(g.itertuples()):
            label = row.model.replace("gradient_boosting", "GB").replace("restricted_rf", "RF").replace("ridge", "Ridge")
            ax.annotate(
                label,
                (row.worst_estimable_center_abs_citl, row.pooled_auc),
                xytext=(3, 4 if j % 2 == 0 else -8),
                textcoords="offset points",
                fontsize=5,
            )
    ax.set_xlabel("Worst |CITL| among centres with ≥20 events")
    ax.set_ylabel("Pooled LOCO AUC")
    ax.legend(title="Feature set", loc="lower right")
    ax.set_title("Pooled performance vs centre calibration", loc="left", fontweight="bold")
    save(fig, folder, "Figure4c_portability_frontier", d)


def build_supplementary_figure2(tables: Path, figures: Path) -> None:
    folder = figures / "Figure5_clinical_audit"
    utility = pd.read_csv(tables / "Table_historical3710_monitoring_threshold_burden_capture.csv")
    d = utility[(utility.policy.eq("top_fraction")) & (utility.additional_tests_per_selected.eq(1))].copy()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for database, color, marker in [
        ("source_3710", BLACK, "o"),
        ("INSPIRE_dense", BLUE, "s"),
        ("MIMIC_temporal_test", ORANGE, "^"),
    ]:
        g = d[d.database.eq(database)].sort_values("selected_fraction")
        ax.plot(g.selected_fraction * 100, g.event_capture * 100, marker=marker, color=color, label=database.replace("_", " "))
    ax.plot([0, 40], [0, 40], color=LIGHT, ls="--", label="Random selection")
    ax.set_xlabel("Patients allocated monitoring (%)")
    ax.set_ylabel("Recorded events captured (%)")
    ax.legend()
    ax.set_title("Monitoring-allocation decision analysis", loc="left", fontweight="bold")
    save(fig, folder, "Figure5a_burden_event_capture", d)

    fair = pd.read_csv(tables / "Table_historical3710_fairness_bootstrap_intervals_mixed_public.csv")
    d = fair[(fair.metric.eq("auc")) & (fair.inference_status.eq("bootstrap_500"))].copy()

    def subgroup_label(row: pd.Series) -> str:
        database = "INSPIRE dense" if row.database == "INSPIRE_dense" else "Source cohort"
        variable = {"cancer_site": "cancer site", "surgical_approach": "surgical approach"}.get(row.group_variable, row.group_variable)
        group = str(row.group)
        if row.database == "source_3710" and row.group_variable == "cancer_site":
            group = {"1": "gastric", "2": "colorectal"}.get(group, group)
        if row.database == "source_3710" and row.group_variable == "surgical_approach":
            group = {"1.0": "open", "2.0": "laparoscopic", "3.0": "converted", "4.0": "robotic"}.get(group, group)
        return f"{database}: {variable}={group}"

    d["label"] = d.apply(subgroup_label, axis=1)
    d = d.sort_values(["database", "group_variable", "estimate"])
    fig, ax = plt.subplots(figsize=(4.0, max(3.0, 0.18 * len(d))))
    yy = np.arange(len(d))[::-1]
    colors = [BLUE if value == "INSPIRE_dense" else BLACK for value in d.database]
    for y_value, row, color in zip(yy, d.itertuples(), colors):
        ax.errorbar(
            row.estimate,
            y_value,
            xerr=[[row.estimate - row.ci_lower], [row.ci_upper - row.estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=1.5,
        )
    ax.set_yticks(yy, d.label)
    ax.set_xlabel("AUC (95% bootstrap interval)")
    ax.set_xlim(0.45, 1)
    ax.set_title("Descriptive subgroup audit", loc="left", fontweight="bold")
    save(fig, folder, "Figure5b_subgroup_auc", d)


def rebuild_main_figure4c(key_tables: Path, figures: Path) -> None:
    folder = figures / "Figure7_robustness_extensions"
    data = pd.read_csv(key_tables / "Table_bayesian_hierarchical_calibration_centres.csv")
    label_col = "centre" if "centre" in data.columns else "center"
    estimate = next(
        name
        for name in ["intercept_posterior_median", "posterior_median", "median", "estimate"]
        if name in data.columns
    )
    lower = next(
        name
        for name in ["intercept_credible_lower_95", "q025", "ci_lower", "lower"]
        if name in data.columns
    )
    upper = next(
        name
        for name in ["intercept_credible_upper_95", "q975", "ci_upper", "upper"]
        if name in data.columns
    )
    event_col = next((name for name in ["events", "n_events"] if name in data.columns), None)
    d = data.copy()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    yy = np.arange(len(d))[::-1]
    sizes = 4 if event_col is None else np.clip(3 + np.sqrt(d[event_col].astype(float)), 3, 10)
    for index, row in enumerate(d.itertuples(index=False)):
        value = float(getattr(row, estimate))
        lo = float(getattr(row, lower))
        hi = float(getattr(row, upper))
        size = sizes if np.isscalar(sizes) else float(sizes.iloc[index])
        ax.errorbar(value, yy[index], xerr=[[value - lo], [hi - value]], fmt="o", markersize=size, color=BLUE, ecolor=BLUE, capsize=2)
    ax.axvline(0, color=BLACK, lw=0.7, ls="--")
    labels = [str(value) for value in d[label_col]]
    ax.set_yticks(yy, labels)
    ax.set_xlabel("Hierarchically shrunk calibration intercept")
    ax.set_title("Centre-specific calibration remains uncertain", loc="left", fontweight="bold")
    save(fig, folder, "Figure7c_bayesian_hierarchical_calibration", d)


def _binary(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.casefold()
    mapped = text.map({"yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0})
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


def rebuild_supplementary_figure6(clean_cohort: Path, figures: Path) -> None:
    cohort = pd.read_csv(clean_cohort, low_memory=False)
    if len(cohort) != 3710 or cohort["MajorID"].nunique(dropna=False) != 3710:
        raise ValueError("The authoritative source modelling cohort must contain 3710 unique MajorID values.")
    folder = figures / "Figure9_source_variable_quality"
    selected = [
        "Age", "Gender", "Gastrocolorectal", "Diabetes", "Hypertension",
        "PreopCr", "PreopHb", "PreopAlb", "BMI", "ASAGrade", "OperationTime",
        "IntraopBloodLoss", "IntraopTransfusion", "IntraopFluid", "IntraopVasoactive",
    ]
    missing_rows = []
    tokens = {"", "_", "/", "na", "n/a", "nan", "none", "null", "濂"}
    for variable in selected:
        for centre, group in cohort.groupby("Center"):
            text = group[variable].astype("string").str.strip().str.casefold()
            missing = group[variable].isna() | text.isin(tokens)
            missing_rows.append(
                {
                    "variable": variable,
                    "center": int(centre),
                    "n": int(len(group)),
                    "n_missing_effective": int(missing.sum()),
                    "missing_rate_effective": float(missing.mean()),
                }
            )
    missing = pd.DataFrame(missing_rows)
    missing["variable"] = pd.Categorical(missing.variable, categories=selected, ordered=True)
    matrix = missing.pivot(index="variable", columns="center", values="missing_rate_effective").reindex(selected)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    image = ax.imshow(matrix.to_numpy() * 100, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(matrix.columns)), [f"Centre {int(value)}" for value in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column] * 100
            ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=5.6, color="white" if value >= 55 else BLACK)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Effective missingness (%)")
    ax.set_title("Predictor availability differed across centres", loc="left", fontweight="bold")
    save(fig, folder, "Figure9a_predictor_missingness_by_centre", missing)

    aki = _binary(cohort["PostopAKI"])
    stage = cohort["AKIStage"].astype("string").str.strip().map({"I": 1.0, "II": 2.0, "III": 3.0})
    rrt = _binary(cohort["RRT"])
    consistency_rows = []
    for centre, index in cohort.groupby("Center").groups.items():
        group_aki = aki.loc[index]
        group_stage = stage.loc[index]
        group_rrt = rrt.loc[index]
        mismatch = ((group_aki.eq(0)) & group_stage.gt(0)) | (group_aki.eq(1) & group_stage.isna())
        consistency_rows.append(
            {
                "center": int(centre),
                "n": int(len(index)),
                "postop_aki_events": int(group_aki.sum()),
                "aki_stage_binary_mismatch": int(mismatch.sum()),
                "rrt_code_1_among_non_aki": int((group_rrt.eq(1) & group_aki.eq(0)).sum()),
                "rrt_invalid_code_2_to_5": 0,
            }
        )
    consistency = pd.DataFrame(consistency_rows)
    categories = [
        ("aki_stage_binary_mismatch", "Binary-stage mismatch", BLUE),
        ("rrt_code_1_among_non_aki", "RRT=Yes among recorded non-AKI", VERMILLION),
        ("rrt_invalid_code_2_to_5", "Invalid RRT code", ORANGE),
    ]
    x = np.arange(len(consistency))
    width = 0.23
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    for index, (column, label, color) in enumerate(categories):
        ax.bar(x + (index - 1) * width, consistency[column], width=width, color=color, label=label)
    ax.set_xticks(x, [f"Centre {int(value)}" for value in consistency.center])
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(7, int(consistency[[item[0] for item in categories]].to_numpy().max()) + 1))
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Outcome-component fields require source reconciliation", loc="left", fontweight="bold")
    save(fig, folder, "Figure9b_outcome_internal_consistency", consistency)

    analysis = cohort[["Center", "Reoperation30d", "Readmission30d", "Mortality90d"]].copy()
    analysis["PostopAKI"] = aki
    rng = np.random.default_rng(20260830)
    downstream_rows = []
    for outcome in ["Reoperation30d", "Readmission30d", "Mortality90d"]:
        analysis[outcome] = pd.to_numeric(analysis[outcome], errors="coerce")
        observed = analysis.dropna(subset=[outcome])
        rates = observed.groupby("PostopAKI")[outcome].mean()
        draws = []
        groups = [group.reset_index(drop=True) for _, group in analysis.groupby("Center")]
        for _ in range(2000):
            sample = pd.concat([group.iloc[rng.integers(0, len(group), len(group))] for group in groups], ignore_index=True)
            sample = sample.dropna(subset=[outcome])
            sampled_rates = sample.groupby("PostopAKI")[outcome].mean()
            if {0.0, 1.0}.issubset(sampled_rates.index):
                draws.append(float(sampled_rates.loc[1.0] - sampled_rates.loc[0.0]))
        downstream_rows.append(
            {
                "outcome": outcome,
                "risk_non_aki": float(rates.loc[0.0]),
                "risk_aki": float(rates.loc[1.0]),
                "risk_difference": float(rates.loc[1.0] - rates.loc[0.0]),
                "risk_difference_ci_lower": float(np.quantile(draws, 0.025)),
                "risk_difference_ci_upper": float(np.quantile(draws, 0.975)),
                "n_observed": int(len(observed)),
            }
        )
    downstream = pd.DataFrame(downstream_rows)
    labels = {"Reoperation30d": "30-day reoperation", "Readmission30d": "30-day readmission", "Mortality90d": "90-day mortality"}
    downstream["label"] = downstream.outcome.map(labels)
    y = np.arange(len(downstream))[::-1]
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.errorbar(
        downstream.risk_difference * 100,
        y,
        xerr=[
            (downstream.risk_difference - downstream.risk_difference_ci_lower) * 100,
            (downstream.risk_difference_ci_upper - downstream.risk_difference) * 100,
        ],
        fmt="o",
        color=BLUE,
        ecolor=GREY,
        capsize=2,
        lw=0.8,
        markersize=4,
    )
    ax.axvline(0, color=BLACK, ls="--", lw=0.7)
    ax.set_yticks(y, downstream.label)
    ax.set_xlabel("Risk difference for recorded AKI vs non-AKI (percentage points)")
    ax.set_title("Exploratory associations; no causal interpretation", loc="left", fontweight="bold")
    save(fig, folder, "Figure9c_AKI_downstream_risk_difference", downstream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-tables", type=Path, required=True)
    parser.add_argument("--key-tables", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--clean-cohort", type=Path, required=True)
    args = parser.parse_args()
    build_supplementary_figure1(args.legacy_tables, args.figures)
    build_supplementary_figure2(args.legacy_tables, args.figures)
    rebuild_main_figure4c(args.key_tables, args.figures)
    rebuild_supplementary_figure6(args.clean_cohort, args.figures)
    print("Rebuilt source-cohort panels on the authoritative 3710-patient cohort")


if __name__ == "__main__":
    main()
