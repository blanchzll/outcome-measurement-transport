# %% [markdown]
# # Publication figures for the VitalDB four-database extension

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATABASE_ORDER = ["INSPIRE", "MIMIC", "EICU", "VitalDB"]
OKABE_ITO = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "gray": "#7A7A7A",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def heatmap(matrix: pd.DataFrame, title: str, colorbar_label: str, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 3.6))
    image = ax.imshow(matrix.to_numpy(float), cmap="cividis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_xlabel("Donor measurement schedule")
    ax.set_ylabel("Target retained patient trajectories")
    ax.set_title(title, loc="left", weight="bold")
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix.iloc[row, column])
            text_color = "white" if value < 0.45 else "#111111"
            ax.text(column, row, f"{100 * value:.0f}%", ha="center", va="center", color=text_color, fontsize=8, weight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def selection_plot(table: pd.DataFrame, output_stem: Path) -> None:
    labels = [
        "Full observable reference",
        "Naive dense subset",
        "IPAW (raw)",
        "IPAW (99% truncation)",
        "AIPW",
    ]
    values = table.event_rate.to_numpy(float)
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["sky"], OKABE_ITO["green"], OKABE_ITO["orange"]]
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    y = np.arange(len(labels))
    ax.hlines(y, 0, values, color="#D7D7D7", linewidth=1.2, zorder=1)
    ax.scatter(values, y, s=42, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    truth = values[0]
    ax.axvline(truth, color=OKABE_ITO["blue"], linestyle="--", linewidth=1.0, alpha=0.8)
    for yi, value in zip(y, values):
        ax.text(value + 0.002, yi, f"{100 * value:.1f}%", va="center", fontsize=7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) + 0.025)
    ax.set_xlabel("Operational creatinine-event rate")
    ax.set_title("Dense-reference selection is only partly corrected by measured covariates", loc="left", weight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-schedule", required=True, type=Path)
    parser.add_argument("--vitaldb-schedule", required=True, type=Path)
    parser.add_argument("--selection-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(exist_ok=True)

    schedule = pd.concat(
        [pd.read_csv(args.existing_schedule), pd.read_csv(args.vitaldb_schedule)], ignore_index=True
    )
    schedule.to_csv(source_dir / "four_database_empirical_schedule_transport.csv", index=False)
    base = schedule.loc[
        schedule.tolerance_hours.eq(24)
        & schedule.method.eq("full_reference")
        & schedule.evaluation_target.eq("retained_reference")
    ].copy()
    matrices = {}
    for metric in ("outcome_observed_fraction", "reconstructed_sensitivity"):
        subset = base.loc[base.metric.eq(metric)]
        matrix = subset.pivot(index="target_database", columns="donor_schedule_database", values="mean")
        matrix = matrix.reindex(index=DATABASE_ORDER, columns=DATABASE_ORDER)
        if matrix.isna().any().any():
            raise RuntimeError(f"Incomplete four-database schedule matrix for {metric}")
        matrices[metric] = matrix
        matrix.to_csv(source_dir / f"{metric}_24h_matrix.csv")

    heatmap(
        matrices["outcome_observed_fraction"],
        "The same retained trajectories become differently observable under transported schedules",
        "Outcome observable fraction",
        args.output_dir / "Figure5_empirical_schedule_observability",
    )
    heatmap(
        matrices["reconstructed_sensitivity"],
        "Measurement schedules alter event capture from the same retained trajectories",
        "Reconstructed-event sensitivity",
        args.output_dir / "SupplementaryFigure13_empirical_schedule_sensitivity",
    )
    selection = pd.read_csv(args.selection_table)
    selection.to_csv(source_dir / "vitaldb_dense_selection_correction.csv", index=False)
    selection_plot(selection, args.output_dir / "SupplementaryFigure14_vitaldb_selection_correction")

    audit = {
        "status": "PASS",
        "figures": [
            "Figure5_empirical_schedule_observability.pdf",
            "SupplementaryFigure13_empirical_schedule_sensitivity.pdf",
            "SupplementaryFigure14_vitaldb_selection_correction.pdf",
        ],
        "formats": ["PDF vector", "TIFF 600 dpi"],
        "palette": "cividis plus Okabe-Ito colorblind-safe accents",
        "schedule_tolerance_hours": 24,
        "numeric_labels": "percentages derived directly from source tables",
    }
    (args.output_dir / "VITALDB_FIGURE_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
