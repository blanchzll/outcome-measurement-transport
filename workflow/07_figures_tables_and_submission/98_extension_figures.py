#!/usr/bin/env python3
# %% [markdown]
# # Supplementary figures for empirical schedules, reference design and bounds

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WIDTH = 7.086614173  # 180 mm, current Nature Communications double-column width
DB_ORDER = ["INSPIRE", "MIMIC", "EICU"]
DB_LABEL = {"INSPIRE": "INSPIRE", "MIMIC": "MIMIC-IV", "EICU": "eICU"}
STRATEGIES = ["random", "risk_quintile_equal", "risk_enriched", "cluster_stratified"]
STRATEGY_LABEL = {
    "random": "Random",
    "risk_quintile_equal": "Risk-quintile balanced",
    "risk_enriched": "High-risk enriched",
    "cluster_stratified": "Cluster stratified",
}
COLORS = {
    "random": "#4C78A8",
    "risk_quintile_equal": "#72B7B2",
    "risk_enriched": "#ECA82C",
    "cluster_stratified": "#B279A2",
    "random_bound": "#4C78A8",
    "high_bound": "#D55E00",
}


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def panel(ax, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")


def save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def heatmap(ax, table: pd.DataFrame, title: str, fmt: str, vmin=None, vmax=None, cmap="Blues") -> None:
    values = table.reindex(index=DB_ORDER, columns=DB_ORDER).to_numpy(float)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(3), [DB_LABEL[x] for x in DB_ORDER], rotation=25, ha="right")
    ax.set_yticks(range(3), [DB_LABEL[x] for x in DB_ORDER])
    ax.set_xlabel("Donor measurement schedule")
    ax.set_ylabel("Target patients and predictions")
    ax.set_title(title, pad=5)
    threshold = np.nanmean(values)
    for i in range(3):
        for j in range(3):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center",
                        color="white" if value > threshold else "#222222", fontsize=6)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5)


def schedule_figure(table_path: Path, output_root: Path, figure_number: int, two_database: bool = False) -> None:
    data = pd.read_csv(table_path)
    data = data.loc[data.tolerance_hours.eq(12)].copy()
    order = ["MIMIC", "EICU"] if two_database else DB_ORDER
    specs = [
        ("full_reference", "retained_reference", "outcome_observed_fraction", "Endpoint observable", ".2f", 0, 1, "Blues"),
        ("full_reference", "retained_reference", "reconstructed_sensitivity", "Reconstructed sensitivity", ".2f", 0, 1, "Blues"),
        ("local_recalibration", "reconstructed_observed", "oe", "Updated O/E vs reconstructed endpoint", ".2f", 0.5, 1.5, "PuBuGn"),
        ("local_recalibration", "retained_reference", "oe", "Same updated O/E vs retained reference", ".2f", 0.5, 1.5, "YlOrBr"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * 0.78), constrained_layout=True)
    for label, ax, spec in zip("abcd", axes.ravel(), specs):
        method, target, metric, title, fmt, vmin, vmax, cmap = spec
        subset = data.loc[(data.method == method) & (data.evaluation_target == target) & (data.metric == metric)]
        pivot = subset.pivot(index="target_database", columns="donor_schedule_database", values="mean")
        values = pivot.reindex(index=order, columns=order).to_numpy(float)
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(order)), [DB_LABEL[x] for x in order], rotation=25, ha="right")
        ax.set_yticks(range(len(order)), [DB_LABEL[x] for x in order])
        ax.set_xlabel("Donor measurement schedule")
        ax.set_ylabel("Target patients and predictions")
        ax.set_title(title, pad=5)
        threshold = (vmin + vmax) / 2
        for i in range(len(order)):
            for j in range(len(order)):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(j, i, format(value, fmt), ha="center", va="center",
                            color="white" if value > threshold else "#222222", fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
        panel(ax, label)
    folder = output_root / f"SupplementaryFigure{figure_number}"
    source_name = f"SupplementaryFigure{figure_number}_source_data.csv"
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / source_name, index=False)
    save(fig, folder, f"SupplementaryFigure{figure_number}")


def reference_sampling_figure(table_path: Path, output_root: Path) -> None:
    data = pd.read_csv(table_path)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, WIDTH * 0.63), sharex=True, constrained_layout=True)
    for col, database in enumerate(DB_ORDER):
        for strategy in STRATEGIES:
            color = COLORS[strategy]
            event = data.loc[(data.database == database) & (data.strategy == strategy) & (data.metric == "reference_events")].sort_values("reference_fraction")
            error = data.loc[(data.database == database) & (data.strategy == strategy) & (data.metric == "absolute_log_oe")].sort_values("reference_fraction")
            axes[0, col].plot(event.reference_fraction * 100, event["mean"], marker="o", ms=2.5, lw=1.1, color=color, label=STRATEGY_LABEL[strategy])
            axes[0, col].fill_between(event.reference_fraction * 100, event.q025, event.q975, color=color, alpha=0.10, linewidth=0)
            axes[1, col].plot(error.reference_fraction * 100, error["mean"], marker="o", ms=2.5, lw=1.1, color=color)
            axes[1, col].fill_between(error.reference_fraction * 100, error.q025, error.q975, color=color, alpha=0.10, linewidth=0)
        axes[0, col].set_title(DB_LABEL[database])
        axes[1, col].set_xlabel("Reference sample (%)")
        axes[0, col].grid(axis="y", color="#dddddd", lw=0.5)
        axes[1, col].grid(axis="y", color="#dddddd", lw=0.5)
    axes[0, 0].set_ylabel("Reference events")
    axes[1, 0].set_ylabel("Absolute ln O/E, held out")
    for label, ax in zip("abcdef", axes.ravel()):
        panel(ax, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    folder = output_root / "SupplementaryFigure10"
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / "SupplementaryFigure10_source_data.csv", index=False)
    save(fig, folder, "SupplementaryFigure10")


def source_bounds_figure(table_path: Path, output_root: Path) -> None:
    data = pd.read_csv(table_path)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * 0.35), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ["oe", "auc"], ["Observed-to-expected ratio", "AUC"]):
        random = data.loc[(data.assignment_mechanism == "random") & (data.metric == metric)].sort_values("assumed_postdischarge_event_fraction")
        high = data.loc[(data.assignment_mechanism == "highest_predicted_risk") & (data.metric == metric)].sort_values("assumed_postdischarge_event_fraction")
        x = random.assumed_postdischarge_event_fraction * 100
        if metric == "oe":
            ax.plot(x, random["mean"], color="#333333", marker="o", ms=3, lw=1.2,
                    label="Both assignment mechanisms (identical total events)")
        else:
            ax.plot(x, random["mean"], color=COLORS["random_bound"], marker="o", ms=3, lw=1.2, label="Random among short-stay negatives")
            ax.fill_between(x, random.q025, random.q975, color=COLORS["random_bound"], alpha=0.16, linewidth=0)
            ax.plot(high.assumed_postdischarge_event_fraction * 100, high["mean"], color=COLORS["high_bound"], marker="s", ms=3, lw=1.2, label="Highest predicted risk")
        ax.set_xlabel("Assumed event fraction among 167 short-stay negatives (%)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
    axes[0].axhline(1, color="#777777", ls="--", lw=0.8)
    panel(axes[0], "a"); panel(axes[1], "b")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    folder = output_root / "SupplementaryFigure12"
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / "SupplementaryFigure12_source_data.csv", index=False)
    save(fig, folder, "SupplementaryFigure12")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    schedule_figure(args.tables / "Table_empirical_schedule_transport.csv", args.output, 9)
    reference_sampling_figure(args.tables / "Table_optimized_reference_sampling.csv", args.output)
    if (args.tables / "Table_hemoglobin_endpoint_replication.csv").exists():
        schedule_figure(args.tables / "Table_hemoglobin_endpoint_replication.csv", args.output, 11, two_database=True)
    source_bounds_figure(args.tables / "Table_source_postdischarge_sensitivity_bounds.csv", args.output)
    contracts = {
        "SupplementaryFigure9": "Observed cross-database measurement schedules change endpoint observability and retained-reference calibration for the same target patients and predictions.",
        "SupplementaryFigure10": "Reference-sample design trades event yield and risk coverage against weighting stability; high-risk enrichment is not uniformly most accurate.",
        "SupplementaryFigure11": "The measurement-transport failure mode is assessed for an independent longitudinal haemoglobin-decline endpoint without claiming adjudicated bleeding.",
        "SupplementaryFigure12": "Source-model conclusions are bounded across explicit assumptions about unobserved post-discharge events among short-stay recorded negatives.",
    }
    (args.output / "FIGURE_CONTRACTS.json").write_text(json.dumps(contracts, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
