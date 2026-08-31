#!/usr/bin/env python3
"""Build the complete four-database 12-h primary schedule-compatibility figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATABASES = ["INSPIRE", "MIMIC", "EICU", "VitalDB"]
METRICS = (
    ("outcome_observed_fraction", "Outcome observability"),
    ("reconstructed_sensitivity", "Reconstructed-endpoint sensitivity"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--public-source-dir", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.table)
    base = data.loc[
        data.tolerance_hours.eq(12)
        & data.method.eq("full_reference")
        & data.evaluation_target.eq("retained_reference")
        & data.metric.isin([item[0] for item in METRICS])
    ].copy()
    expected = len(DATABASES) ** 2 * len(METRICS)
    if len(base) != expected:
        raise RuntimeError(f"Expected {expected} primary rows, found {len(base)}")
    if not base.n_replicates.eq(200).all():
        raise RuntimeError("All primary cells must contain 200 schedule replicates")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.15), constrained_layout=True)
    for panel, (metric, title) in enumerate(METRICS):
        subset = base.loc[base.metric.eq(metric)]
        matrix = subset.pivot(index="target_database", columns="donor_schedule_database", values="mean")
        matrix = matrix.reindex(index=DATABASES, columns=DATABASES)
        if matrix.isna().any().any():
            raise RuntimeError(f"Incomplete matrix: {metric}")
        ax = axes[panel]
        image = ax.imshow(matrix.to_numpy(float), cmap="cividis", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(np.arange(4), DATABASES, rotation=30, ha="right")
        ax.set_yticks(np.arange(4), DATABASES)
        ax.set_xlabel("Donor timing pattern")
        if panel == 0:
            ax.set_ylabel("Target retained measurement grid")
        ax.set_title(f"{chr(97 + panel)}  {title}", loc="left", weight="bold")
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(4):
            for column in range(4):
                value = float(matrix.iloc[row, column])
                ax.text(
                    column,
                    row,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if value < 0.45 else "#111111",
                    fontsize=7.2,
                    weight="bold",
                )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.02)
    colorbar.set_label("Fraction")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "Figure5"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    fig.savefig(args.output_dir / "Figure5_300dpi.png", dpi=300)
    plt.close(fig)

    base["mcse_of_mean"] = base["sd"] / np.sqrt(base["n_replicates"])
    source_columns = [
        "target_database", "donor_schedule_database", "tolerance_hours", "metric",
        "n_replicates", "mean", "sd", "mcse_of_mean", "q025", "q975",
    ]
    source = base[source_columns].sort_values(["metric", "target_database", "donor_schedule_database"])
    source_path = args.output_dir / "Figure5_primary_12h_source_data.csv"
    source.to_csv(source_path, index=False)
    old_source = args.output_dir / "Figure5_empirical_schedule_observability_source_data.csv"
    if old_source.exists():
        old_source.unlink()
    args.public_source_dir.mkdir(parents=True, exist_ok=True)
    for old in args.public_source_dir.glob("Figure5*_source_data.csv"):
        old.unlink()
    shutil.copy2(source_path, args.public_source_dir / source_path.name)

    audit = {
        "status": "PASS",
        "primary_tolerance_hours": 12,
        "databases": DATABASES,
        "metrics": [item[0] for item in METRICS],
        "cells": int(len(source)),
        "replicates_per_cell": 200,
        "mcse_definition": "replicate SD divided by sqrt(200); donor-schedule Monte Carlo error conditional on fixed target trajectories and donor pools",
        "figure_pdf_sha256": sha256(stem.with_suffix(".pdf")),
        "source_sha256": sha256(source_path),
        "interpretation": "Donor timing-pattern to target retained-grid compatibility; not a hospital-policy intervention.",
    }
    (args.output_dir / "FIGURE5_PRIMARY_12H_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
