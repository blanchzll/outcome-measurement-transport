# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: '.py'
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # VitalDB waveform extension figures
#
# Each quantitative panel is exported as a separate vector PDF and 600-dpi TIFF
# in one supplementary-figure directory. All plotted values are also written as
# aggregate source data.

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "clinical_table_ridge": "#0072B2",
    "waveform_enhanced_ridge": "#D55E00",
    "apparent": "#56B4E9",
    "retained": "#E69F00",
}
MODEL_LABELS = {
    "clinical_table_ridge": "Clinical-table ridge",
    "waveform_enhanced_ridge": "Clinical + waveform ridge",
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


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def model_auc_plot(table: pd.DataFrame, output: Path) -> pd.DataFrame:
    data = table.loc[
        table.comparison.eq("model_performance") & table.metric.eq("auc")
    ].copy()
    data["label"] = data.model.map(MODEL_LABELS)
    data = data.set_index("model").loc[list(MODEL_LABELS)].reset_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    y = np.arange(len(data))
    for index, row in data.iterrows():
        ax.errorbar(
            row.estimate,
            y[index],
            xerr=[[row.estimate - row.ci_lower], [row.ci_upper - row.estimate]],
            fmt="o",
            color=COLORS[row.model],
            capsize=2.5,
            markersize=5,
            linewidth=1.2,
        )
    ax.set_yticks(y, data.label)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out area under the ROC curve (95% CI)")
    ax.set_title("Prespecified VitalDB risk engines", loc="left", weight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    lower = max(0.45, float(data.ci_lower.min()) - 0.03)
    upper = min(1.00, float(data.ci_upper.max()) + 0.03)
    ax.set_xlim(lower, upper)
    save(fig, output)
    return data[["model", "label", "estimate", "ci_lower", "ci_upper", "bootstrap_replicates"]]


def aggregate_stress(stress: pd.DataFrame) -> pd.DataFrame:
    method_map = {
        "recalibration_intercept_slope_apparent": "Apparent reconstructed endpoint",
        "recalibration_intercept_slope_truth": "Retained reference endpoint",
    }
    data = stress.loc[
        stress.method.isin(method_map)
        & stress.evaluation_target.isin(["reconstructed", "full"])
    ].copy()
    data["target"] = data.method.map(method_map)
    rows = []
    for (model, target), group in data.groupby(["model", "target"], sort=False):
        for metric in ("oe", "calibration_intercept", "calibration_slope"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "metric": metric,
                    "n_replicates": int(len(values)),
                    "mean": float(values.mean()),
                    "q025": float(values.quantile(0.025)),
                    "q975": float(values.quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def calibration_plot(summary: pd.DataFrame, metric: str, label: str, ideal: float, output: Path) -> None:
    data = summary.loc[summary.metric.eq(metric)].copy()
    targets = ["Apparent reconstructed endpoint", "Retained reference endpoint"]
    models = list(MODEL_LABELS)
    fig, ax = plt.subplots(figsize=(4.6, 2.55))
    x = np.arange(len(models))
    offsets = {targets[0]: -0.12, targets[1]: 0.12}
    for target in targets:
        selected = data.loc[data.target.eq(target)].set_index("model").loc[models]
        color = COLORS["apparent"] if target == targets[0] else COLORS["retained"]
        ax.errorbar(
            x + offsets[target],
            selected["mean"],
            yerr=[selected["mean"] - selected["q025"], selected["q975"] - selected["mean"]],
            fmt="o",
            color=color,
            label=target,
            capsize=2.5,
            linewidth=1.2,
            markersize=4.5,
        )
    ax.axhline(ideal, color="#444444", linestyle="--", linewidth=0.9)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in models])
    ax.set_ylabel(label)
    ax.set_title(
        "Calibration after updating on the reconstructed endpoint",
        loc="left",
        weight="bold",
    )
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(loc="best")
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-table", required=True, type=Path)
    parser.add_argument("--stress-replicates", required=True, type=Path)
    parser.add_argument("--qa-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    qa = json.loads(args.qa_audit.read_text(encoding="utf-8"))
    if qa.get("status") != "PASS":
        raise RuntimeError("Waveform extension integrity/timing QA must pass before plotting")
    model = pd.read_csv(args.model_table)
    stress = pd.read_csv(args.stress_replicates, low_memory=False)
    style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "source_data"
    source.mkdir(exist_ok=True)

    auc_source = model_auc_plot(model, args.output_dir / "Panel_A_model_discrimination")
    auc_source.to_csv(source / "Panel_A_model_discrimination.csv", index=False)
    stress_source = aggregate_stress(stress)
    stress_source.to_csv(source / "Panels_B_C_calibration_stress.csv", index=False)
    calibration_plot(
        stress_source,
        "oe",
        "Observed-to-expected ratio",
        1.0,
        args.output_dir / "Panel_B_apparent_vs_retained_OE",
    )
    calibration_plot(
        stress_source,
        "calibration_slope",
        "Calibration slope",
        1.0,
        args.output_dir / "Panel_C_apparent_vs_retained_slope",
    )

    delta = model.loc[
        model.comparison.eq("waveform_minus_clinical_paired_delta") & model.metric.eq("auc")
    ].iloc[0]
    audit = {
        "status": "PASS",
        "manuscript_integration": qa.get("manuscript_integration"),
        "figure_directory": args.output_dir.name,
        "panels": [
            "Panel_A_model_discrimination.pdf",
            "Panel_B_apparent_vs_retained_OE.pdf",
            "Panel_C_apparent_vs_retained_slope.pdf",
        ],
        "formats": ["PDF vector", "TIFF 600 dpi"],
        "palette": "Okabe-Ito colourblind-safe",
        "paired_auc_delta": {
            "estimate": float(delta.estimate),
            "ci_lower": float(delta.ci_lower),
            "ci_upper": float(delta.ci_upper),
        },
        "source_data": [
            "source_data/Panel_A_model_discrimination.csv",
            "source_data/Panels_B_C_calibration_stress.csv",
        ],
    }
    (args.output_dir / "VITALDB_WAVEFORM_FIGURE_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
