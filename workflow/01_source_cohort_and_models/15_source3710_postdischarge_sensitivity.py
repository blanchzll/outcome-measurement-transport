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
# # Post-discharge event sensitivity in the locked 3,710-patient cohort
#
# This partial-identification analysis does not recover unrecorded events. It
# quantifies how explicit assumptions about recorded-negative short stays would
# alter locked-prediction AUC and O/E.

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from release_paths import release_path

INPUT = release_path("analysis", "secure_work/SOURCE_3710_LOCKED_LOCO_PREDICTIONS_SECURE.csv.gz")
OUT = release_path("analysis", "outputs/source3710_postdischarge_sensitivity")
MODEL = "pred_PI_restricted_rf"
FRACTIONS = (0.0, 0.02, 0.05, 0.10, 0.20)
REPS = 1000
SEED = 20260830


def metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    auc = roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan
    oe = float(y.sum() / p.sum())
    return float(auc), oe


# %%
d = pd.read_csv(INPUT, low_memory=False)
if len(d) != 3710 or int(d["PostopAKI"].sum()) != 152:
    raise ValueError("Expected the locked 3,710-patient, 152-event cohort.")
if d["MajorID"].nunique() != 3710:
    raise ValueError("MajorID must identify one operation per unique patient.")

y0 = d["PostopAKI"].astype(int).to_numpy()
p = np.clip(pd.to_numeric(d[MODEL], errors="raise").to_numpy(float), 1e-8, 1 - 1e-8)
days = pd.to_numeric(d["PostopHospitalDays"], errors="coerce")
eligible = np.flatnonzero(days.lt(7).to_numpy() & (y0 == 0))
if eligible.size != 162:
    raise ValueError(f"Expected 162 recorded-negative short stays; found {eligible.size}.")

rng = np.random.default_rng(SEED)
rows: list[dict[str, object]] = []
for fraction in FRACTIONS:
    n_assigned = int(round(fraction * eligible.size))
    high_risk = eligible[np.argsort(p[eligible])[::-1][:n_assigned]]
    y_high = y0.copy()
    y_high[high_risk] = 1
    auc_high, oe_high = metrics(y_high, p)
    rows.append(
        {
            "assumed_event_fraction": fraction,
            "assignment": "highest_predicted_risk",
            "replicate": 0,
            "n_assigned": n_assigned,
            "auc": auc_high,
            "oe": oe_high,
        }
    )
    for replicate in range(REPS):
        chosen = rng.choice(eligible, size=n_assigned, replace=False)
        y = y0.copy()
        y[chosen] = 1
        auc, oe = metrics(y, p)
        rows.append(
            {
                "assumed_event_fraction": fraction,
                "assignment": "random",
                "replicate": replicate + 1,
                "n_assigned": n_assigned,
                "auc": auc,
                "oe": oe,
            }
        )

replicates = pd.DataFrame(rows)
summary = (
    replicates.groupby(["assumed_event_fraction", "assignment", "n_assigned"], as_index=False)
    .agg(
        auc_mean=("auc", "mean"),
        auc_q025=("auc", lambda x: x.quantile(0.025)),
        auc_q975=("auc", lambda x: x.quantile(0.975)),
        oe_mean=("oe", "mean"),
        oe_q025=("oe", lambda x: x.quantile(0.025)),
        oe_q975=("oe", lambda x: x.quantile(0.975)),
        n_replicates=("replicate", "size"),
    )
)

# %%
tables = OUT / "tables"
figures = OUT / "figures"
secure = OUT / "secure_work"
for directory in (tables, figures, secure):
    directory.mkdir(parents=True, exist_ok=True)
replicates.to_csv(
    secure / "SOURCE3710_POSTDISCHARGE_SENSITIVITY_REPLICATES_INTERNAL.csv.gz",
    index=False,
    compression="gzip",
)
summary.to_csv(tables / "Table_source3710_postdischarge_sensitivity.csv", index=False)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
colors = {"random": "#0072B2", "highest_predicted_risk": "#D55E00"}
labels = {"random": "Random assignment", "highest_predicted_risk": "Highest predicted risk"}
for assignment, group in summary.groupby("assignment"):
    x = 100 * group["assumed_event_fraction"].to_numpy(float)
    axes[0].plot(x, group["oe_mean"], marker="o", color=colors[assignment], label=labels[assignment])
    axes[1].plot(x, group["auc_mean"], marker="o", color=colors[assignment], label=labels[assignment])
    if assignment == "random":
        axes[0].fill_between(x, group["oe_q025"], group["oe_q975"], color=colors[assignment], alpha=0.18)
        axes[1].fill_between(x, group["auc_q025"], group["auc_q975"], color=colors[assignment], alpha=0.18)
axes[0].axhline(1, color="#555555", linestyle="--", linewidth=0.8)
axes[0].set_ylabel("Observed-to-expected ratio")
axes[1].set_ylabel("Area under the ROC curve")
for label, ax in zip(("a", "b"), axes):
    ax.set_xlabel("Assumed events among short-stay recorded negatives (%)")
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
axes[1].legend(frameon=False, loc="best")
fig.tight_layout()
fig.savefig(figures / "SupplementaryFigure12_source3710_postdischarge_sensitivity.pdf")
fig.savefig(figures / "SupplementaryFigure12_source3710_postdischarge_sensitivity.tiff", dpi=600)
plt.close(fig)

audit = {
    "analysis": "assumption-based post-discharge sensitivity bound",
    "n": int(len(d)),
    "events": int(y0.sum()),
    "short_stays_lt7_days": int(days.lt(7).sum()),
    "short_stay_recorded_negatives": int(eligible.size),
    "fractions": list(FRACTIONS),
    "random_replicates": REPS,
    "seed": SEED,
    "prediction": MODEL,
    "boundary": "No unrecorded event is identified; scenarios are partial-identification assumptions.",
}
(OUT / "SOURCE3710_POSTDISCHARGE_SENSITIVITY_AUDIT.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2))
