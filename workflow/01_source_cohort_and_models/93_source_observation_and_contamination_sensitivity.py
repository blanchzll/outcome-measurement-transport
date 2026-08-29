# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Source outcome-observation and preoperative-AKI contamination sensitivity
#
# This audit uses the already locked leave-one-centre-out predictions. It does not
# refit, retune, or select a model. Postoperative creatinine slots and inpatient
# length of stay are used only as outcome-observation proxies, never as predictors.
# The preoperative-AKI analysis is a partial-identification sensitivity exercise;
# it cannot determine which patients truly had AKI before surgery.

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, roc_auc_score


BASE = Path(__file__).resolve().parents[1]
PREDICTIONS = BASE / "secure_work" / "SOURCE_4014_LOCKED_LOCO_PREDICTIONS_SECURE.csv.gz"
TABLES = BASE / "tables"
OUTPUTS = BASE / "outputs"
TABLES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

MODEL = "pred_PI_restricted_rf"
SEED = 20260828


# %%
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def recalibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Return logistic recalibration intercept and slope."""
    eps = 1e-8
    lp = logit(np.clip(p.astype(float), eps, 1 - eps))
    y = y.astype(float)

    def objective(beta: np.ndarray) -> float:
        mu = expit(beta[0] + beta[1] * lp)
        mu = np.clip(mu, eps, 1 - eps)
        return float(-np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))

    fit = minimize(
        objective,
        x0=np.array([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-10.0, 10.0)],
    )
    if not fit.success:
        return np.nan, np.nan
    return float(fit.x[0]), float(fit.x[1])


def metrics(frame: pd.DataFrame, label: str, scenario: str) -> dict[str, float | str | int]:
    y = frame["PostopAKI"].astype(int).to_numpy()
    p = frame[MODEL].astype(float).to_numpy()
    intercept, slope = recalibration(y, p)
    auc = roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan
    return {
        "analysis": label,
        "scenario": scenario,
        "n": int(len(frame)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auc": float(auc),
        "oe_ratio": float(y.sum() / p.sum()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "brier_score": float(brier_score_loss(y, p)),
    }


# %%
df = pd.read_csv(PREDICTIONS, low_memory=False)
for column in [
    "Center",
    "PostopAKI",
    "PostopHospitalDays",
    "PreopCr",
    "PostopPOD1_Cr",
    "PostopPOD2_POD3_Cr",
    MODEL,
]:
    df[column] = safe_numeric(df[column])

assert len(df) == 4014
assert int(df["PostopAKI"].sum()) == 155
assert df[MODEL].notna().all()

df["pod1_cr_observed"] = df["PostopPOD1_Cr"].notna()
df["pod23_cr_observed"] = df["PostopPOD2_POD3_Cr"].notna()
df["both_postop_cr_slots_observed"] = df["pod1_cr_observed"] & df["pod23_cr_observed"]
df["postop_stay_ge7"] = df["PostopHospitalDays"] >= 7


# %% [markdown]
# ## Centre-level outcome-observation proxies

# %%
rows = []
for centre, g in df.groupby("Center", sort=True):
    rows.append(
        {
            "center": int(centre),
            "n": int(len(g)),
            "events": int(g["PostopAKI"].sum()),
            "event_rate": float(g["PostopAKI"].mean()),
            "preop_cr_observed_rate": float(g["PreopCr"].notna().mean()),
            "pod1_cr_observed_rate": float(g["pod1_cr_observed"].mean()),
            "pod2_or_3_cr_observed_rate": float(g["pod23_cr_observed"].mean()),
            "both_postop_cr_slots_observed_rate": float(g["both_postop_cr_slots_observed"].mean()),
            "postop_stay_ge7_rate": float(g["postop_stay_ge7"].mean()),
            "postop_stay_lt7_n": int((~g["postop_stay_ge7"]).sum()),
            "postop_stay_lt7_events": int(g.loc[~g["postop_stay_ge7"], "PostopAKI"].sum()),
            "interpretation": (
                "two discrete creatinine slots and inpatient stay are observation-opportunity proxies; "
                "they do not recover measurement frequency, urine-output density, RRT timing, or post-discharge surveillance"
            ),
        }
    )
opportunity = pd.DataFrame(rows)
opportunity.to_csv(TABLES / "Table_source_outcome_observation_proxy_by_center.csv", index=False)


# %% [markdown]
# ## Locked-model restriction by inpatient observation opportunity

# %%
restriction_rows = [metrics(df, "locked_LOCO_PI_restricted_rf", "all_4014")]
restriction_rows.append(
    metrics(df.loc[df["postop_stay_ge7"]].copy(), "locked_LOCO_PI_restricted_rf", "postoperative_stay_ge7_days")
)
restriction_rows.append(
    metrics(df.loc[df["both_postop_cr_slots_observed"]].copy(), "locked_LOCO_PI_restricted_rf", "both_postoperative_creatinine_slots_observed")
)
restriction_rows.append(
    metrics(
        df.loc[df["postop_stay_ge7"] & df["both_postop_cr_slots_observed"]].copy(),
        "locked_LOCO_PI_restricted_rf",
        "stay_ge7_and_both_creatinine_slots_observed",
    )
)
restriction = pd.DataFrame(restriction_rows)
restriction["interpretation"] = (
    "fixed-prediction sensitivity conditional on observed monitoring opportunity; selection may be outcome-dependent"
)
restriction.to_csv(TABLES / "Table_source_locked_model_observation_restriction.csv", index=False)


# %% [markdown]
# ## Partial-identification sensitivity to unobserved preoperative AKI
#
# The analysis excludes a specified fraction of recorded AKI-positive patients as if
# they had been ineligible because AKI preceded surgery. Random allocation is repeated
# 1000 times. Deterministic allocations use preoperative creatinine or locked risk and
# are sensitivity anchors, not claims about true preoperative AKI status.

# %%
event_index = df.index[df["PostopAKI"].eq(1)].to_numpy()
rng = np.random.default_rng(SEED)
contamination_rows = []

for fraction in [0.0, 0.05, 0.10, 0.20]:
    k = int(round(fraction * len(event_index)))
    if k == 0:
        m = metrics(df, "preoperative_AKI_contamination", "none")
        m.update({"assumed_fraction_of_recorded_events_preoperative": fraction, "allocation": "none", "replicate": 0})
        contamination_rows.append(m)
        continue

    event_frame = df.loc[event_index]
    allocations = {
        "highest_preop_creatinine": event_frame["PreopCr"].fillna(-np.inf).nlargest(k).index.to_numpy(),
        "highest_locked_risk": event_frame[MODEL].nlargest(k).index.to_numpy(),
        "lowest_locked_risk": event_frame[MODEL].nsmallest(k).index.to_numpy(),
    }
    for allocation, excluded in allocations.items():
        retained = df.drop(index=excluded)
        m = metrics(retained, "preoperative_AKI_contamination", f"exclude_{k}_recorded_events")
        m.update(
            {
                "assumed_fraction_of_recorded_events_preoperative": fraction,
                "allocation": allocation,
                "replicate": 0,
            }
        )
        contamination_rows.append(m)

    for replicate in range(1, 1001):
        excluded = rng.choice(event_index, size=k, replace=False)
        retained = df.drop(index=excluded)
        m = metrics(retained, "preoperative_AKI_contamination", f"exclude_{k}_recorded_events")
        m.update(
            {
                "assumed_fraction_of_recorded_events_preoperative": fraction,
                "allocation": "random",
                "replicate": replicate,
            }
        )
        contamination_rows.append(m)

contamination = pd.DataFrame(contamination_rows)
contamination.to_csv(TABLES / "Table_source_preoperative_AKI_contamination_replicates.csv", index=False)

summary_rows = []
group_cols = ["assumed_fraction_of_recorded_events_preoperative", "allocation"]
for keys, g in contamination.groupby(group_cols, sort=True):
    row = dict(zip(group_cols, keys))
    row.update({"replicates": int(len(g)), "n_median": float(g["n"].median()), "events_median": float(g["events"].median())})
    for metric in ["auc", "oe_ratio", "calibration_intercept", "calibration_slope", "brier_score"]:
        row[f"{metric}_median"] = float(g[metric].median())
        row[f"{metric}_q025"] = float(g[metric].quantile(0.025))
        row[f"{metric}_q975"] = float(g[metric].quantile(0.975))
    row["interpretation"] = (
        "sensitivity envelope only; cannot identify which patients had preoperative AKI or restore source eligibility"
    )
    summary_rows.append(row)
contamination_summary = pd.DataFrame(summary_rows)
contamination_summary.to_csv(TABLES / "Table_source_preoperative_AKI_contamination_sensitivity.csv", index=False)


# %%
audit = {
    "status": "PASS",
    "input": str(PREDICTIONS.relative_to(BASE)),
    "locked_model": MODEL,
    "n": int(len(df)),
    "events": int(df["PostopAKI"].sum()),
    "postoperative_stay_lt7_n": int((~df["postop_stay_ge7"]).sum()),
    "postoperative_stay_lt7_events": int(df.loc[~df["postop_stay_ge7"], "PostopAKI"].sum()),
    "outputs": [
        "tables/Table_source_outcome_observation_proxy_by_center.csv",
        "tables/Table_source_locked_model_observation_restriction.csv",
        "tables/Table_source_preoperative_AKI_contamination_replicates.csv",
        "tables/Table_source_preoperative_AKI_contamination_sensitivity.csv",
    ],
    "claim_boundary": (
        "The analyses quantify robustness and limited observation opportunity. They do not prove preoperative-AKI exclusion, "
        "complete 0-168 h ascertainment, or a causal measurement explanation for centre calibration differences."
    ),
}
(OUTPUTS / "SOURCE_OBSERVATION_AND_PREOP_AKI_SENSITIVITY_AUDIT.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, ensure_ascii=False, indent=2))
