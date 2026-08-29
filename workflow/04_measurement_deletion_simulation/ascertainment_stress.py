"""Outcome-ascertainment stress tests for longitudinal creatinine endpoints.

The module deliberately separates latent/complete operational outcomes, observation
of a sufficient measurement pattern, and reconstructed outcomes. It is not a KDIGO
adjudication engine and must not be described as one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def clip_prob(x):
    return np.clip(np.asarray(x, float), EPS, 1 - EPS)


def effective_sample_size(w) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.square(w).sum()) if np.square(w).sum() else np.nan


def weighted_metrics(y, p, w=None) -> dict:
    y, p = np.asarray(y, float), clip_prob(p)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y, p, w = y[ok], p[ok], w[ok]
    if len(y) == 0:
        return {k: np.nan for k in ("n", "events", "event_rate", "event_rate_se", "mean_prediction", "oe", "brier", "auc", "calibration_intercept", "calibration_slope", "ess")}
    event_rate = np.average(y, weights=w)
    mean_prediction = np.average(p, weights=w)
    auc = roc_auc_score(y, p, sample_weight=w) if np.unique(y).size == 2 else np.nan
    z = logit(p).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=300).fit(z, y.astype(int), sample_weight=w)
        intercept, slope = float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        intercept = slope = np.nan
    normalized_weight = w / w.sum()
    event_rate_se = float(np.sqrt(np.sum(np.square(normalized_weight) * np.square(y - event_rate))))
    return {
        "n": int(len(y)), "events": float(np.sum(w * y)), "event_rate": float(event_rate),
        "event_rate_se": event_rate_se,
        "mean_prediction": float(mean_prediction), "oe": float(event_rate / mean_prediction),
        "brier": float(np.average((y - p) ** 2, weights=w)), "auc": float(auc),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "ess": effective_sample_size(w),
    }


def recalibrate(p, y, w=None, intercept_only=False):
    p = clip_prob(p)
    y = np.asarray(y, int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    z = logit(p)
    if np.unique(y).size < 2:
        return p.copy(), np.nan, np.nan
    if intercept_only:
        # Solve weighted mean(expit(a + logit(p))) = weighted mean(y).
        target = np.average(y, weights=w)
        lo, hi = -15.0, 15.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if np.average(expit(mid + z), weights=w) < target:
                lo = mid
            else:
                hi = mid
        a, b = (lo + hi) / 2, 1.0
    else:
        fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=300).fit(z.reshape(-1, 1), y, sample_weight=w)
        a, b = float(fit.intercept_[0]), float(fit.coef_[0, 0])
    return expit(a + b * z), a, b


def _standardize(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return np.zeros_like(x) if not np.isfinite(sd) or sd < EPS else (x - np.nanmean(x)) / sd


def _mean_calibrated_probability(score, target):
    score = np.asarray(score, float)
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if expit(mid + score).mean() < target:
            lo = mid
        else:
            hi = mid
    return expit((lo + hi) / 2 + score)


@dataclass
class StressResult:
    patient: pd.DataFrame
    kept_serial: pd.DataFrame
    mean_measurement_retention: float


def delete_and_reconstruct(patient, serial, mechanism, retention, strength, rng) -> StressResult:
    """Delete individual post-landmark creatinine measurements and rebuild endpoint."""
    pcols = ["reference_id", "baseline_creatinine", "y_full", "risk", "age_z", "sex_z", "stratum_z"]
    s = serial.merge(patient[pcols], on="reference_id", how="inner", validate="many_to_one").copy()
    s = s.sort_values(["reference_id", "hour"], kind="stable")
    s["current_ratio"] = s.creatinine / s.baseline_creatinine
    risk_z = _standardize(logit(clip_prob(s.risk)))
    late_z = _standardize(np.minimum(s.hour, 168) / 168)
    y_z = _standardize(s.y_full)
    stratum = 0.6 * s.age_z.to_numpy() + 0.4 * s.sex_z.to_numpy() + 0.5 * s.stratum_z.to_numpy()
    if mechanism == "MCAR":
        score = np.zeros(len(s))
    elif mechanism == "stratum_MAR":
        score = stratum
    elif mechanism == "risk_MAR":
        score = risk_z
    elif mechanism == "history_MAR":
        score = None
    elif mechanism == "outcome_MNAR":
        score = 0.65 * y_z + 0.35 * _standardize(s.current_ratio - 1)
    elif mechanism == "mixed_MNAR":
        score = None
    else:
        raise ValueError(mechanism)
    scale = 0.65 if strength == "weak" else 1.35
    if mechanism in {"history_MAR", "mixed_MNAR"}:
        # Generate measurement sequentially. The history term is the most recent
        # *observed* creatinine ratio, never the hidden previous value from the
        # complete trajectory. Reusing one uniform draw per measurement makes the
        # intercept search deterministic and targets the requested retention.
        uniforms = rng.random(len(s))
        ratio_delta = s.current_ratio.to_numpy(dtype=float) - 1.0
        ratio_scale = float(np.nanstd(ratio_delta))
        if not np.isfinite(ratio_scale) or ratio_scale < EPS:
            ratio_scale = 1.0
        static = np.asarray(late_z, dtype=float) * 0.25
        if mechanism == "mixed_MNAR":
            static = 0.35 * np.asarray(risk_z) + 0.25 * np.asarray(stratum) + 0.2 * np.asarray(y_z)

        def sequential_draw(intercept):
            probability = np.empty(len(s), dtype=float)
            kept = np.zeros(len(s), dtype=bool)
            last_reference = None
            last_observed_ratio = 1.0
            for index, (reference_id, current_ratio) in enumerate(
                zip(s.reference_id.to_numpy(), s.current_ratio.to_numpy(dtype=float), strict=True)
            ):
                if reference_id != last_reference:
                    last_reference = reference_id
                    last_observed_ratio = 1.0
                prior_signal = np.clip((last_observed_ratio - 1.0) / ratio_scale, -4.0, 4.0)
                if mechanism == "history_MAR":
                    row_score = static[index] + 0.75 * prior_signal
                else:
                    row_score = static[index] + 0.2 * prior_signal
                probability[index] = expit(intercept + scale * row_score)
                kept[index] = uniforms[index] < probability[index]
                if kept[index]:
                    last_observed_ratio = current_ratio
            return probability, kept

        # Calibrate the intercept against the pre-history static score, then draw
        # once sequentially. Realised retention is reported rather than forced;
        # repeatedly tuning to the same random draws would condition the design
        # on its realised missingness pattern and is computationally excessive.
        lo, hi = -15.0, 15.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if expit(mid + scale * static).mean() < retention:
                lo = mid
            else:
                hi = mid
        intercept = (lo + hi) / 2
        for _ in range(6):
            probability, kept = sequential_draw(intercept)
            realised_probability = np.clip(probability.mean(), EPS, 1 - EPS)
            intercept += float(logit(retention) - logit(realised_probability))
        probability, kept = sequential_draw(intercept)
        s["measurement_probability"] = probability
        s["kept"] = kept
    else:
        s["measurement_probability"] = _mean_calibrated_probability(scale * score, retention)
        s["kept"] = rng.random(len(s)) < s.measurement_probability.to_numpy()

    def window_probability(frame):
        if len(frame) == 0:
            return 0.0
        return 1.0 - float(np.prod(1.0 - frame.measurement_probability.to_numpy()))

    def grouped_window_probability(mask, name):
        subset = s.loc[mask]
        if subset.empty:
            empty = pd.Series(dtype=float, name=name)
            empty.index.name = "reference_id"
            return empty
        return subset.groupby("reference_id", sort=False).apply(
            window_probability, include_groups=False
        ).rename(name)

    q_early = grouped_window_probability(s.hour.le(48), "q_early")
    late_window = s.hour.gt(48) & s.hour.le(96)
    q_late = grouped_window_probability(late_window, "q_late")
    kept = s.loc[s.kept].copy()
    early = kept.loc[kept.hour.le(48)].groupby("reference_id").creatinine.max().rename("max48_obs")
    allmax = kept.groupby("reference_id").creatinine.max().rename("max168_obs")
    has_early = kept.loc[kept.hour.le(48)].groupby("reference_id").size().rename("has_early")
    has_late = kept.loc[kept.hour.gt(48) & kept.hour.le(96)].groupby("reference_id").size().rename("has_late")
    out = patient.copy().merge(q_early, on="reference_id", how="left").merge(q_late, on="reference_id", how="left")
    out = out.merge(early, on="reference_id", how="left").merge(allmax, on="reference_id", how="left")
    out = out.merge(has_early, on="reference_id", how="left").merge(has_late, on="reference_id", how="left")
    out[["q_early", "q_late"]] = out[["q_early", "q_late"]].fillna(0)
    out["q_observed"] = clip_prob(out.q_early * out.q_late)
    out["R"] = (out.has_early.fillna(0).gt(0) & out.has_late.fillna(0).gt(0)).astype(int)
    out["y_reconstructed"] = np.where(
        out.R.eq(1),
        ((out.max48_obs >= out.baseline_creatinine + 0.3) |
         (out.max168_obs >= 1.5 * out.baseline_creatinine)).astype(float),
        np.nan,
    )
    return StressResult(out, kept, float(s.measurement_probability.mean()))


def aipw_event_rate(frame, outcome_model=None):
    obs = frame.R.to_numpy() == 1
    y = frame.y_reconstructed.fillna(0).to_numpy(float)
    q = clip_prob(frame.q_observed)
    m = clip_prob(frame.risk if outcome_model is None else outcome_model)
    psi = m + obs * (y - m) / q
    return float(np.mean(psi)), float(np.std(psi, ddof=1) / np.sqrt(len(psi)))


def mnar_event_bounds(frame, gamma=2.0):
    obs = frame.R.to_numpy() == 1
    y = frame.y_reconstructed.fillna(0).to_numpy(float)
    base = clip_prob(frame.risk)
    lo = expit(logit(base) - np.log(gamma))
    hi = expit(logit(base) + np.log(gamma))
    return float(np.mean(obs * y + (~obs) * lo)), float(np.mean(obs * y + (~obs) * hi))
