# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""MIMIC transportability analysis as a concurrent exploratory external stress test."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from analysis import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    TARGET,
    RANDOM_STATE,
    N_JOBS,
    bootstrap_interval,
    build_estimators,
    evaluate,
    harmonize_gender_values,
    net_benefit,
    paired_auc_difference,
    plot_roc_calibration,
    select_threshold_for_sensitivity,
    save_json,
    load_cohort,
)


LAB_ITEM_TO_FEATURE = {
    "PreopCr": {50912, 52546, 52024},
    "PreopHb": {51222, 51640, 51641, 50811},
    "PreopAlb": {50862, 53085},
}

DIAG_GASTRO_PREFIX = ("C16", "151", "152")
DIAG_COLORECT_PREFIX = ("C18", "C19", "C20", "153", "154")
DIAG_DIABETES_PREFIX_ICD9 = ("250",)
DIAG_DIABETES_PREFIX_ICD10 = ("E08", "E09", "E10", "E11", "E12", "E13", "E14")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", required=True, type=Path)
    parser.add_argument("--development-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--analysis-window-hours", type=float, default=168.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--preop-window-hours", type=float, default=24.0)
    return parser.parse_args()


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _icd_prefix_match(code_series: pd.Series, prefixes: Iterable[str]) -> pd.Series:
    text = (
        code_series.astype("string")
        .str.strip()
        .str.upper()
        .fillna("")
        .str.replace(".", "", regex=False)
    )
    mask = pd.Series(False, index=text.index)
    for prefix in prefixes:
        mask |= text.str.startswith(prefix)
    return mask


def _clean_icd(code: pd.Series, version: pd.Series) -> pd.Series:
    return (
        pd.DataFrame(
            {
                "code": code.astype("string").str.strip().str.upper().fillna(""),
                "ver": pd.to_numeric(version, errors="coerce").fillna(-1).astype(int),
            }
        )
        .astype(str)
        .assign()
    )


def _load_diagnosis_flags(
    mimic_root: Path,
) -> tuple[pd.DataFrame, set[int], set[int]]:
    path = mimic_root / "hosp/diagnoses_icd.csv.gz"
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=["hadm_id", "icd_code", "icd_version"],
        chunksize=800_000,
        low_memory=False,
    ):
        chunk = chunk.loc[chunk["hadm_id"].notna()].copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        hadm_id = chunk["hadm_id"]
        codes = chunk["icd_code"].astype("string").str.strip().str.upper().str.replace(".", "", regex=False).fillna("")
        version = pd.to_numeric(chunk["icd_version"], errors="coerce").fillna(-1).astype(int)
        gastro = ((version == 9) & _icd_prefix_match(codes, DIAG_GASTRO_PREFIX[:2])) | (
            (version == 10) & _icd_prefix_match(codes, DIAG_GASTRO_PREFIX[0:1])
        )
        colorectal = ((version == 9) & _icd_prefix_match(codes, DIAG_COLORECT_PREFIX[-2:])) | (
            (version == 10) & _icd_prefix_match(codes, DIAG_COLORECT_PREFIX[:3])
        )
        diabetes = ((version == 9) & _icd_prefix_match(codes, DIAG_DIABETES_PREFIX_ICD9)) | (
            (version == 10) & _icd_prefix_match(codes, DIAG_DIABETES_PREFIX_ICD10)
        )
        frame = pd.DataFrame(
            {
                "hadm_id": hadm_id,
                "gastro": gastro.astype("int8"),
                "colorectal": colorectal.astype("int8"),
                "diabetes": diabetes.astype("int8"),
            }
        )
        chunks.append(frame)

    diag = pd.concat(chunks, ignore_index=True)
    agg = diag.groupby("hadm_id", as_index=False).max()
    agg["has_target_cancer"] = (agg["gastro"] > 0) | (agg["colorectal"] > 0)
    cancer = agg.loc[agg["has_target_cancer"], ["hadm_id", "gastro", "colorectal", "diabetes"]].copy()
    hadm_ids = set(cancer["hadm_id"].astype(int).tolist())
    return cancer.reset_index(drop=True), hadm_ids, set(cancer.loc[cancer["diabetes"] > 0, "hadm_id"].astype(int))


def _load_admissions(mimic_root: Path, hadm_ids: set[int]) -> pd.DataFrame:
    path = mimic_root / "hosp/admissions.csv.gz"
    frame = pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"],
        low_memory=False,
    )
    frame["hadm_id"] = pd.to_numeric(frame["hadm_id"], errors="coerce").astype("Int64")
    frame = frame.loc[frame["hadm_id"].notna()].copy()
    frame["hadm_id"] = frame["hadm_id"].astype(int)
    frame = frame.loc[frame["hadm_id"].isin(hadm_ids)].copy()
    frame["admittime"] = _to_datetime(frame["admittime"])
    frame["dischtime"] = _to_datetime(frame["dischtime"])
    return frame.reset_index(drop=True)


def _load_patients(mimic_root: Path, subject_ids: Iterable[int]) -> pd.DataFrame:
    path = mimic_root / "hosp/patients.csv.gz"
    frame = pd.read_csv(
        path,
        usecols=["subject_id", "gender", "anchor_age"],
        low_memory=False,
    )
    frame["subject_id"] = pd.to_numeric(frame["subject_id"], errors="coerce").astype("Int64")
    frame = frame.loc[frame["subject_id"].notna()].copy()
    frame["subject_id"] = frame["subject_id"].astype(int)
    frame = frame.loc[frame["subject_id"].isin(set(subject_ids))].copy()
    frame["Age"] = pd.to_numeric(frame["anchor_age"], errors="coerce")
    frame["Gender"] = harmonize_gender_values(frame["gender"])
    return frame.reset_index(drop=True)


def _load_procedure_anchor(mimic_root: Path, hadm_ids: set[int]) -> pd.DataFrame:
    path = mimic_root / "hosp/procedures_icd.csv.gz"
    proc = pd.read_csv(
        path,
        usecols=["hadm_id", "chartdate"],
        low_memory=False,
    )
    proc["hadm_id"] = pd.to_numeric(proc["hadm_id"], errors="coerce").astype("Int64")
    proc = proc.loc[proc["hadm_id"].notna()].copy()
    proc["hadm_id"] = proc["hadm_id"].astype(int)
    proc = proc.loc[proc["hadm_id"].isin(hadm_ids)].copy()
    proc["operation_time"] = _to_datetime(proc["chartdate"]).dt.floor("D")
    anchor = proc.groupby("hadm_id", as_index=False)["operation_time"].min()
    return anchor


def _extract_labs_by_hadm(
    mimic_root: Path,
    hadm_ids: set[int],
    lab_items: set[int],
) -> pd.DataFrame:
    path = mimic_root / "hosp/labevents.csv.gz"
    rows = []
    for chunk in pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        chunk = chunk.loc[
            (chunk["hadm_id"].notna()) & (chunk["itemid"].isin(lab_items))
        ].copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.loc[chunk["hadm_id"].notna() & chunk["hadm_id"].isin(hadm_ids)]
        if chunk.empty:
            continue
        chunk["charttime"] = _to_datetime(chunk["charttime"])
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "valuenum", "hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        rows.append(chunk[["hadm_id", "itemid", "charttime", "valuenum"]])
    if not rows:
        return pd.DataFrame(columns=["hadm_id", "itemid", "charttime", "valuenum"])
    return pd.concat(rows, ignore_index=True)


def _build_feature_map(frame: pd.DataFrame, op_time, admit_time, preop_window_hours: float) -> dict[str, float | np.ndarray]:
    result: dict[str, float | np.ndarray] = {}
    if frame.empty:
        return {
            "PreopCr": np.nan,
            "PreopHb": np.nan,
            "PreopAlb": np.nan,
            "PostopAKI": np.nan,
        }

    chart = frame.sort_values("charttime").copy()
    chart["hours_from_op"] = (chart["charttime"] - op_time).dt.total_seconds() / 3600.0
    chart["hours_from_admit"] = (chart["charttime"] - admit_time).dt.total_seconds() / 3600.0

    baseline_time: dict[str, pd.Timestamp | pd.NaT] = {}
    for feature, items in LAB_ITEM_TO_FEATURE.items():
        feature_rows = chart.loc[chart["itemid"].isin(items)].copy()
        if feature_rows.empty:
            result[feature] = np.nan
            baseline_time[feature] = pd.NaT
            continue
        preop_window = feature_rows.loc[
            (feature_rows["hours_from_op"] >= -preop_window_hours)
            & (feature_rows["hours_from_op"] <= 0)
        ]
        if preop_window.empty:
            preop_window = feature_rows.loc[
                (feature_rows["hours_from_admit"] >= 0)
                & (feature_rows["hours_from_admit"] <= preop_window_hours)
            ]
        if preop_window.empty:
            result[feature] = np.nan
            baseline_time[feature] = pd.NaT
            continue
        preop_record = preop_window.iloc[0]
        result[feature] = float(preop_record["valuenum"])
        baseline_time[feature] = preop_record["charttime"]

    # Creatinine-only postoperative AKI definition inside 168h after op start:
    cr_rows = chart.loc[chart["itemid"].isin(LAB_ITEM_TO_FEATURE["PreopCr"])].copy()
    if cr_rows.empty or not np.isfinite(result["PreopCr"]) or pd.isna(baseline_time["PreopCr"]):
        result["PostopAKI"] = np.nan
    else:
        postop = cr_rows.loc[
            (cr_rows["hours_from_op"] >= 0) & (cr_rows["hours_from_op"] <= 168.0)
        ].copy()
        if postop.empty:
            result["PostopAKI"] = np.nan
        else:
            baseline_val = float(result["PreopCr"])
            baseline_t = pd.Timestamp(baseline_time["PreopCr"])
            postop["hours_from_baseline"] = (postop["charttime"] - baseline_t).dt.total_seconds() / 3600.0
            post48 = postop.loc[postop["hours_from_baseline"] <= 48]
            criterion_abs = (
                (post48["valuenum"] - baseline_val >= 0.3).any()
            )
            criterion_ratio = (postop["valuenum"] >= baseline_val * 1.5).any()
            result["PostopAKI"] = float(criterion_abs or criterion_ratio)

    return result


def build_mimic_cohort(
    mimic_root: Path,
    preop_window_hours: float,
) -> pd.DataFrame:
    diagnosis_flags, hadm_ids, diabetes_hadm = _load_diagnosis_flags(mimic_root)
    if diagnosis_flags.empty:
        raise ValueError("No cancer-labelled admissions were identified from MIMIC diagnoses.")

    admissions = _load_admissions(mimic_root, hadm_ids)
    if admissions.empty:
        raise ValueError("No target cancer admissions found in admissions after diagnosis filtering.")
    patients = _load_patients(mimic_root, admissions["subject_id"].astype(int).unique())
    procedure_anchor = _load_procedure_anchor(mimic_root, hadm_ids)

    cohort = diagnosis_flags.merge(admissions, on="hadm_id", how="inner")
    cohort = cohort.merge(patients, on="subject_id", how="left")
    cohort = cohort.merge(procedure_anchor, on="hadm_id", how="left")

    all_lab_items = set().union(*LAB_ITEM_TO_FEATURE.values())
    lab_rows = _extract_labs_by_hadm(mimic_root, set(cohort["hadm_id"].astype(int).tolist()), all_lab_items)
    if not lab_rows.empty:
        lab_rows = lab_rows.merge(
            cohort[["hadm_id", "operation_time", "admittime"]].rename(
                columns={"admittime": "admission_time"}
            ),
            on="hadm_id",
            how="left",
        )
    feature_rows = []
    for hadm_id, group in cohort.groupby("hadm_id"):
        op_time = pd.Timestamp(group["operation_time"].iloc[0])
        admit_time = pd.Timestamp(group["admittime"].iloc[0])
        if pd.isna(op_time):
            op_time = admit_time
        if pd.isna(op_time):
            continue
        lab_for_hadm = lab_rows.loc[lab_rows["hadm_id"] == int(hadm_id)] if not lab_rows.empty else pd.DataFrame(
            columns=["hadm_id", "itemid", "charttime", "valuenum", "operation_time", "admission_time"]
        )
        feature_map = _build_feature_map(lab_for_hadm, op_time, admit_time, preop_window_hours)
        row = {
            "hadm_id": int(hadm_id),
            "subject_id": int(group["subject_id"].iloc[0]),
            "Age": group["Age"].iloc[0] if pd.notna(group["Age"].iloc[0]) else np.nan,
            "Gender": group["Gender"].iloc[0],
            "Diabetes": ("1" if int(hadm_id) in diabetes_hadm else "0"),
            "Gastrocolorectal": (
                "1"
                if int(group["gastro"].iloc[0]) == 1
                else ("2" if int(group["colorectal"].iloc[0]) == 1 else pd.NA)
            ),
            "operation_time": op_time,
            "admittime": admit_time,
        }
        row.update(feature_map)
        feature_rows.append(row)

    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise RuntimeError("Derived MIMIC feature table is empty.")

    # Model-consistent feature alignment: keep required core fields, set unavailable ones missing.
    for field in CONTINUOUS_FEATURES + CATEGORICAL_FEATURES:
        if field not in features.columns:
            features[field] = np.nan
    features["ASAGrade"] = pd.Series([pd.NA] * len(features), dtype="string")
    features["OperationTime"] = pd.Series([np.nan] * len(features), dtype=float)
    features["IntraopBloodLoss"] = pd.Series([np.nan] * len(features), dtype=float)
    features["IntraopTransfusion"] = pd.Series([np.nan] * len(features), dtype=float)
    features["IntraopVasoactive"] = pd.Series([pd.NA] * len(features), dtype="string")
    features["SurgicalApproach"] = pd.Series([pd.NA] * len(features), dtype="string")
    if "PostopAKI" not in features.columns:
        features["PostopAKI"] = np.nan
    features[TARGET] = pd.to_numeric(features["PostopAKI"], errors="coerce")
    for column in features.columns:
        if pd.api.types.is_string_dtype(features[column].dtype):
            series = features[column].astype("object")
            features[column] = series.where(pd.notna(series), np.nan)
        else:
            features[column] = features[column].where(features[column].notna(), np.nan)
    features = features.drop(columns=["operation_time", "admittime"])
    return features[["hadm_id", TARGET] + CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + ["subject_id"]].copy()


def _coerce_for_ml(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for col in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[col].dtype):
            cleaned[col] = cleaned[col].astype("object").replace({pd.NA: np.nan})
        else:
            cleaned[col] = cleaned[col].replace({pd.NA: np.nan})
    return cleaned


def build_feature_coverage(frame: pd.DataFrame) -> dict[str, int | float]:
    return {
        "n_total": int(len(frame)),
        "n_evaluable_target": int(pd.notna(frame[TARGET]).sum()),
        "events": int(frame[TARGET].sum(skipna=True)),
        "event_rate": float(frame.loc[frame[TARGET].notna(), TARGET].mean())
        if frame[TARGET].notna().any()
        else np.nan,
        "missing_rate_age": float(frame["Age"].isna().mean()),
        "missing_rate_gender": float(frame["Gender"].isna().mean()),
        "missing_rate_preop_cr": float(frame["PreopCr"].isna().mean()),
        "missing_rate_preop_hb": float(frame["PreopHb"].isna().mean()),
        "missing_rate_preop_alb": float(frame["PreopAlb"].isna().mean()),
    }


def external_validation(
    development_path: Path,
    external: pd.DataFrame,
    bootstrap: int,
    fast: bool,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development = load_cohort(development_path)
    required = set([TARGET] + CONTINUOUS_FEATURES + CATEGORICAL_FEATURES)
    missing = sorted(required - set(development.columns))
    if missing:
        raise ValueError(f"Development dataset missing required columns: {missing}")

    dev_x = development[[col for col in CONTINUOUS_FEATURES + CATEGORICAL_FEATURES if col in development.columns]].copy()
    dev_x = _coerce_for_ml(dev_x)
    dev_y = pd.to_numeric(development[TARGET], errors="coerce").astype(int)
    ext = external.dropna(subset=[TARGET]).copy()
    ext_x = ext[[col for col in dev_x.columns if col in ext.columns]].copy()
    ext_x = _coerce_for_ml(ext_x)
    ext_y = pd.to_numeric(ext[TARGET], errors="coerce").astype(int)
    # Keep alignment exactly as development feature columns.
    missing_features = [col for col in dev_x.columns if col not in ext_x.columns]
    if missing_features:
        raise RuntimeError(
            "MIMIC derived external dataset lacks required development feature columns: "
            + ", ".join(missing_features)
        )

    if ext_y.nunique() < 2:
        raise ValueError("MIMIC external cohort has only one outcome class; cannot compute ROC safely.")
    if min(np.bincount(ext_y)) == 0:
        raise ValueError("MIMIC external cohort has one empty class after label coercion.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if min(np.bincount(dev_y)) < cv.n_splits:
        raise ValueError("Development dataset has insufficient events to fit five-fold OOF selection.")

    estimators = build_estimators(fast=fast)
    rows = []
    probs_external = {}
    thresholds = {}
    for index, (name, estimator) in enumerate(estimators.items(), start=1):
        print(f"[{index}/{len(estimators)}] MIMIC external evaluation: {name}")
        oof = cross_val_predict(
            clone(estimator),
            dev_x,
            dev_y,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = select_threshold_for_sensitivity(dev_y.to_numpy(), oof, target=0.80)
        thresholds[name] = threshold

        dev_metrics = evaluate(dev_y.to_numpy(), oof, threshold)
        dev_metrics.update({"model": name, "cohort": "development_oof"})
        rows.append(dev_metrics)

        fitted = clone(estimator).fit(dev_x, dev_y)
        ext_prob = fitted.predict_proba(ext_x)[:, 1]
        probs_external[name] = ext_prob
        ext_metrics = evaluate(ext_y.to_numpy(), ext_prob, threshold)
        ext_metrics.update({"model": name, "cohort": "external_mimic"})
        bootstrap_pairs = bootstrap if not fast else min(bootstrap, 100)
        for metric_name, metric in [
            ("roc_auc", roc_auc_score),
            ("average_precision", average_precision_score),
            ("brier", brier_score_loss),
        ]:
            lower, upper = bootstrap_interval(
                ext_y.to_numpy(),
                ext_prob,
                metric,
                n_bootstrap=bootstrap_pairs,
                seed=RANDOM_STATE + index,
            )
            ext_metrics[f"{metric_name}_ci_lower"] = lower
            ext_metrics[f"{metric_name}_ci_upper"] = upper
        rows.append(ext_metrics)

    performance = pd.DataFrame(rows)
    ext_ridge = probs_external.get("ridge_logistic")
    comparisons = []
    if ext_ridge is not None and "random_forest" in probs_external:
        diff = paired_auc_difference(
            ext_y.to_numpy(),
            probs_external["random_forest"],
            ext_ridge,
            n_bootstrap=bootstrap,
            seed=RANDOM_STATE + 10,
        )
        diff.update({"candidate": "random_forest", "reference": "ridge_logistic"})
        comparisons.append(diff)
    comparison = pd.DataFrame(comparisons)

    dca_rows = []
    for name, probs in probs_external.items():
        frame = net_benefit(ext_y.to_numpy(), probs, np.linspace(0.01, 0.20, 40))
        frame.insert(0, "model", name)
        dca_rows.append(frame)
    if dca_rows:
        pd.concat(dca_rows, ignore_index=True).to_csv(output / "decision_curve.csv", index=False)

    if "ridge_logistic" in probs_external and "ridge_logistic" in thresholds:
        from analysis import guarded_subgroups

        subgroup = guarded_subgroups(
            ext,
            probs_external["ridge_logistic"],
            thresholds["ridge_logistic"],
        )
        subgroup.assign(model="ridge_logistic").to_csv(
            output / "subgroup_metrics_ridge.csv", index=False
        )

    plot_roc_calibration(
        ext_y.to_numpy(),
        probs_external,
        output / "external_roc_calibration.png",
    )
    return performance, comparison, ext


def write_summary(output: Path, args: argparse.Namespace, cohort: pd.DataFrame) -> None:
    coverage = build_feature_coverage(cohort)
    summary = {
        "dataset": "MIMIC (selected gastric/colorectal admissions)",
        "mimic_root": str(args.mimic_root),
        "analysis_window_hours": args.analysis_window_hours,
        "preop_window_hours": args.preop_window_hours,
        "n_candidates": int(coverage["n_total"]),
        "n_evaluable_with_outcome": int(coverage["n_evaluable_target"]),
        "n_events": int(coverage["events"]),
        "event_rate": coverage["event_rate"],
        "missing_rate_age": coverage["missing_rate_age"],
        "missing_rate_gender": coverage["missing_rate_gender"],
        "missing_rate_precognition_creatinine": coverage["missing_rate_preop_cr"],
        "missing_rate_precognition_hemoglobin": coverage["missing_rate_preop_hb"],
        "missing_rate_precognition_albumin": coverage["missing_rate_preop_alb"],
        "status": (
            "Exploratory transportability stress test only; AKI endpoint derives from MIMIC "
            "creatinine changes within 168h and is not equivalent to source-recorded PostopAKI."
        ),
        "feature_overlap_reported": (
            "Continuous features: "
            + ", ".join([name for name in CONTINUOUS_FEATURES if cohort[name].notna().any()])
        ),
        "categorical_features_reported": (
            "Categorical features: "
            + ", ".join([name for name in CATEGORICAL_FEATURES if cohort[name].notna().any()])
        ),
    }
    (output / "mimic_external_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be at least 1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cohort = build_mimic_cohort(args.mimic_root, preop_window_hours=args.preop_window_hours)
    if cohort[TARGET].notna().sum() < 20:
        raise ValueError(
            "Insufficient evaluable MIMIC outcomes for meaningful external ROC comparison (minimum 20)."
        )

    performance, comparison, cohort_for_eval = external_validation(
        development_path=args.development_data,
        external=cohort,
        bootstrap=args.bootstrap,
        fast=args.fast,
        output=output,
    )
    performance.to_csv(output / "model_performance.csv", index=False)
    comparison.to_csv(output / "paired_auc_differences.csv", index=False)
    cohort[[TARGET] + CONTINUOUS_FEATURES + CATEGORICAL_FEATURES].to_csv(
        output / "mimic_cohort_features.csv",
        index=False,
    )
    write_summary(output, args, cohort_for_eval)
    print(f"MIMIC stress-test outputs saved to: {output}")


if __name__ == "__main__":
    main()
