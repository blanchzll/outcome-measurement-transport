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
"""INSPIRE v1.4.2 external transportability validation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from analysis import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    RANDOM_STATE,
    TARGET,
    bootstrap_interval,
    build_estimators,
    evaluate,
    guarded_subgroups,
    harmonize_gender_values,
    net_benefit,
    paired_auc_difference,
    plot_roc_calibration,
    select_threshold_for_sensitivity,
)

LAB_ITEM_TO_FEATURE = {
    "creatinine": "PreopCr",
    "hb": "PreopHb",
    "albumin": "PreopAlb",
}
VITAL_BLOODLOSS_ITEM = "ebl"
VITAL_BLOOD_ITEMS = {"rbc", "ffp", "pc", "pheresis", "cryo"}
VITAL_VASO_ITEMS = {
    "vaso",
    "nepi",
    "epii",
    "pepi",
    "phe",
    "eph",
    "ntgi",
    "mlni",
    "dopai",
    "dobui",
    "epi",
}
DIAG_PREFIX = {
    "diabetes": ("E10", "E11", "E12", "E13", "E14"),
    "gastro": ("C16",),
    "colorectal": ("C18", "C19", "C20"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspire-root",
        required=True,
        type=Path,
        help="Path to INSPIRE v1.4.2 root directory.",
    )
    parser.add_argument(
        "--development-data",
        required=True,
        type=Path,
        help="Reference 4014-row cohort CSV used for model fitting.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--department",
        default="",
        help="Department filter in operations. Pass empty to include all departments (default: all).",
    )
    parser.add_argument("--analysis-window-hours", type=float, default=168.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def _codes_with_prefix(series: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    code = series.astype("string").str.strip().str.upper().str.replace(".", "", regex=False)
    mask = pd.Series(False, index=series.index)
    for prefix in prefixes:
        mask |= code.str.startswith(prefix)
    return mask


def _map_surgical_approach(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip().str.lower()
    mapped = pd.Series(pd.NA, index=series.index, dtype="string")
    mapped.loc[raw.eq("general")] = "1"
    mapped.loc[raw.eq("neuraxial")] = "2"
    mapped.loc[raw.eq("regional")] = "3"
    mapped.loc[raw.eq("mac")] = "4"
    return mapped


def _binary_mask_by_prefix(icd_series: pd.Series, prefix_list: tuple[str, ...]) -> pd.Series:
    return _codes_with_prefix(icd_series, prefix_list)


def load_inspire_diagnosis(root: Path) -> pd.DataFrame:
    path = root / "diagnosis.csv.gz"
    frame = pd.read_csv(path, compression="gzip", usecols=["subject_id", "icd10_cm"], low_memory=False)
    frame["subject_id"] = pd.to_numeric(frame["subject_id"], errors="coerce")
    frame["icd10_cm"] = frame["icd10_cm"].astype("string").str.strip()

    has_diabetes = _binary_mask_by_prefix(frame["icd10_cm"], DIAG_PREFIX["diabetes"])
    has_gastric = _binary_mask_by_prefix(frame["icd10_cm"], DIAG_PREFIX["gastro"])
    has_colorectal = _binary_mask_by_prefix(frame["icd10_cm"], DIAG_PREFIX["colorectal"])
    reduced = pd.DataFrame(
        {
            "subject_id": frame["subject_id"],
            "has_diabetes": has_diabetes,
            "has_gastro": has_gastric,
            "has_colorectal": has_colorectal,
        }
    ).groupby("subject_id", dropna=False).agg(
        has_diabetes=("has_diabetes", "max"),
        has_gastro=("has_gastro", "max"),
        has_colorectal=("has_colorectal", "max"),
    )
    return reduced.reset_index()


def load_inspire_operations(root: Path, department: str | None) -> pd.DataFrame:
    path = root / "operations.csv.gz"
    columns = [
        "op_id",
        "subject_id",
        "age",
        "sex",
        "weight",
        "height",
        "asa",
        "department",
        "antype",
        "opstart_time",
        "opend_time",
    ]
    frame = pd.read_csv(path, compression="gzip", usecols=columns, low_memory=False)
    frame["subject_id"] = pd.to_numeric(frame["subject_id"], errors="coerce")
    frame["op_id"] = pd.to_numeric(frame["op_id"], errors="coerce")
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame["height"] = pd.to_numeric(frame["height"], errors="coerce")
    frame["asa"] = pd.to_numeric(frame["asa"], errors="coerce")
    frame["opstart_time"] = pd.to_numeric(frame["opstart_time"], errors="coerce")
    frame["opend_time"] = pd.to_numeric(frame["opend_time"], errors="coerce")
    if department:
        frame = frame.loc[frame["department"].astype("string").str.strip().eq(department)]
    return (
        frame.dropna(subset=["op_id", "subject_id", "opstart_time", "opend_time"])
        .copy()
        .reset_index(drop=True)
    )


def load_development_for_external(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = set([TARGET] + CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + ["Gender"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Development dataset is missing required variables: {missing}")
    mapped = frame[TARGET].map({"No": 0, "Yes": 1, 0: 0, 1: 1, "0": 0, "1": 1})
    if mapped.isna().any():
        bad = sorted(frame.loc[mapped.isna(), TARGET].astype(str).unique().tolist())
        raise ValueError(f"Unrecognized target values in development: {bad}")
    frame = frame.copy()
    frame[TARGET] = mapped.astype(int)
    frame["Gender"] = harmonize_gender_values(frame["Gender"])
    for column in CONTINUOUS_FEATURES + CATEGORICAL_FEATURES:
        if column in frame.columns:
            if column in CONTINUOUS_FEATURES:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                series = frame[column].astype("object")
                frame[column] = series.replace({"<NA>": np.nan, pd.NA: np.nan})
    return frame


def load_labs_by_subject(root: Path, subject_ids: pd.Series, required_items: set[str]) -> dict[str, pd.DataFrame]:
    path = root / "labs.csv.gz"
    required_subjects = set(subject_ids.dropna().astype(int).tolist())
    buckets: dict[str, list[pd.DataFrame]] = {item: [] for item in required_items}
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=["subject_id", "chart_time", "item_name", "value"],
        dtype={"subject_id": "int64", "chart_time": "float64", "item_name": "string", "value": "string"},
        chunksize=750_000,
        low_memory=False,
    ):
        chunk = chunk.loc[chunk["subject_id"].isin(required_subjects)]
        if chunk.empty:
            continue
        chunk["item_name"] = chunk["item_name"].astype("string").str.strip().str.lower()
        chunk = chunk.loc[chunk["item_name"].isin(required_items)]
        if chunk.empty:
            continue
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["value", "chart_time", "subject_id"])
        for item in required_items:
            part = chunk.loc[chunk["item_name"].eq(item), ["subject_id", "chart_time", "value"]]
            if not part.empty:
                buckets[item].append(part)

    result: dict[str, pd.DataFrame] = {}
    for item, pieces in buckets.items():
        if not pieces:
            result[item] = pd.DataFrame(columns=["subject_id", "chart_time", "value"])
        else:
            merged = pd.concat(pieces, ignore_index=True).sort_values(["subject_id", "chart_time"])
            result[item] = merged.drop_duplicates(["subject_id", "chart_time"], keep="last")
    return result


def add_inspire_labs(opportunities: pd.DataFrame, labs: dict[str, pd.DataFrame], analysis_window_hours: float) -> pd.DataFrame:
    op_windowed = (
        opportunities[["op_id", "subject_id", "opstart_time", "opend_time"]]
        .copy()
        .reset_index(drop=True)
    )

    for source, target in LAB_ITEM_TO_FEATURE.items():
        frame_item = labs.get(source)
        if frame_item is None or frame_item.empty:
            op_windowed[target] = np.nan
            continue
        pre_values = np.full(len(op_windowed), np.nan, dtype=float)
        for subject_id, op_group in op_windowed.groupby("subject_id"):
            subject_lab = frame_item.loc[frame_item["subject_id"].eq(subject_id)].sort_values("chart_time")
            if subject_lab.empty:
                continue
            lab_times = subject_lab["chart_time"].to_numpy()
            lab_values = subject_lab["value"].to_numpy()
            for row_idx, op_time in op_group[["opstart_time"]].itertuples(index=True):
                pos = int(np.searchsorted(lab_times, op_time, side="right") - 1)
                if pos >= 0:
                    pre_values[row_idx] = lab_values[pos]
        op_windowed[target] = pre_values

    creatinine = labs.get("creatinine")
    if creatinine is not None and not creatinine.empty:
        postop = op_windowed[["op_id", "subject_id", "opend_time"]].merge(
            creatinine,
            on="subject_id",
            how="left",
        )
        postop = postop.loc[
            (postop["chart_time"] >= postop["opend_time"])
            & (postop["chart_time"] <= postop["opend_time"] + analysis_window_hours * 60.0)
        ]
        postop = postop.merge(
            op_windowed[["op_id", "PreopCr"]],
            on="op_id",
            how="left",
        )
        postop = postop.loc[postop["PreopCr"].notna() & postop["value"].notna()]
        postop["abs_48h"] = (postop["chart_time"] <= postop["opend_time"] + 48 * 60.0) & (
            (postop["value"] - postop["PreopCr"]) >= 0.3
        )
        postop["ratio_168"] = (postop["value"] / postop["PreopCr"]) >= 1.5
        aki = postop[["op_id", "abs_48h", "ratio_168"]].groupby("op_id", as_index=False).max()
        aki["PostopAKI"] = aki[["abs_48h", "ratio_168"]].max(axis=1)
        op_windowed = op_windowed.merge(aki[["op_id", "PostopAKI"]], on="op_id", how="left")
    else:
        op_windowed["PostopAKI"] = pd.NA

    return op_windowed[
        [
            "op_id",
            "PreopCr",
            "PreopHb",
            "PreopAlb",
            "PostopAKI",
        ]
    ]


def aggregate_intraop_vitals(root: Path, opportunities: pd.DataFrame) -> pd.DataFrame:
    path = root / "vitals.csv.gz"
    operation_meta = opportunities[["op_id", "opstart_time", "opend_time"]].copy()
    op_ids = set(operation_meta["op_id"].dropna().astype(int).tolist())
    op_lookup = operation_meta.set_index("op_id")

    blood_loss = defaultdict(float)
    transfusion = defaultdict(float)
    vaso = set[int]()

    need_items = {VITAL_BLOODLOSS_ITEM} | VITAL_BLOOD_ITEMS | VITAL_VASO_ITEMS
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=["op_id", "chart_time", "item_name", "value"],
        dtype={"op_id": "int64", "chart_time": "float64", "item_name": "string", "value": "string"},
        chunksize=1_000_000,
        low_memory=False,
    ):
        chunk = chunk.loc[chunk["op_id"].isin(op_ids)]
        if chunk.empty:
            continue
        chunk["item_name"] = chunk["item_name"].astype("string").str.strip().str.lower()
        chunk = chunk.loc[chunk["item_name"].isin(need_items)]
        if chunk.empty:
            continue
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["value", "chart_time", "op_id"])
        chunk = chunk.merge(op_lookup[["opstart_time", "opend_time"]], left_on="op_id", right_index=True, how="left")
        chunk = chunk.loc[
            chunk["chart_time"].between(chunk["opstart_time"], chunk["opend_time"], inclusive="both")
        ]
        if chunk.empty:
            continue

        for op_id, value in chunk.loc[chunk["item_name"] == VITAL_BLOODLOSS_ITEM, ["op_id", "value"]].to_numpy():
            blood_loss[int(op_id)] += float(value)
        for op_id, value in chunk.loc[chunk["item_name"].isin(VITAL_BLOOD_ITEMS), ["op_id", "value"]].to_numpy():
            transfusion[int(op_id)] += float(value)
        for op_id in chunk.loc[chunk["item_name"].isin(VITAL_VASO_ITEMS) & (chunk["value"] > 0), "op_id"]:
            vaso.add(int(op_id))

    frame = opportunities[["op_id"]].copy()
    frame["IntraopBloodLoss"] = frame["op_id"].map(lambda op_id: blood_loss.get(int(op_id), np.nan))
    frame["IntraopTransfusion"] = frame["op_id"].map(lambda op_id: transfusion.get(int(op_id), np.nan))
    frame["IntraopVasoactive"] = frame["op_id"].apply(lambda op_id: "1" if int(op_id) in vaso else pd.NA)
    return frame


def build_inspire_cohort(
    root: Path,
    department: str,
    analysis_window_hours: float,
    *,
    gastrocolorectal_only: bool = False,
) -> pd.DataFrame:
    operations = load_inspire_operations(root, department=department if department else None)
    if operations.empty:
        raise ValueError("INSPIRE filter returns empty cohort.")

    diagnosis_summary = load_inspire_diagnosis(root)
    feature = operations.copy()
    feature["Age"] = feature["age"]
    sex_map = feature["sex"].astype("string").str.strip().str.lower().replace({"m": "male", "f": "female"})
    feature["Gender"] = harmonize_gender_values(sex_map)
    feature["BMI"] = feature["weight"] / (feature["height"] / 100.0) ** 2
    feature["BMI"] = feature["BMI"].replace([np.inf, -np.inf], np.nan)
    feature["ASAGrade"] = feature["asa"].round().astype("Int64").astype("string")
    feature["ASAGrade"] = feature["ASAGrade"].where(feature["ASAGrade"].ne("<NA>"), pd.NA)
    feature["SurgicalApproach"] = _map_surgical_approach(feature["antype"])
    feature["OperationTime"] = feature["opend_time"] - feature["opstart_time"]
    feature.loc[feature["OperationTime"] <= 0, "OperationTime"] = np.nan

    diag = diagnosis_summary.copy()
    diag["Gastrocolorectal"] = pd.NA
    diag.loc[diag["has_gastro"], "Gastrocolorectal"] = "1"
    diag.loc[diag["has_colorectal"] & ~diag["has_gastro"], "Gastrocolorectal"] = "2"
    feature = feature.merge(
        diag[["subject_id", "has_diabetes", "Gastrocolorectal"]],
        on="subject_id",
        how="left",
    )
    feature["Diabetes"] = feature["has_diabetes"].map({True: "1", False: "0"}).astype("string")
    feature.loc[feature["Diabetes"].eq("<NA>"), "Diabetes"] = pd.NA
    if gastrocolorectal_only:
        # Filter before reading patient labs and the large intraoperative-vital
        # table. The filter depends only on diagnosis codes, not on AKI labels.
        feature = (
            feature.loc[feature["Gastrocolorectal"].notna()]
            .copy()
            .reset_index(drop=True)
        )

    labs = load_labs_by_subject(root, feature["subject_id"], set(LAB_ITEM_TO_FEATURE.keys()))
    feature = feature.merge(
        add_inspire_labs(
            feature[["op_id", "subject_id", "opstart_time", "opend_time"]],
            labs,
            analysis_window_hours=analysis_window_hours,
        ),
        on="op_id",
        how="left",
    )
    feature = feature.merge(
        aggregate_intraop_vitals(
            root,
            feature[["op_id", "opstart_time", "opend_time"]],
        ),
        on="op_id",
        how="left",
    )

    feature[TARGET] = feature["PostopAKI"].map({True: 1, False: 0, pd.NA: np.nan}).astype("float")
    for column in CONTINUOUS_FEATURES + CATEGORICAL_FEATURES:
        if column in feature.columns:
            if column in CONTINUOUS_FEATURES:
                feature[column] = pd.to_numeric(feature[column], errors="coerce")
            else:
                series = feature[column].astype("object")
                feature[column] = series.replace({"<NA>": np.nan, pd.NA: np.nan})

    cols = [
        TARGET,
        "op_id",
        "Age",
        "BMI",
        "PreopHb",
        "PreopAlb",
        "PreopCr",
        "OperationTime",
        "IntraopBloodLoss",
        "IntraopTransfusion",
        "Gender",
        "Diabetes",
        "ASAGrade",
        "Gastrocolorectal",
        "SurgicalApproach",
        "IntraopVasoactive",
    ]
    return feature[cols].copy()


def external_validation(
    development_path: Path,
    external: pd.DataFrame,
    bootstrap: int,
    fast: bool,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = load_development_for_external(development_path)
    feature_set = [name for name in CONTINUOUS_FEATURES + CATEGORICAL_FEATURES if name in development.columns]

    train_x = development[feature_set]
    train_y = development[TARGET].astype(int)
    external = external.dropna(subset=[TARGET]).copy()
    external_x = external[feature_set]
    external_y = external[TARGET].astype(int)

    if external_y.nunique() < 2:
        raise ValueError("INSPIRE external cohort has only one outcome class; cannot report ROC-based metrics safely.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if min(np.bincount(train_y)) < cv.n_splits:
        raise ValueError("Development split requires at least 5 events per class for five-fold OOF selection.")

    estimators = build_estimators(fast=fast)
    rows: list[dict] = []
    probs_external: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for index, (name, estimator) in enumerate(estimators.items(), start=1):
        print(f"[{index}/{len(estimators)}] INSPIRE external evaluation: {name}")
        oof = cross_val_predict(
            clone(estimator),
            train_x,
            train_y,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = select_threshold_for_sensitivity(train_y.to_numpy(), oof, target=0.80)
        thresholds[name] = threshold

        dev = evaluate(train_y.to_numpy(), oof, threshold)
        dev.update({"model": name, "cohort": "development_oof"})
        rows.append(dev)

        fitted = clone(estimator).fit(train_x, train_y)
        p_ext = fitted.predict_proba(external_x)[:, 1]
        probs_external[name] = p_ext
        ext = evaluate(external_y.to_numpy(), p_ext, threshold)
        ext.update({"model": name, "cohort": "external_inspire"})
        for metric_name, metric in [
            ("roc_auc", roc_auc_score),
            ("average_precision", average_precision_score),
            ("brier", brier_score_loss),
        ]:
            lower, upper = bootstrap_interval(
                external_y.to_numpy(),
                p_ext,
                metric,
                n_bootstrap=bootstrap,
                seed=RANDOM_STATE + index,
            )
            ext[f"{metric_name}_ci_lower"] = lower
            ext[f"{metric_name}_ci_upper"] = upper
        rows.append(ext)

    performance = pd.DataFrame(rows)
    external_ridge = probs_external.get("ridge_logistic")
    if external_ridge is not None and "random_forest" in probs_external:
        comp = paired_auc_difference(
            external_y.to_numpy(),
            probs_external["random_forest"],
            external_ridge,
            n_bootstrap=bootstrap,
            seed=RANDOM_STATE + 1000,
        )
        comp.update({"candidate": "random_forest", "reference": "ridge_logistic"})
        comparison = pd.DataFrame([comp])
    else:
        comparison = pd.DataFrame()

    dca_frames = []
    for name, probs in probs_external.items():
        dca = net_benefit(external_y.to_numpy(), probs, np.linspace(0.01, 0.20, 40))
        dca.insert(0, "model", name)
        dca_frames.append(dca)
    if dca_frames:
        pd.concat(dca_frames, ignore_index=True).to_csv(output / "decision_curve.csv", index=False)

    if "ridge_logistic" in probs_external and "ridge_logistic" in thresholds:
        subgroup = guarded_subgroups(
            external,
            probs_external["ridge_logistic"],
            thresholds["ridge_logistic"],
        ).assign(model="ridge_logistic")
        subgroup.to_csv(output / "subgroup_metrics_ridge.csv", index=False)

    plot_roc_calibration(external_y.to_numpy(), probs_external, output / "external_roc_calibration.png")

    return performance, comparison


def write_summary(output: Path, args: argparse.Namespace, external: pd.DataFrame) -> None:
    summary = {
        "n_operations": int(len(external)),
        "n_evaluable_with_outcome": int(external[TARGET].notna().sum()),
        "n_events": int(external.loc[external[TARGET].notna(), TARGET].sum()),
        "event_rate": float(
            (external.loc[external[TARGET].notna(), TARGET].mean() if external[TARGET].notna().any() else np.nan)
        ),
        "department_filter": args.department,
        "analysis_window_hours": args.analysis_window_hours,
        "outcome_definition": (
            "Creatinine-only AKI: >=0.3 mg/dL within 48 h after end of surgery "
            "OR >=1.5x baseline within 168 h"
        ),
        "dataset_root": str(args.inspire_root),
    }
    (output / "inspire_external_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)

    external = build_inspire_cohort(
        args.inspire_root,
        department=args.department,
        analysis_window_hours=args.analysis_window_hours,
    )
    performance, comparison = external_validation(
        args.development_data,
        external,
        bootstrap=args.bootstrap,
        fast=args.fast,
        output=args.output,
    )
    performance.to_csv(args.output / "model_performance.csv", index=False)
    comparison.to_csv(args.output / "paired_auc_differences.csv", index=False)
    external.to_csv(args.output / "inspire_cohort_features.csv", index=False)
    write_summary(args.output, args, external)
    print(f"INSPIRE validation outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
