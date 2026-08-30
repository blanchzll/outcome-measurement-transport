# %% [markdown]
# # VitalDB GI model transported unchanged to the five-centre source cohort
#
# This is a same-model, same-landmark clinical transport analysis with a
# deliberately non-equivalent outcome reference: creatinine-only in VitalDB
# versus site-adjudicated KDIGO (creatinine, urine output, and RRT) in the source
# cohort. It must not be called same-endpoint external validation.

# %%
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260830
CONTINUOUS = ["Age", "LogPreopCr", "PreopHb"]
CATEGORICAL_BASE = ["Gender", "Diabetes"]
SITE_PREDICTOR = "Gastrocolorectal"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("source_preparer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("continuous", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), CONTINUOUS),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]), categorical),
        ]
    )
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(C=0.25, solver="lbfgs", max_iter=5000, random_state=SEED))])


def prepare_vitaldb(path: Path, scope: str) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    cases_per_patient = data.groupby("subjectid")["caseid"].nunique()
    single_operation_patients = cases_per_patient.index[cases_per_patient.eq(1)]
    data = data.loc[
        data["adult"]
        & data["dense_reference"]
        & data["subjectid"].isin(single_operation_patients)
    ].copy()
    if scope == "gi":
        data = data.loc[data["gi_stomach_colorectal"]].copy()
    gender = data["sex"].astype(str).str.strip().str.upper().map({"M": "Male", "F": "Female"})
    site = np.where(data["optype"].astype(str).str.casefold().eq("stomach"), "1", "2")
    frame = pd.DataFrame(
        {
            "PostopAKI": data["creatinine_event_168h"].astype(int),
            "Age": pd.to_numeric(data["age"], errors="coerce"),
            "PreopCr": pd.to_numeric(data["preop_cr"], errors="coerce") * 88.4,
            "PreopHb": pd.to_numeric(data["preop_hb"], errors="coerce") * 10.0,
            "Gender": gender,
            "Diabetes": pd.to_numeric(data["preop_dm"], errors="coerce").round().astype("Int64").astype("string"),
            "Gastrocolorectal": pd.Series(site, index=data.index, dtype="string"),
            "subjectid": data["subjectid"].astype(int),
        }
    )
    frame["LogPreopCr"] = np.log(frame["PreopCr"].where(frame["PreopCr"] > 0))
    frame = frame.loc[frame["PostopAKI"].isin([0, 1])].reset_index(drop=True)
    if frame["PostopAKI"].nunique() < 2:
        raise RuntimeError("VitalDB GI development cohort has fewer than two outcome classes")
    return frame


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = logit(p).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(z, y)
    prevalence = float(y.mean())
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": prevalence,
        "mean_prediction": float(p.mean()),
        "oe_ratio": float(prevalence / p.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def bootstrap(y: np.ndarray, p: np.ndarray, strata: np.ndarray | None, draws: int, seed: int) -> dict[str, tuple[float, float]]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    groups = None if strata is None else {level: np.flatnonzero(strata == level) for level in pd.unique(strata)}

    def one(child_seed):
        rng = np.random.default_rng(child_seed)
        take = rng.choice(len(y), len(y), replace=True) if groups is None else np.concatenate(
            [rng.choice(index, len(index), replace=True) for index in groups.values()]
        )
        if np.unique(y[take]).size < 2:
            return None
        try:
            return metrics(y[take], p[take])
        except Exception:
            return None

    results = Parallel(n_jobs=8, backend="threading")(
        delayed(one)(child) for child in np.random.SeedSequence(seed).spawn(draws)
    )
    valid = [item for item in results if item]
    intervals = {}
    for key in ("event_rate", "oe_ratio", "roc_auc", "average_precision", "brier", "calibration_intercept", "calibration_slope"):
        values = [item[key] for item in valid if math.isfinite(item[key])]
        intervals[key] = (
            (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) if values else (math.nan, math.nan)
        )
    return intervals


def result_row(label: str, y: np.ndarray, p: np.ndarray, strata: np.ndarray | None, draws: int, seed: int) -> dict[str, object]:
    point = metrics(y, p)
    intervals = bootstrap(y, p, strata, draws, seed)
    row: dict[str, object] = {"dataset": label, **point}
    for metric, (lower, upper) in intervals.items():
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper
    return row


def calibration_curve(label: str, y: np.ndarray, p: np.ndarray) -> list[dict[str, object]]:
    bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
    frame = pd.DataFrame({"y": y, "p": p, "bin": bins})
    return [
        {
            "dataset": label,
            "bin": i,
            "n": int(len(group)),
            "events": int(group.y.sum()),
            "mean_prediction": float(group.p.mean()),
            "observed_fraction": float(group.y.mean()),
        }
        for i, (_, group) in enumerate(frame.groupby("bin", observed=True), start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vitaldb-case-level", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--source-preparer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--development-scope", choices=["all", "gi"], default="all")
    args = parser.parse_args()

    vitaldb = prepare_vitaldb(args.vitaldb_case_level, args.development_scope)
    sys.path.insert(0, str(args.source_preparer.parent))
    source_module = load_module(args.source_preparer)
    source = source_module.prepare_development(args.source_csv).reset_index(drop=True)
    source = source.loc[source["PostopAKI"].isin([0, 1])].copy()
    source["PostopAKI"] = source["PostopAKI"].astype(int)

    categorical = CATEGORICAL_BASE + ([SITE_PREDICTOR] if args.development_scope == "gi" else [])
    predictors = CONTINUOUS + categorical
    model = make_model(categorical)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p_development = cross_val_predict(
        model, vitaldb[predictors], vitaldb["PostopAKI"], cv=folds, method="predict_proba", n_jobs=5
    )[:, 1]
    model.fit(vitaldb[predictors], vitaldb["PostopAKI"])
    p_source = model.predict_proba(source[predictors])[:, 1]

    rows = [
        result_row("VitalDB development (5-fold out-of-fold)", vitaldb.PostopAKI.to_numpy(), p_development, None, args.bootstrap, SEED),
        result_row(
            "Five-centre expert-KDIGO external cohort",
            source.PostopAKI.to_numpy(),
            p_source,
            source["Center"].to_numpy(),
            args.bootstrap,
            SEED + 1,
        ),
    ]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"vitaldb_{args.development_scope}_model_to_source"
    pd.DataFrame(rows).to_csv(out / f"Table_{stem}_transport.csv", index=False)
    curves = calibration_curve("VitalDB development (5-fold out-of-fold)", vitaldb.PostopAKI.to_numpy(), p_development)
    curves += calibration_curve("Five-centre expert-KDIGO external cohort", source.PostopAKI.to_numpy(), p_source)
    pd.DataFrame(curves).to_csv(out / f"Table_{stem}_calibration_curve.csv", index=False)
    by_center = []
    for center, index in source.groupby("Center").groups.items():
        take = np.asarray(list(index), dtype=int)
        y = source.loc[take, "PostopAKI"].to_numpy(int)
        p = p_source[take]
        row = {"center": int(center), "n": len(take), "events": int(y.sum()), "event_rate": float(y.mean()), "mean_prediction": float(p.mean()), "oe_ratio": float(y.mean() / p.mean())}
        if np.unique(y).size == 2:
            row["roc_auc"] = float(roc_auc_score(y, p))
        else:
            row["roc_auc"] = np.nan
        by_center.append(row)
    pd.DataFrame(by_center).to_csv(out / f"Table_{stem}_by_center.csv", index=False)
    joblib.dump(model, out / f"{stem}_ridge_model_INTERNAL.joblib")
    audit = {
        "analysis": f"VitalDB {args.development_scope}-surgery ridge model transported unchanged to five-centre clinical cohort",
        "prediction_landmark": "surgery end in both datasets",
        "predictors": predictors,
        "development_endpoint": f"VitalDB 0-168 h creatinine-only operational reference in dense-reference {args.development_scope}-surgery cohort",
        "validation_endpoint": "site-adjudicated KDIGO 2012 using creatinine, urine output, and RRT",
        "same_model": True,
        "same_prediction_landmark": True,
        "same_predictor_definitions_after_unit_harmonization": True,
        "same_endpoint_reference": False,
        "same_clinical_case_mix": args.development_scope == "gi",
        "claim_boundary": "independent endpoint-transport validation, not same-endpoint expert-KDIGO validation",
        "n_development": int(len(vitaldb)),
        "events_development": int(vitaldb.PostopAKI.sum()),
        "n_external": int(len(source)),
        "events_external": int(source.PostopAKI.sum()),
    }
    (out / f"{stem.upper()}_TRANSPORT_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
