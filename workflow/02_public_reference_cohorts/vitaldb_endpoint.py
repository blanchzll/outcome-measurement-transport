from __future__ import annotations

import numpy as np
import pandas as pd


SECONDS_PER_HOUR = 3600.0


def build_creatinine_endpoint(
    clinical: pd.DataFrame,
    labs: pd.DataFrame,
    *,
    lower_creatinine: float = 0.1,
    upper_creatinine: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a creatinine-only postoperative AKI operational reference.

    The prediction landmark is operation end. The function intentionally does
    not label this endpoint as complete KDIGO because urine output, timed RRT,
    and expert adjudication are not available in VitalDB.
    """
    required_clinical = {"caseid", "subjectid", "opend", "age", "optype"}
    required_labs = {"caseid", "dt", "name", "result"}
    if missing := required_clinical.difference(clinical.columns):
        raise KeyError(f"Missing clinical columns: {sorted(missing)}")
    if missing := required_labs.difference(labs.columns):
        raise KeyError(f"Missing laboratory columns: {sorted(missing)}")

    case = clinical.copy()
    case["caseid"] = pd.to_numeric(case["caseid"], errors="coerce")
    case["subjectid"] = pd.to_numeric(case["subjectid"], errors="coerce")
    case["opend"] = pd.to_numeric(case["opend"], errors="coerce")
    case["age"] = pd.to_numeric(case["age"], errors="coerce")
    if "dis" in case:
        case["dis"] = pd.to_numeric(case["dis"], errors="coerce")
    if "preop_cr" in case:
        case["preop_cr"] = pd.to_numeric(case["preop_cr"], errors="coerce")
    case = case.dropna(subset=["caseid", "subjectid", "opend"]).drop_duplicates("caseid", keep=False)

    lab = labs.loc[labs["name"].astype(str).str.lower().eq("cr"), ["caseid", "dt", "result"]].copy()
    for column in ("caseid", "dt", "result"):
        lab[column] = pd.to_numeric(lab[column], errors="coerce")
    n_raw_cr = len(lab)
    lab = lab.dropna(subset=["caseid", "dt", "result"])
    lab = lab.loc[lab["result"].between(lower_creatinine, upper_creatinine, inclusive="both")]
    n_valid_cr = len(lab)
    duplicate_mask = lab.duplicated(["caseid", "dt", "result"], keep="first")
    n_duplicate_cr = int(duplicate_mask.sum())
    lab = lab.loc[~duplicate_mask]
    lab = lab.merge(case[["caseid", "opend"]], on="caseid", how="inner", validate="many_to_one")
    lab["hours_from_opend"] = (lab["dt"] - lab["opend"]) / SECONDS_PER_HOUR

    pre = lab.loc[lab["hours_from_opend"] < 0].sort_values(["caseid", "dt"])
    latest_pre = (
        pre.groupby("caseid", as_index=False).tail(1)[["caseid", "dt", "result", "hours_from_opend"]]
        .rename(
            columns={
                "dt": "baseline_dt",
                "result": "baseline_cr",
                "hours_from_opend": "baseline_hours_from_opend",
            }
        )
    )

    post = lab.loc[(lab["hours_from_opend"] > 0) & (lab["hours_from_opend"] <= 168)].copy()
    post = post.merge(latest_pre[["caseid", "baseline_cr"]], on="caseid", how="left", validate="many_to_one")
    post["aki_abs_48h"] = (post["hours_from_opend"] <= 48) & ((post["result"] - post["baseline_cr"]) >= 0.3)
    post["aki_ratio_168h"] = (post["result"] / post["baseline_cr"]) >= 1.5

    if len(post):
        summary = post.groupby("caseid").agg(
            n_postop_cr=("result", "size"),
            n_cr_0_48h=("hours_from_opend", lambda x: int(((x > 0) & (x <= 48)).sum())),
            n_cr_48_96h=("hours_from_opend", lambda x: int(((x > 48) & (x <= 96)).sum())),
            n_cr_96_168h=("hours_from_opend", lambda x: int(((x > 96) & (x <= 168)).sum())),
            first_postop_cr_h=("hours_from_opend", "min"),
            last_postop_cr_h=("hours_from_opend", "max"),
            max_postop_cr=("result", "max"),
            creatinine_event_abs_48h=("aki_abs_48h", "max"),
            creatinine_event_ratio_168h=("aki_ratio_168h", "max"),
        ).reset_index()
    else:
        summary = pd.DataFrame(
            {
                "caseid": pd.Series(dtype=float),
                "n_postop_cr": pd.Series(dtype=int),
                "n_cr_0_48h": pd.Series(dtype=int),
                "n_cr_48_96h": pd.Series(dtype=int),
                "n_cr_96_168h": pd.Series(dtype=int),
                "first_postop_cr_h": pd.Series(dtype=float),
                "last_postop_cr_h": pd.Series(dtype=float),
                "max_postop_cr": pd.Series(dtype=float),
                "creatinine_event_abs_48h": pd.Series(dtype=bool),
                "creatinine_event_ratio_168h": pd.Series(dtype=bool),
            }
        )

    result = case.merge(latest_pre, on="caseid", how="left", validate="one_to_one")
    result = result.merge(summary, on="caseid", how="left", validate="one_to_one")
    count_columns = ["n_postop_cr", "n_cr_0_48h", "n_cr_48_96h", "n_cr_96_168h"]
    for column in count_columns:
        result[column] = result[column].fillna(0).astype(int)
    for column in ("creatinine_event_abs_48h", "creatinine_event_ratio_168h"):
        result[column] = result[column].astype("boolean").fillna(False).astype(bool)

    result["has_timestamped_baseline"] = result["baseline_cr"].notna()
    result["has_postop_cr_168h"] = result["n_postop_cr"] > 0
    result["reference_observable"] = result["has_timestamped_baseline"] & result["has_postop_cr_168h"]
    result["creatinine_event_168h"] = np.where(
        result["reference_observable"],
        result["creatinine_event_abs_48h"] | result["creatinine_event_ratio_168h"],
        np.nan,
    )
    result["postop_cr_span_h"] = result["last_postop_cr_h"] - result["first_postop_cr_h"]
    result["dense_reference"] = (
        result["reference_observable"]
        & (result["n_postop_cr"] >= 3)
        & (result["n_cr_0_48h"] >= 1)
        & (result["n_cr_48_96h"] >= 1)
        & (result["postop_cr_span_h"] >= 72)
    )
    result["adult"] = result["age"] >= 18
    result["gi_stomach_colorectal"] = result["optype"].fillna("").astype(str).str.contains(
        r"stomach|colorectal", case=False, regex=True
    )
    if "dis" in result:
        result["discharge_time_valid"] = result["dis"] >= result["opend"]
        result["known_in_hospital_168h"] = result["discharge_time_valid"] & ((result["dis"] - result["opend"]) >= 168 * 3600)
    else:
        result["discharge_time_valid"] = False
        result["known_in_hospital_168h"] = False

    audit = {
        "n_raw_creatinine_rows": int(n_raw_cr),
        "n_valid_creatinine_rows": int(n_valid_cr),
        "n_exact_duplicate_creatinine_rows_removed": n_duplicate_cr,
        "n_cases_with_timestamped_preop_baseline": int(result["has_timestamped_baseline"].sum()),
        "n_cases_with_any_postop_cr_168h": int(result["has_postop_cr_168h"].sum()),
        "n_reference_observable": int(result["reference_observable"].sum()),
        "n_dense_reference": int(result["dense_reference"].sum()),
        "n_discharge_time_invalid": int((~result["discharge_time_valid"]).sum()),
    }
    return result, audit
