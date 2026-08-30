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
# # VitalDB waveform feature extraction
#
# Prespecified, surgery-end-safe haemodynamic summaries for the waveform
# extension. Patient-level output remains restricted; the public audit is
# aggregate only.

# %%
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb


INTERVAL_SECONDS = 1
MAX_HOLD_SECONDS = 5
ART_TRACKS = ("Solar8000/ART_MBP", "EV1000/ART_MBP", "Solar8000/FEM_MBP")
NIBP_TRACK = "Solar8000/NIBP_MBP"
HR_TRACKS = ("Solar8000/HR", "Solar8000/PLETH_HR")
TRACKS = list(ART_TRACKS) + [NIBP_TRACK] + list(HR_TRACKS)


def bounded(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    result = values.astype(float, copy=True)
    result[~np.isfinite(result) | (result < lower) | (result > upper)] = np.nan
    return result


def short_hold(values: np.ndarray, limit: int = MAX_HOLD_SECONDS) -> np.ndarray:
    return pd.Series(values).ffill(limit=limit).to_numpy(float)


def select_track(data: np.ndarray, names: list[str], candidates: tuple[str, ...], lower: float, upper: float):
    for name in candidates:
        column = bounded(data[:, names.index(name)], lower, upper)
        if np.isfinite(column).any():
            return name, column
    return "none", np.full(data.shape[0], np.nan)


def safe_summary(values: np.ndarray) -> tuple[float, float, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmedian(valid)), float(np.nanpercentile(valid, 5)), float(np.nanstd(valid))


def extract_case(payload: tuple[int, float, float, str]) -> dict[str, object]:
    caseid, opstart, opend, path_text = payload
    path = Path(path_text)
    base = {"caseid": int(caseid), "waveform_file_present": path.is_file()}
    if not path.is_file():
        return {**base, "extraction_status": "missing_file"}
    if not np.isfinite(opstart) or not np.isfinite(opend) or opend <= opstart:
        return {**base, "extraction_status": "invalid_operation_window"}
    try:
        record = vitaldb.VitalFile(str(path), track_names=TRACKS)
        data = record.to_numpy(TRACKS, interval=INTERVAL_SECONDS)
    except Exception as exc:
        return {**base, "extraction_status": "read_failure", "error_class": type(exc).__name__}
    start = max(0, int(np.floor(opstart)))
    stop = min(len(data), int(np.ceil(opend)))
    if stop <= start:
        return {**base, "extraction_status": "window_outside_record"}
    data = data[start:stop]
    duration_seconds = stop - start

    art_name, art_raw = select_track(data, TRACKS, ART_TRACKS, 20, 200)
    hr_name, hr_raw = select_track(data, TRACKS, HR_TRACKS, 20, 250)
    nibp = bounded(data[:, TRACKS.index(NIBP_TRACK)], 20, 200)
    art = short_hold(art_raw)
    hr = short_hold(hr_raw)
    art_valid = np.isfinite(art)
    hr_valid = np.isfinite(hr)
    art_median, art_p05, art_sd = safe_summary(art)
    hr_median, _, hr_sd = safe_summary(hr)

    result: dict[str, object] = {
        **base,
        "extraction_status": "ok",
        "operation_window_seconds": int(duration_seconds),
        "art_map_track": art_name,
        "hr_track": hr_name,
        "art_map_raw_samples": int(np.isfinite(art_raw).sum()),
        "art_map_covered_seconds": int(art_valid.sum()),
        "art_map_coverage_fraction": float(art_valid.mean()),
        "art_map_median": art_median,
        "art_map_p05": art_p05,
        "art_map_sd": art_sd,
        "hr_raw_samples": int(np.isfinite(hr_raw).sum()),
        "hr_covered_seconds": int(hr_valid.sum()),
        "hr_coverage_fraction": float(hr_valid.mean()),
        "hr_median": hr_median,
        "hr_sd": hr_sd,
        "hr_above_100_fraction_observed": float(np.mean(hr[hr_valid] > 100)) if hr_valid.any() else np.nan,
        "nibp_map_count": int(np.isfinite(nibp).sum()),
        "nibp_map_median": float(np.nanmedian(nibp)) if np.isfinite(nibp).any() else np.nan,
    }
    for threshold in (65, 60, 55):
        below = art_valid & (art < threshold)
        result[f"art_map_below_{threshold}_minutes"] = float(below.sum() / 60)
        result[f"art_map_below_{threshold}_fraction_observed"] = (
            float(below.sum() / art_valid.sum()) if art_valid.any() else np.nan
        )
    deficit = np.where(art_valid, np.maximum(65 - art, 0), np.nan)
    result["art_map_deficit_65_mmHg_minutes"] = (
        float(np.nansum(deficit) / 60) if art_valid.any() else np.nan
    )
    result["art_map_twa_deficit_65"] = (
        float(np.nansum(deficit) / art_valid.sum()) if art_valid.any() else np.nan
    )
    usable = result["art_map_covered_seconds"] >= 1800 and result["art_map_coverage_fraction"] >= 0.20
    result["art_map_duration_features_usable"] = bool(usable)
    if not usable:
        for threshold in (65, 60, 55):
            result[f"art_map_below_{threshold}_minutes"] = np.nan
            result[f"art_map_below_{threshold}_fraction_observed"] = np.nan
        result["art_map_deficit_65_mmHg_minutes"] = np.nan
        result["art_map_twa_deficit_65"] = np.nan
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical-csv", required=True, type=Path)
    parser.add_argument("--waveform-dir", required=True, type=Path)
    parser.add_argument("--manifest-verification", required=True, type=Path)
    parser.add_argument("--secure-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    verification = json.loads(args.manifest_verification.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS" or verification.get("verified_files") != 6394:
        raise RuntimeError("Full official VitalDB manifest verification must pass before extraction")
    clinical = pd.read_csv(args.clinical_csv, encoding="utf-8-sig", low_memory=False)
    required = {"caseid", "opstart", "opend"}
    if not required.issubset(clinical):
        raise KeyError(f"Missing clinical columns: {sorted(required - set(clinical))}")
    payloads = [
        (
            int(row.caseid),
            float(row.opstart) if pd.notna(row.opstart) else np.nan,
            float(row.opend) if pd.notna(row.opend) else np.nan,
            str(args.waveform_dir / f"{int(row.caseid):04d}.vital"),
        )
        for row in clinical[["caseid", "opstart", "opend"]].itertuples(index=False)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_case, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.append(future.result())
    features = pd.DataFrame(rows).sort_values("caseid").reset_index(drop=True)
    args.secure_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    features.to_csv(args.secure_output, index=False, compression="gzip")

    ok = features.extraction_status.eq("ok")
    usable = features.get("art_map_duration_features_usable", pd.Series(False, index=features.index)).fillna(False)
    coverage = pd.to_numeric(features.get("art_map_coverage_fraction"), errors="coerce")
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "VitalDB 1.0.0",
        "manifest_verification": "PASS; 6394 official objects",
        "prediction_landmark": "opend",
        "n_clinical_cases": int(len(clinical)),
        "n_extraction_ok": int(ok.sum()),
        "status_counts": {
            str(key): int(value)
            for key, value in features.extraction_status.value_counts(dropna=False).items()
        },
        "n_usable_art_map_duration_features": int(usable.sum()),
        "usable_art_map_percent": float(100 * usable.mean()),
        "art_map_coverage_median": float(coverage.median()) if coverage.notna().any() else None,
        "art_map_track_counts": {
            str(key): int(value)
            for key, value in features.loc[ok, "art_map_track"].value_counts().items()
        },
        "hr_track_counts": {
            str(key): int(value)
            for key, value in features.loc[ok, "hr_track"].value_counts().items()
        },
        "feature_protocol": {
            "interval_seconds": INTERVAL_SECONDS,
            "maximum_forward_hold_seconds": MAX_HOLD_SECONDS,
            "art_map_valid_range_mmHg": [20, 200],
            "heart_rate_valid_range_bpm": [20, 250],
            "duration_feature_minimum_seconds": 1800,
            "duration_feature_minimum_coverage": 0.20,
        },
        "patient_level_output_public": False,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
