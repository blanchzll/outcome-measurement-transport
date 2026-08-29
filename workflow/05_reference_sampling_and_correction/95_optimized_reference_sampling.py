#!/usr/bin/env python3
# %% [markdown]
# # Reference-standard sampling design comparison

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit


SEED = 20260829
FRACTIONS = (0.05, 0.10, 0.20)
STRATEGIES = ("random", "risk_quintile_equal", "risk_enriched", "cluster_stratified")


def load_simulation_module(base: Path):
    sys.path.insert(0, str(base / "code"))
    path = base / "code/52_measurement_deletion_simulation.py"
    spec = importlib.util.spec_from_file_location("measurement_simulation_sampling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_for(*parts: object) -> int:
    token = "|".join(map(str, parts))
    return SEED + int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 2_000_000_000


def allocate(total: int, sizes: np.ndarray) -> np.ndarray:
    sizes = sizes.astype(int)
    ideal = total * sizes / sizes.sum()
    result = np.floor(ideal).astype(int)
    result[(sizes > 0) & (result == 0)] = 1
    while result.sum() > total:
        candidates = np.flatnonzero(result > 1)
        if not len(candidates):
            break
        result[candidates[np.argmax(result[candidates] - ideal[candidates])]] -= 1
    while result.sum() < total:
        candidates = np.flatnonzero(result < sizes)
        if not len(candidates):
            break
        result[candidates[np.argmax(ideal[candidates] - result[candidates])]] += 1
    return np.minimum(result, sizes)


def stratified_sample(labels: pd.Series, total: int, rng: np.random.Generator, equal: bool) -> tuple[np.ndarray, np.ndarray]:
    levels = labels.astype("string").fillna("Missing")
    groups = [np.flatnonzero(levels.to_numpy() == level) for level in sorted(levels.unique())]
    sizes = np.array([len(group) for group in groups])
    if equal:
        target = np.repeat(total / len(groups), len(groups))
        counts = np.floor(target).astype(int)
        counts[(sizes > 0) & (counts == 0)] = 1
        counts = np.minimum(counts, sizes)
        while counts.sum() < total:
            candidates = np.flatnonzero(counts < sizes)
            counts[candidates[np.argmin(counts[candidates])]] += 1
        while counts.sum() > total:
            candidates = np.flatnonzero(counts > 1)
            counts[candidates[np.argmax(counts[candidates])]] -= 1
    else:
        counts = allocate(total, sizes)
    selected: list[np.ndarray] = []
    inclusion = np.zeros(len(labels), dtype=float)
    for group, count in zip(groups, counts, strict=True):
        inclusion[group] = count / len(group)
        if count:
            selected.append(rng.choice(group, size=count, replace=False))
    return np.concatenate(selected), inclusion


def choose_sample(frame: pd.DataFrame, fraction: float, strategy: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    total = max(30, int(np.ceil(fraction * len(frame))))
    if strategy == "random":
        selected = rng.choice(len(frame), size=total, replace=False)
        inclusion = np.repeat(total / len(frame), len(frame))
        return selected, inclusion
    if strategy == "risk_quintile_equal":
        labels = pd.qcut(frame.risk.rank(method="first"), 5, labels=False)
        return stratified_sample(labels, total, rng, equal=True)
    if strategy == "risk_enriched":
        high = frame.risk.ge(frame.risk.quantile(0.80)).map({True: "top20", False: "lower80"})
        # Equal allocation places half the reference budget in the highest-risk quintile.
        return stratified_sample(high, total, rng, equal=True)
    if strategy == "cluster_stratified":
        return stratified_sample(frame.cluster, total, rng, equal=False)
    raise ValueError(strategy)


def fit_weighted_recalibration(p: np.ndarray, y: np.ndarray, weights: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    if np.unique(y).size < 2:
        raise ValueError("Both outcome classes are required")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y, sample_weight=weights)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def one_replicate(database: str, frame: pd.DataFrame, fraction: float, strategy: str, replicate: int, module) -> dict[str, object]:
    rng = np.random.default_rng(seed_for(database, fraction, strategy, replicate))
    selected, inclusion = choose_sample(frame, fraction, strategy, rng)
    evaluation_mask = np.ones(len(frame), dtype=bool)
    evaluation_mask[selected] = False
    weights = 1 / np.clip(inclusion[selected], 1e-6, 1)
    result: dict[str, object] = {
        "database": database,
        "reference_fraction": fraction,
        "strategy": strategy,
        "replicate": replicate,
        "reference_n": int(len(selected)),
        "reference_events": int(frame.y_full.iloc[selected].sum()),
        "reference_weight_ess": float(weights.sum() ** 2 / np.square(weights).sum()),
        "fit_failed": 0,
    }
    try:
        intercept, slope = fit_weighted_recalibration(
            frame.risk.iloc[selected].to_numpy(),
            frame.y_full.iloc[selected].to_numpy(int),
            weights,
        )
        updated = expit(intercept + slope * logit(np.clip(frame.risk.to_numpy(), 1e-6, 1 - 1e-6)))
        metrics = module.weighted_metrics(frame.y_full.to_numpy()[evaluation_mask], updated[evaluation_mask])
        result.update({
            "fitted_intercept": intercept,
            "fitted_slope": slope,
            "heldout_oe": metrics["oe"],
            "heldout_calibration_slope": metrics["calibration_slope"],
            "heldout_brier": metrics["brier"],
            "absolute_log_oe": abs(float(np.log(metrics["oe"]))) if metrics["oe"] > 0 else np.nan,
        })
    except Exception:
        result["fit_failed"] = 1
    return result


def prepare_frames(base: Path, output_root: Path, module) -> dict[str, pd.DataFrame]:
    module.OUTPUTS = output_root / "outputs"
    inspire, _ = module.prepare_inspire()
    inspire_map = pd.read_csv(
        base / "secure_work/INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz",
        usecols=["reference_id", "Gastrocolorectal"],
    )
    inspire = inspire.merge(inspire_map, on="reference_id", how="left")
    inspire["cluster"] = "site_" + inspire.Gastrocolorectal.astype("string")

    mimic, _ = module.prepare_mimic()
    mimic_map = pd.read_csv(
        base / "secure_work/MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz",
        usecols=["reference_id", "calendar_year"],
    )
    mimic = mimic.merge(mimic_map, on="reference_id", how="left")
    mimic["cluster"] = "year_" + mimic.calendar_year.astype("string")

    eicu, _ = module.prepare_eicu()
    eicu_map = pd.read_csv(
        base / "eicu/secure/EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz",
        usecols=["reference_id", "hospitalid"],
    )
    eicu = eicu.merge(eicu_map, on="reference_id", how="left")
    eicu["cluster"] = "hospital_" + eicu.hospitalid.astype("string")
    return {"INSPIRE": inspire, "MIMIC": mimic, "EICU": eicu}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["database", "reference_fraction", "strategy"]
    metrics = ["reference_n", "reference_events", "reference_weight_ess", "fit_failed", "heldout_oe", "heldout_calibration_slope", "heldout_brier", "absolute_log_oe"]
    for values, group in raw.groupby(keys):
        prefix = dict(zip(keys, values))
        for metric in metrics:
            x = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(x):
                rows.append({
                    **prefix, "metric": metric, "n_replicates": len(x),
                    "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                    "q025": float(np.quantile(x, 0.025)), "q50": float(np.quantile(x, 0.5)),
                    "q975": float(np.quantile(x, 0.975)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    for name in ("secure_work", "tables", "outputs"):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)
    module = load_simulation_module(args.base)
    frames = prepare_frames(args.base, args.output_root, module)
    tasks = [
        (database, frame, fraction, strategy, replicate, module)
        for database, frame in frames.items()
        for fraction in FRACTIONS
        for strategy in STRATEGIES
        for replicate in range(args.replicates)
    ]
    rows = Parallel(n_jobs=args.jobs, backend="threading", verbose=5)(delayed(one_replicate)(*task) for task in tasks)
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_root / "secure_work/OPTIMIZED_REFERENCE_SAMPLING_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
    summary = summarize(raw)
    summary.to_csv(args.output_root / "tables/Table_optimized_reference_sampling.csv", index=False)
    audit = {
        "status": "PASS",
        "replicates_per_design": args.replicates,
        "databases": list(frames),
        "fractions": list(FRACTIONS),
        "strategies": list(STRATEGIES),
        "raw_rows": int(len(raw)),
        "summary_rows": int(len(summary)),
        "inference": "Recalibration used inverse known sampling probabilities; evaluation excluded reference-sample records.",
    }
    (args.output_root / "outputs/OPTIMIZED_REFERENCE_SAMPLING_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
