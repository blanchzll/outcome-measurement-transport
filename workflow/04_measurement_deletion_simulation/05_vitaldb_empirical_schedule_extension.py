# %% [markdown]
# # VitalDB extension of empirical monitoring-schedule transport
#
# Adds VitalDB as a donor and target to the existing INSPIRE/MIMIC-IV/eICU
# schedule-transport experiment. Only pairs involving VitalDB are recomputed;
# previously released three-database pairs remain unchanged.

# %%
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def vitaldb_donor_schedules(case_path: Path, serial_path: Path) -> list[np.ndarray]:
    cases = pd.read_csv(case_path, usecols=["caseid", "subjectid", "adult", "dense_reference"])
    counts = cases.groupby("subjectid")["caseid"].nunique()
    single = counts.index[counts.eq(1)]
    eligible = cases.loc[cases.adult & cases.dense_reference & cases.subjectid.isin(single), "caseid"]
    serial = pd.read_csv(serial_path, usecols=["caseid", "hours_from_opend"])
    serial = serial.loc[serial.caseid.isin(eligible) & serial.hours_from_opend.gt(0) & serial.hours_from_opend.le(168)]
    return [
        np.sort(group.hours_from_opend.to_numpy(float))
        for _, group in serial.groupby("caseid", sort=False)
        if len(group)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", required=True, type=Path)
    parser.add_argument("--case-level", required=True, type=Path)
    parser.add_argument("--serial", required=True, type=Path)
    parser.add_argument("--vitaldb-runner", required=True, type=Path)
    parser.add_argument("--empirical-module", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    code = args.framework_root / "code"
    stress = load_module(code / "ascertainment_stress.py", "ascertainment_stress_schedule_extension")
    simulation = load_module(code / "52_measurement_deletion_simulation.py", "measurement_simulation_schedule_extension")
    empirical = load_module(args.empirical_module, "empirical_schedule_extension_base")
    vitaldb_runner = load_module(args.vitaldb_runner, "vitaldb_measurement_transport_runner")

    vitaldb_patient, vitaldb_serial, vitaldb_audit = vitaldb_runner.prepare_analysis(
        args.case_level, args.serial, stress
    )
    targets = {
        "INSPIRE": simulation.prepare_inspire(),
        "MIMIC": simulation.prepare_mimic()[0:2],
        "EICU": simulation.prepare_eicu(),
        "VitalDB": (vitaldb_patient, vitaldb_serial),
    }
    schedules = empirical.donor_schedules(args.framework_root)
    schedules["VitalDB"] = vitaldb_donor_schedules(args.case_level, args.serial)

    tasks = []
    for target_name, (patient, serial) in targets.items():
        trajectories = empirical.trajectory_lookup(serial)
        patient = patient.loc[patient.reference_id.isin(trajectories)].reset_index(drop=True)
        donor_names = list(schedules) if target_name == "VitalDB" else ["VitalDB"]
        for donor_name in donor_names:
            for tolerance in (12.0, 24.0):
                for replicate in range(args.replicates):
                    tasks.append(
                        (
                            target_name,
                            donor_name,
                            patient,
                            trajectories,
                            schedules[donor_name],
                            tolerance,
                            replicate,
                        )
                    )
    nested = Parallel(n_jobs=args.jobs, backend="threading", verbose=5)(
        delayed(empirical.run_replicate)(*task, simulation) for task in tasks
    )
    raw = pd.DataFrame([row for group in nested for row in group])
    summary = empirical.summarize(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(
        args.output_dir / "VITALDB_EMPIRICAL_SCHEDULE_EXTENSION_REPLICATES_INTERNAL.csv.gz",
        index=False,
        compression="gzip",
    )
    summary.to_csv(args.output_dir / "Table_vitaldb_empirical_schedule_extension.csv", index=False)
    audit = {
        "status": "PASS",
        "replicates_per_condition": args.replicates,
        "computed_pairs": sorted({f"{target}->{donor}" for target, donor in raw[["target_database", "donor_schedule_database"]].itertuples(index=False)}),
        "target_n": {name: int(len(value[0])) for name, value in targets.items()},
        "donor_schedule_n": {name: int(len(value)) for name, value in schedules.items()},
        "vitaldb_target_audit": vitaldb_audit,
        "raw_rows": int(len(raw)),
        "summary_rows": int(len(summary)),
        "interpretation_boundary": (
            "Observed donor schedules are descriptive measurement patterns, not randomized or causal hospital policies. "
            "Nearest retained target measurements are mapped within prespecified 12-hour and 24-hour tolerances."
        ),
    }
    (args.output_dir / "VITALDB_EMPIRICAL_SCHEDULE_EXTENSION_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
