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
# # Source-cohort date extraction and temporal-split utilities
#
# The authoritative workbook is read without exporting names, medical-record
# numbers, or exact patient-level dates. Only the six fields required for the
# integrity audit are extracted in memory.

# %%
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REQUIRED_COLUMN_NUMBERS = {
    1: "Center",
    4: "MajorID",
    8: "AdmissionDate",
    9: "SurgeryDate",
    10: "DischargeDate",
    11: "PostopAKI",
}


def read_authoritative_sheet1_header(workbook: str | Path) -> list[str]:
    """Read the ordered Sheet1 header without loading patient-level cells."""
    workbook = Path(workbook)
    with ZipFile(workbook) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.text or "" for node in item.iter(MAIN_NS + "t"))
            for item in shared_root.findall(MAIN_NS + "si")
        ]
        with archive.open("xl/worksheets/sheet1.xml") as worksheet:
            for _, row in ET.iterparse(worksheet, events=("end",)):
                if row.tag != MAIN_NS + "row":
                    continue
                if int(row.attrib["r"]) != 1:
                    raise ValueError("Sheet1 header row was not the first worksheet row.")
                values: dict[int, str] = {}
                for cell in row.findall(MAIN_NS + "c"):
                    column = excel_column_number(cell.attrib["r"])
                    value_node = cell.find(MAIN_NS + "v")
                    value = "" if value_node is None else (value_node.text or "")
                    if cell.attrib.get("t") == "s" and value:
                        value = shared_strings[int(value)]
                    values[column] = value.strip()
                if not values:
                    raise ValueError("Sheet1 header row is empty.")
                return [values.get(column, "") for column in range(1, max(values) + 1)]
    raise ValueError("Sheet1 header row was not found.")


def excel_column_number(cell_reference: str) -> int:
    """Convert an Excel cell reference such as ``DF12`` to a one-based column."""
    match = re.match(r"[A-Z]+", str(cell_reference).upper())
    if match is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference!r}")
    value = 0
    for character in match.group():
        value = value * 26 + ord(character) - 64
    return value


def parse_source_date(value) -> pd.Timestamp:
    """Parse Excel serials and mixed source-system date encodings.

    Exact dates remain in memory only. The seven-digit value found in the
    source follows YYYYMMd formatting (for example, 2022027 is 7 Feb 2022).
    """
    if value is None or str(value).strip() == "":
        return pd.NaT
    text = str(value).strip()
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        numeric = np.nan
    if np.isfinite(numeric) and 20_000 <= numeric <= 60_000:
        return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=float(numeric)))
    if re.fullmatch(r"20\d{6}", text):
        return pd.Timestamp(datetime.strptime(text, "%Y%m%d"))
    if re.fullmatch(r"20\d{5}", text):
        year = int(text[:4])
        remainder = text[4:]
        candidates = (
            (int(remainder[:2]), int(remainder[2:])),
            (int(remainder[:1]), int(remainder[1:])),
        )
        for month, day in candidates:
            try:
                return pd.Timestamp(datetime(year, month, day))
            except ValueError:
                continue
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_convert(None) if parsed.tzinfo is not None else pd.Timestamp(parsed)


def read_authoritative_sheet1_dates(workbook: str | Path) -> pd.DataFrame:
    """Read only centre, stable ID, dates, and outcome from Sheet1 of an xlsx file."""
    workbook = Path(workbook)
    with ZipFile(workbook) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.text or "" for node in item.iter(MAIN_NS + "t"))
            for item in shared_root.findall(MAIN_NS + "si")
        ]
        rows: list[dict] = []
        with archive.open("xl/worksheets/sheet1.xml") as worksheet:
            for _, row in ET.iterparse(worksheet, events=("end",)):
                if row.tag != MAIN_NS + "row":
                    continue
                row_number = int(row.attrib["r"])
                if row_number == 1:
                    row.clear()
                    continue
                values: dict[int, object] = {}
                for cell in row.findall(MAIN_NS + "c"):
                    column = excel_column_number(cell.attrib["r"])
                    if column not in REQUIRED_COLUMN_NUMBERS:
                        continue
                    value_node = cell.find(MAIN_NS + "v")
                    value = None if value_node is None else value_node.text
                    if cell.attrib.get("t") == "s" and value is not None:
                        value = shared_strings[int(value)]
                    values[column] = value
                rows.append(
                    {
                        REQUIRED_COLUMN_NUMBERS[column]: values.get(column)
                        for column in REQUIRED_COLUMN_NUMBERS
                    }
                )
                row.clear()

    frame = pd.DataFrame(rows)
    for variable in ("Center", "MajorID", "PostopAKI"):
        frame[variable] = pd.to_numeric(frame[variable], errors="coerce")
    for variable in ("AdmissionDate", "SurgeryDate", "DischargeDate"):
        frame[variable] = frame[variable].map(parse_source_date)
        frame[variable] = pd.to_datetime(frame[variable], errors="coerce")
    return frame


def within_centre_chronological_split(
    frame: pd.DataFrame,
    training_fraction: float = 0.70,
) -> tuple[pd.Series, pd.DataFrame]:
    """Assign the earliest observations within each centre to development.

    The cutoff uses dates only, never outcomes. All operations on a cutoff date
    remain in development, avoiding arbitrary separation of same-day records.
    """
    if not 0.5 <= training_fraction < 1:
        raise ValueError("training_fraction must be in [0.5, 1).")
    if frame["SurgeryDate"].isna().any():
        raise ValueError("A complete SurgeryDate is required for chronological splitting.")
    split = pd.Series(index=frame.index, dtype="string")
    rows: list[dict] = []
    for centre, group in frame.groupby("Center", sort=True):
        ordered = group.sort_values(["SurgeryDate", "MajorID"])
        target_index = max(
            0,
            min(len(ordered) - 2, int(np.floor(len(ordered) * training_fraction)) - 1),
        )
        cutoff = ordered.iloc[target_index]["SurgeryDate"]
        train_index = ordered.index[ordered["SurgeryDate"] <= cutoff]
        test_index = ordered.index[ordered["SurgeryDate"] > cutoff]
        if len(train_index) == 0 or len(test_index) == 0:
            raise ValueError(f"Centre {centre} does not support a chronological holdout.")
        split.loc[train_index] = "development"
        split.loc[test_index] = "validation"
        rows.append(
            {
                "split_definition": "within_centre_70_30",
                "center": int(centre),
                "cutoff_date": pd.Timestamp(cutoff),
                "development_n": int(len(train_index)),
                "validation_n": int(len(test_index)),
            }
        )
    if split.isna().any():
        raise AssertionError("Every patient must receive exactly one temporal split label.")
    return split, pd.DataFrame(rows)


def fixed_calendar_split(frame: pd.DataFrame, cutoff: str = "2022-01-01") -> pd.Series:
    """Create a secondary fixed-calendar split without reading the outcome."""
    if frame["SurgeryDate"].isna().any():
        raise ValueError("A complete SurgeryDate is required for chronological splitting.")
    cutoff_date = pd.Timestamp(cutoff)
    return pd.Series(
        np.where(frame["SurgeryDate"] < cutoff_date, "development", "validation"),
        index=frame.index,
        dtype="string",
    )


def wilson_interval(
    events: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    proportion = events / n
    denominator = 1 + z**2 / n
    centre = (proportion + z**2 / (2 * n)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2))
        / denominator
    )
    return float(centre - half_width), float(centre + half_width)
