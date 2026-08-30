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
# # Assemble the 3,710-patient temporal audit as Supplementary Figure 5

# %%
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pymupdf
from release_paths import release_path

ROOT = release_path("analysis", "outputs/source3710_temporal")
INPUT = ROOT / "figures/Figure8_source_temporal_audit"
OUTPUT = ROOT / "figures/SupplementaryFigure5"
OUTPUT.mkdir(parents=True, exist_ok=True)

PAGE_WIDTH = 183 / 25.4 * 72
MARGIN = 10.0
GAP = 8.0
ROW_HEIGHT = 170.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rects() -> list[tuple[float, float, float, float]]:
    width = (PAGE_WIDTH - 2 * MARGIN - GAP) / 2
    result = []
    for row in range(2):
        y0 = MARGIN + row * (ROW_HEIGHT + GAP)
        result.append((MARGIN, y0, MARGIN + width, y0 + ROW_HEIGHT))
        result.append((MARGIN + width + GAP, y0, PAGE_WIDTH - MARGIN, y0 + ROW_HEIGHT))
    return result


names = [
    "Figure8a_recruitment_by_year.pdf",
    "Figure8b_within_centre_temporal_auc.pdf",
    "Figure8c_within_centre_temporal_oe.pdf",
    "Figure8d_inpatient_observation_opportunity.pdf",
]
doc = pymupdf.open()
page = doc.new_page(width=PAGE_WIDTH, height=370)
manifest = {"figure": "SupplementaryFigure5", "panels": []}
for label, name, rect in zip("abcd", names, rects()):
    source = INPUT / name
    panel = pymupdf.open(source)
    page.show_pdf_page(pymupdf.Rect(*rect), panel, 0, keep_proportion=True, overlay=True)
    page.insert_text(
        pymupdf.Point(rect[0] + 1.5, rect[1] + 8.0),
        label,
        fontsize=8,
        fontname="helv",
        color=(0.08, 0.08, 0.08),
        overlay=True,
    )
    panel.close()
    for companion in (
        source,
        source.with_suffix(".svg"),
        source.with_suffix(".tiff"),
        source.with_name(source.stem + "_source_data.csv"),
    ):
        if companion.exists():
            shutil.copy2(companion, OUTPUT / companion.name)
    manifest["panels"].append(
        {"label": label, "input": str(source), "sha256": sha256(source), "rect_pt": list(rect)}
    )

composite = OUTPUT / "SupplementaryFigure5.pdf"
doc.save(composite, garbage=4, deflate=True)
doc.close()
manifest["composite"] = str(composite)
manifest["composite_sha256"] = sha256(composite)
(OUTPUT / "SupplementaryFigure5_assembly_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
