# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Nature-style figures for locked cross-database validation

# %%
from release_paths import release_path as _release_path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(str(_release_path('analysis')))
STAGE = ROOT / 'cross_database_external_validation'
TABLES = STAGE / 'tables'
FIGURES = STAGE / 'figures'

BLUE = '#0072B2'
ORANGE = '#E69F00'
GREEN = '#009E73'
VERMILION = '#D55E00'
GREY = '#6B7280'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'legend.fontsize': 7, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'figure.dpi': 150, 'savefig.dpi': 600,
})


def save(fig, folder: Path, stem: str, *, tight: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    save_kwargs = {'bbox_inches': 'tight'} if tight else {}
    fig.savefig(folder / f'{stem}.pdf', **save_kwargs)
    fig.savefig(folder / f'{stem}.svg', **save_kwargs)
    fig.savefig(
        folder / f'{stem}.tiff', dpi=600,
        pil_kwargs={'compression': 'tiff_lzw'}, **save_kwargs
    )
    plt.close(fig)


def forest(ax, data, metric: str, reference: float, xlabel: str, labels: list[str], colors: list[str]) -> None:
    y = np.arange(len(data))[::-1]
    values = data[metric].to_numpy(float)
    lower = data[f'{metric}_ci_lower'].to_numpy(float)
    upper = data[f'{metric}_ci_upper'].to_numpy(float)
    for yi, value, lo, hi, color in zip(y, values, lower, upper, colors):
        ax.errorbar(value, yi, xerr=[[value - lo], [hi - value]], fmt='o', color=color,
                    ecolor=color, elinewidth=1.1, capsize=2.2, markersize=4.2, zorder=3)
    ax.axvline(reference, color='#9CA3AF', linestyle='--', linewidth=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color='#E5E7EB', linewidth=0.6)


def calibration(ax, data, color_map, style_map=None) -> None:
    ax.plot([0, 1], [0, 1], color='#9CA3AF', linestyle='--', linewidth=0.9)
    for keys, group in data.groupby([c for c in ['validation_database', 'model_specification'] if c in data.columns], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        database = keys[0]
        specification = keys[1] if len(keys) > 1 else None
        label = database if specification is None else f'{database}, {specification.replace("_", " ")}'
        linestyle = '-' if style_map is None else style_map.get(specification, '-')
        ax.plot(group['mean_predicted_probability'], group['observed_event_fraction'],
                marker='o', markersize=2.8, linewidth=1.2, color=color_map[database],
                linestyle=linestyle, label=label)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed event fraction')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(color='#E5E7EB', linewidth=0.6)
    ax.legend(loc='best')


def make_public_icu_figure() -> None:
    folder = FIGURES / 'SupplementaryFigure7'
    metrics = pd.read_csv(TABLES / 'Table_inspire_locked_external_validation.csv')
    metrics = metrics.loc[metrics['validation_database'].ne('INSPIRE')].reset_index(drop=True)
    curves = pd.read_csv(TABLES / 'Table_inspire_locked_external_calibration_curve.csv')
    labels = [
        f"{row['validation_database'].replace('MIMIC-IV', 'MIMIC')} | "
        f"{'minimal' if row['model_specification'] == 'minimal' else 'extended'}"
        for _, row in metrics.iterrows()
    ]
    colors = [BLUE if name == 'MIMIC-IV' else ORANGE for name in metrics['validation_database']]
    color_map = {'MIMIC-IV': BLUE, 'eICU': ORANGE}
    style_map = {'minimal': '--', 'extended_common': '-'}

    panel_specs = [
        ('SupplementaryFigure7a', 'roc_auc', 0.5, 'Area under the ROC curve'),
        ('SupplementaryFigure7b', 'oe_ratio', 1.0, 'Observed to expected ratio'),
        ('SupplementaryFigure7c', 'calibration_slope', 1.0, 'Calibration slope'),
    ]
    for stem, metric, reference, xlabel in panel_specs:
        fig, ax = plt.subplots(figsize=(3.55, 2.55))
        forest(ax, metrics, metric, reference, xlabel, labels, colors)
        save(fig, folder, stem)
    fig, ax = plt.subplots(figsize=(3.55, 2.75))
    calibration(ax, curves, color_map, style_map)
    save(fig, folder, 'SupplementaryFigure7d')

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 5.35))
    for label, ax, (_, metric, reference, xlabel) in zip(['a', 'b', 'c'], axes.flat[:3], panel_specs):
        forest(ax, metrics, metric, reference, xlabel, labels, colors)
        ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontweight='bold', fontsize=10)
    calibration(axes.flat[3], curves, color_map, style_map)
    axes.flat[3].text(-0.18, 1.06, 'd', transform=axes.flat[3].transAxes, fontweight='bold', fontsize=10)
    fig.subplots_adjust(wspace=0.68, hspace=0.42)
    save(fig, folder, 'SupplementaryFigure7', tight=False)
    metrics.to_csv(folder / 'SupplementaryFigure7a-c_source_data.csv', index=False)
    curves.to_csv(folder / 'SupplementaryFigure7d_source_data.csv', index=False)


def make_clinical_bridge_figure() -> None:
    folder = FIGURES / 'SupplementaryFigure8'
    metrics = pd.read_csv(TABLES / 'Table_public_model_to_source_clinical_bridge.csv')
    metrics = metrics.loc[metrics['validation_database'].ne('INSPIRE')].reset_index(drop=True)
    curves = pd.read_csv(TABLES / 'Table_public_model_to_source_calibration_curve.csv')
    labels = metrics['validation_database'].replace({'Five-centre-source': 'Five-centre'}).tolist()
    colors = [BLUE, VERMILION]
    color_map = {'MIMIC-IV': BLUE, 'Five-centre-source': VERMILION}

    panel_specs = [
        ('SupplementaryFigure8a', 'roc_auc', 0.5, 'Area under the ROC curve'),
        ('SupplementaryFigure8b', 'oe_ratio', 1.0, 'Observed to expected ratio'),
        ('SupplementaryFigure8c', 'calibration_slope', 1.0, 'Calibration slope'),
    ]
    for stem, metric, reference, xlabel in panel_specs:
        fig, ax = plt.subplots(figsize=(3.55, 2.35))
        forest(ax, metrics, metric, reference, xlabel, labels, colors)
        save(fig, folder, stem)
    fig, ax = plt.subplots(figsize=(3.55, 2.75))
    calibration(ax, curves, color_map)
    save(fig, folder, 'SupplementaryFigure8d')

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 5.15))
    for label, ax, (_, metric, reference, xlabel) in zip(['a', 'b', 'c'], axes.flat[:3], panel_specs):
        forest(ax, metrics, metric, reference, xlabel, labels, colors)
        ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontweight='bold', fontsize=10)
    calibration(axes.flat[3], curves, color_map)
    axes.flat[3].text(-0.18, 1.06, 'd', transform=axes.flat[3].transAxes, fontweight='bold', fontsize=10)
    fig.subplots_adjust(wspace=0.62, hspace=0.42)
    save(fig, folder, 'SupplementaryFigure8', tight=False)
    metrics.to_csv(folder / 'SupplementaryFigure8a-c_source_data.csv', index=False)
    curves.to_csv(folder / 'SupplementaryFigure8d_source_data.csv', index=False)


if __name__ == '__main__':
    make_public_icu_figure()
    make_clinical_bridge_figure()
    print('Created Supplementary Figures 7 and 8 with panel-level source data')
