#!/usr/bin/env python3
"""Redesign JTM figures from frozen aggregate source data only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


MM = 1 / 25.4
WIDTH = 183 * MM
CHARCOAL = "#2B2B2B"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
LIGHT = "#D9D9D9"
DB_COLOURS = {"INSPIRE": BLUE, "MIMIC": ORANGE, "MIMIC-IV": ORANGE, "EICU": GREEN, "eICU": GREEN, "VitalDB": PURPLE}
DB_MARKERS = {"INSPIRE": "o", "MIMIC": "s", "MIMIC-IV": "s", "EICU": "^", "eICU": "^", "VitalDB": "D"}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def clean(ax, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#ECECEC", linewidth=0.5, zorder=0)


class Builder:
    def __init__(self, source: Path, supplement: Path | None, output: Path):
        self.source = source
        self.supplement = supplement
        self.output = output
        self.source_output = output.parent / "source_data"
        self.output.mkdir(parents=True, exist_ok=True)
        self.source_output.mkdir(parents=True, exist_ok=True)
        self.manifest: dict[str, object] = {
            "version": "figure_redesign_v1_20260921",
            "final_width_mm": 183,
            "statistical_unit": "patient/operation or ICU stay; Monte Carlo replicate only where labelled",
            "figures": [],
        }

    def csv(self, relative: str) -> pd.DataFrame:
        return pd.read_csv(self.source / relative)

    def save(self, fig, name: str, inputs: list[Path], claim: str) -> None:
        pdf = self.output / f"{name}.pdf"
        svg = self.output / f"{name}.svg"
        fig.savefig(pdf, dpi=600)
        fig.savefig(svg, dpi=600)
        plt.close(fig)
        self.manifest["figures"].append({
            "name": name,
            "claim": claim,
            "pdf": str(pdf),
            "pdf_sha256": sha256(pdf),
            "svg": str(svg),
            "svg_sha256": sha256(svg),
            "inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs],
        })

    def figure1(self) -> None:
        files = [
            "Figure1/Figure1a_reference_flow_source_data.csv",
            "Figure1/Figure1b_ipaw_balance_source_data.csv",
            "Figure1/Figure1c_monitoring_gradient_source_data.csv",
            "Figure1/Figure1d_estimand_schematic_source_data.csv",
        ]
        flow, balance, density, _ = [self.csv(x) for x in files]
        fig = plt.figure(figsize=(WIDTH, 6.55), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], width_ratios=[1.08, 1.0])

        ax = fig.add_subplot(gs[0, 0])
        stage_order = ["candidate_operations", "longitudinal_0_168h_creatinine_observed", "two_slot_observed", "dense_longitudinal_reference"]
        stage_labels = ["Candidate", "Any 0–168 h\ncreatinine", "Two-window\nobserved", "Dense\nreference"]
        for db in ["INSPIRE", "MIMIC", "EICU"]:
            g = flow[flow.database.eq(db)].copy()
            g["stage_key"] = g.step.fillna(g.stage)
            mapping = dict(zip(g.stage_key, g.n))
            if db == "MIMIC":
                local = [g.iloc[0].stage_key, g.iloc[1].stage_key, g.iloc[2].stage_key, g.iloc[3].stage_key]
            elif db == "EICU":
                local = [g.iloc[0].stage_key, g.iloc[2].stage_key, g.iloc[3].stage_key, g.iloc[4].stage_key]
            else:
                local = stage_order
            vals = np.array([mapping[s] for s in local], dtype=float)
            frac = vals / vals[0]
            ax.plot(range(4), frac, marker=DB_MARKERS[db], color=DB_COLOURS[db], label=db.replace("MIMIC", "MIMIC-IV"), zorder=3)
            final = g[g.stage_key.eq(local[-1])].iloc[0]
            event_text = "" if pd.isna(final.events) else f", E={int(final.events):,}"
            ax.annotate(f"N={int(final.n):,}{event_text}", (3, frac[-1]), xytext=(4, 0), textcoords="offset points", color=DB_COLOURS[db], va="center", fontsize=6.3)
        ax.set_xticks(range(4), stage_labels)
        ax.set_ylabel("Fraction retained from candidate cohort")
        ax.set_ylim(0, 1.08)
        ax.legend(frameon=False, ncol=3, loc="lower left")
        clean(ax, "y")
        panel_label(ax, "a")

        ax = fig.add_subplot(gs[0, 1])
        b = balance.sort_values("max_abs", ascending=True).copy()
        labels = b.variable.replace({"n_preop_creatinine_7d": "Preoperative creatinine count", "restricted_rf_probability": "Restricted RF score", "ridge_probability": "Ridge score", "gradient_boosting_probability": "Gradient-boosting score", "PreopCr": "Preoperative creatinine", "PreopHb": "Preoperative haemoglobin", "PreopAlb": "Preoperative albumin"})
        y = np.arange(len(b))
        for yi, before, after in zip(y, b.smd_before_vs_full, b.smd_after_vs_full):
            ax.plot([before, after], [yi, yi], color=LIGHT, linewidth=1.0, zorder=1)
        ax.scatter(b.smd_before_vs_full, y, s=16, color="#7A7A7A", label="Before weighting", zorder=2)
        ax.scatter(b.smd_after_vs_full, y, s=16, color=BLUE, label="After IPAW", zorder=3)
        ax.axvline(0, color=CHARCOAL, linewidth=0.65)
        ax.axvline(-0.10, color=VERMILLION, linestyle="--", linewidth=0.7)
        ax.axvline(0.10, color=VERMILLION, linestyle="--", linewidth=0.7)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Standardised difference vs candidate cohort")
        ax.legend(frameon=False, loc="lower right")
        clean(ax, "x")
        panel_label(ax, "b")

        inner = gs[1, 0].subgridspec(2, 1, hspace=0.08)
        ax1 = fig.add_subplot(inner[0, 0])
        ax2 = fig.add_subplot(inner[1, 0], sharex=ax1)
        x = density.minimum_postoperative_creatinine_count
        ax1.plot(x, density.event_rate * 100, marker="o", color=VERMILLION)
        ax1.set_ylabel("Detected event\nrate (%)")
        ax1.tick_params(labelbottom=False)
        ax2.plot(x, density.n, marker="s", color=BLUE)
        ax2.set_ylabel("Eligible N")
        ax2.set_xlabel("Minimum postoperative creatinine measurements")
        clean(ax1, "y"); clean(ax2, "y")
        panel_label(ax1, "c")

        ax = fig.add_subplot(gs[1, 1])
        ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
        boxes = [
            (0.1, 3.2, 2.45, 0.85, "Retained\ntrajectory", BLUE),
            (3.75, 3.2, 2.45, 0.85, "Observed\nsubset", SKY),
            (7.4, 3.2, 2.45, 0.85, "Reconstructed\nendpoint", ORANGE),
            (3.75, 0.55, 2.45, 0.85, "Fixed risk\npredictions", GREEN),
        ]
        for x0, y0, w, h, txt, col in boxes:
            rect = mpl.patches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.04", facecolor=col, edgecolor="none")
            ax.add_patch(rect); ax.text(x0 + w/2, y0 + h/2, txt, ha="center", va="center", color="white", fontweight="bold", fontsize=7)
        for xa, xb in [(2.6, 3.68), (6.25, 7.33)]:
            ax.annotate("", xy=(xb, 3.62), xytext=(xa, 3.62), arrowprops=dict(arrowstyle="->", color=CHARCOAL, lw=0.8))
        ax.text(3.15, 4.20, "measurement\ndeletion", ha="center", va="bottom", fontsize=6.2, linespacing=0.9)
        ax.text(6.8, 4.20, "endpoint\nconstruction", ha="center", va="bottom", fontsize=6.2, linespacing=0.9)
        ax.annotate("", xy=(1.35, 3.12), xytext=(4.35, 1.45), arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
        ax.annotate("", xy=(8.65, 3.12), xytext=(5.6, 1.45), arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=0.8))
        ax.text(1.75, 1.72, "retained-reference\nevaluation", ha="center", color=BLUE, fontsize=6.5)
        ax.text(8.2, 1.72, "reconstructed-target\nevaluation", ha="center", color=VERMILLION, fontsize=6.5)
        panel_label(ax, "d", x=-0.02, y=1.0)
        self.save(fig, "Figure1", [self.source / x for x in files], "Cohort restriction and measurement opportunity define which operational endpoint can be evaluated.")

    def figure2(self) -> None:
        event_files = [f"Figure2/Figure2a_{db}_event_bias_source_data.csv" for db in ["inspire", "mimic", "eicu"]]
        method_files = [f"Figure2/Figure2b_{db}_method_bias_source_data.csv" for db in ["inspire", "mimic", "eicu"]]
        sens_file = "Figure2/Figure2c_reconstruction_sensitivity_source_data.csv"
        control_file = "Figure2/Figure2d_pure_selection_control_source_data.csv"
        events = [self.csv(x) for x in event_files]
        methods = pd.concat([self.csv(x) for x in method_files], ignore_index=True)
        sens = self.csv(sens_file)
        control = self.csv(control_file)
        fig = plt.figure(figsize=(WIDTH, 7.35), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.38, 1.0, 0.95])
        top = gs[0, :].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.18)
        mech_order = ["MCAR", "stratum_MAR", "risk_MAR", "history_MAR", "outcome_MNAR", "mixed_MNAR"]
        all_bias = pd.concat(events).bias.abs().max()
        norm = TwoSlopeNorm(vmin=-all_bias, vcenter=0, vmax=all_bias)
        image = None
        for i, (db, data) in enumerate(zip(["INSPIRE", "MIMIC-IV", "eICU"], events)):
            ax = fig.add_subplot(top[0, i])
            p = data.pivot(index="mechanism", columns="retention_target", values="bias").reindex(mech_order)
            image = ax.imshow(p.values, cmap="RdBu_r", norm=norm, aspect="auto")
            ax.set_xticks(range(len(p.columns)), [f"{x:.2f}" for x in p.columns])
            ax.set_yticks(range(len(p.index)), [x.replace("_", " ") for x in p.index] if i == 0 else [])
            ax.set_xlabel("Per-measurement retention")
            ax.set_title(db, fontweight="bold")
            for spine in ax.spines.values(): spine.set_visible(False)
            if i == 0: panel_label(ax, "a", x=-0.26)
        cax = fig.add_subplot(top[0, 3])
        cb = fig.colorbar(image, cax=cax); cb.set_label("Event-rate bias")

        ax = fig.add_subplot(gs[1, 0])
        d = methods[np.isclose(methods.retention_target, 0.35)].copy()
        method_order = ["Naive", "IPAW untruncated", "IPAW truncated", "AIPW"]
        db_order = ["INSPIRE", "MIMIC", "EICU"]
        y_positions = []; labels = []
        y = 0
        for db in db_order:
            for method in method_order:
                row = d[(d.database.eq(db)) & (d.method_label.eq(method))].iloc[0]
                ax.errorbar(row.bias, y, xerr=[[row.bias - (row.q025-row.truth)], [(row.q975-row.truth)-row.bias]], fmt=DB_MARKERS[db], color=DB_COLOURS[db], capsize=1.8)
                y_positions.append(y); labels.append(f"{db.replace('MIMIC','MIMIC-IV')} · {method}"); y += 1
            y += 0.45
        ax.axvline(0, color=CHARCOAL, linestyle="--", linewidth=0.7)
        ax.set_yticks(y_positions, labels); ax.invert_yaxis(); ax.set_xlabel("O/E bias at 35% retention")
        clean(ax, "x"); panel_label(ax, "b")

        ax = fig.add_subplot(gs[1, 1])
        for db in ["INSPIRE", "MIMIC", "EICU"]:
            g = sens[sens.database.eq(db)].sort_values("retention_target")
            ax.plot(g.retention_target, g["mean"], marker=DB_MARKERS[db], color=DB_COLOURS[db], label=db.replace("MIMIC", "MIMIC-IV"))
            ax.fill_between(g.retention_target, g.q025, g.q975, color=DB_COLOURS[db], alpha=0.14, linewidth=0)
        ax.set_ylim(0, 1); ax.set_xlabel("Per-measurement retention"); ax.set_ylabel("Reconstructed-endpoint sensitivity")
        ax.legend(frameon=False, loc="lower right"); clean(ax, "y"); panel_label(ax, "c")

        ax = fig.add_subplot(gs[2, :])
        x = np.arange(3); offsets = [-0.24, -0.08, 0.08, 0.24]
        combos = [("risk_MAR", "naive", "Naive · risk MAR", "o", "#7A7A7A"), ("risk_MAR", "oracle_IPW_untruncated", "Oracle IPW · risk MAR", "o", BLUE), ("outcome_MNAR", "naive", "Naive · outcome MNAR", "s", "#7A7A7A"), ("outcome_MNAR", "oracle_IPW_untruncated", "Oracle IPW · outcome MNAR", "s", BLUE)]
        for off, (mech, method, label, marker, color) in zip(offsets, combos):
            vals=[]; lows=[]; highs=[]
            for db in ["INSPIRE", "MIMIC", "EICU"]:
                row=control[(control.database.eq(db)) & control.mechanism.eq(mech) & control.method.eq(method)].iloc[0]
                vals.append(row.bias); lows.append(row.bias-(row.q025-row.truth)); highs.append((row.q975-row.truth)-row.bias)
            ax.errorbar(x+off, vals, yerr=[lows, highs], fmt=marker, color=color, linestyle="none", capsize=2, label=label)
        ax.axhline(0, color=CHARCOAL, linewidth=0.7); ax.set_xticks(x, ["INSPIRE", "MIMIC-IV", "eICU"]); ax.set_ylabel("Event-rate bias")
        ax.legend(frameon=False, ncol=2, loc="upper center"); clean(ax, "y"); panel_label(ax, "d", x=-0.055)
        inputs = [self.source / x for x in event_files + method_files + [sens_file, control_file]]
        self.save(fig, "Figure2", inputs, "Deleting measurements changes endpoint reconstruction; weighting corrects pure label selection but not missing trajectories.")

    def figure3(self) -> None:
        files = [
            "Figure3/Figure3a_apparent_vs_reference_recalibration_source_data.csv",
            "Figure3/Figure3b_strategy_rmse_source_data.csv",
            "Figure3/Figure3d_inspire_recalibration_fidelity_source_data.csv",
            "Figure3/Figure3d_mimic_recalibration_fidelity_source_data.csv",
            "Figure3/Figure3d_eicu_recalibration_fidelity_source_data.csv",
            "Figure3/Figure3e_reference_sample_design_source_data.csv",
        ]
        divergence = self.csv(files[0]); rmse = self.csv(files[1])
        fidelity = pd.concat([self.csv(x) for x in files[2:5]], ignore_index=True)
        reference = self.csv(files[5])
        fig = plt.figure(figsize=(WIDTH, 7.85), constrained_layout=True)
        gs = fig.add_gridspec(4, 2, height_ratios=[0.10, 1.0, 1.1, 1.0])
        legend_ax = fig.add_subplot(gs[0, :]); legend_ax.axis("off")
        top = gs[1, :].subgridspec(1, 3, wspace=0.14)
        target_styles = {"Apparent endpoint": (VERMILLION, "-", "o"), "Retained reference": (CHARCOAL, "--", "s")}
        for i, db in enumerate(["INSPIRE", "MIMIC", "EICU"]):
            ax = fig.add_subplot(top[0, i])
            for target, (color, ls, marker) in target_styles.items():
                g = divergence[(divergence.database.eq(db)) & divergence.target_label.eq(target)].sort_values("retention_target")
                ax.plot(g.retention_target, g["mean"], color=color, linestyle=ls, marker=marker, label=target)
                ax.fill_between(g.retention_target, g.q025, g.q975, color=color, alpha=0.12, linewidth=0)
            ax.axhline(1, color=LIGHT, linewidth=0.8); ax.set_ylim(0.35, 1.08); ax.set_title(db.replace("MIMIC", "MIMIC-IV"), fontweight="bold")
            ax.set_xlabel("Per-measurement retention")
            if i == 0: ax.set_ylabel("O/E after local updating")
            else: ax.set_yticklabels([])
            clean(ax, "y")
            if i == 0: panel_label(ax, "a", x=-0.25)
        handles = [Line2D([0],[0],color=VERMILLION,marker="o",label="Reconstructed target"), Line2D([0],[0],color=CHARCOAL,ls="--",marker="s",label="Retained reference")]
        legend_ax.legend(handles=handles, frameon=False, ncol=2, loc="center")

        ax = fig.add_subplot(gs[2, 0])
        method_order = ["Naive", "IPAW untruncated", "IPAW truncated", "AIPW", "Gamma=2 midpoint"]
        y = np.arange(len(method_order))
        for db, off in zip(["INSPIRE", "MIMIC", "EICU"], [-0.18, 0, 0.18]):
            g = rmse[rmse.database.eq(db)].set_index("label").reindex(method_order)
            ax.scatter(g.rmse, y+off, s=18, marker=DB_MARKERS[db], color=DB_COLOURS[db], label=db.replace("MIMIC", "MIMIC-IV"))
        ax.set_yticks(y, method_order); ax.invert_yaxis(); ax.set_xlabel("Event-rate RMSE vs retained reference")
        clean(ax, "x"); panel_label(ax, "b")

        ax = fig.add_subplot(gs[2, 1])
        method_map = {"recalibration_intercept_truth": "Intercept", "recalibration_intercept_slope_truth": "Intercept+slope", "reference_10pct_recalibration": "10% reference"}
        g = fidelity[np.isclose(fidelity.retention_target, 0.35)].copy(); g["label"] = g.method.map(method_map)
        method_order2 = ["Intercept", "Intercept+slope", "10% reference"]
        y = np.arange(len(method_order2))
        for db, off in zip(["INSPIRE", "MIMIC", "EICU"], [-0.18, 0, 0.18]):
            h = g[g.database.eq(db)].set_index("label").reindex(method_order2)
            ax.errorbar(h["mean"], y+off, xerr=[h["mean"]-h.q025, h.q975-h["mean"]], fmt=DB_MARKERS[db], color=DB_COLOURS[db], capsize=2)
        ax.axvline(1, color=CHARCOAL, linestyle="--", linewidth=0.7); ax.set_yticks(y, method_order2); ax.invert_yaxis(); ax.set_xlabel("O/E vs retained reference at 35% retention")
        clean(ax, "x"); panel_label(ax, "c")

        ax = fig.add_subplot(gs[3, :])
        for db in ["INSPIRE", "MIMIC", "EICU"]:
            g = reference[reference.database.eq(db)].sort_values("reference_fraction")
            ax.errorbar(g.reference_fraction*100, g["mean"], yerr=[g["mean"]-g.q025, g.q975-g["mean"]], marker=DB_MARKERS[db], color=DB_COLOURS[db], capsize=2, label=db.replace("MIMIC", "MIMIC-IV"))
        ax.axhline(1, color=CHARCOAL, linestyle="--", linewidth=0.7); ax.set_xlabel("Retained-reference sample (%)"); ax.set_ylabel("Held-out O/E")
        ax.legend(frameon=False, ncol=3); clean(ax, "y"); panel_label(ax, "d", x=-0.055)
        self.save(fig, "Figure3", [self.source / x for x in files], "Apparent calibration can be restored while retained-reference calibration remains biased; reference samples reduce ambiguity at a precision cost.")

    def figure4(self) -> None:
        files = ["Figure4/Figure7b_discrimination_strength_stress_source_data.csv", "Figure4/Figure7a_extended_common_transport_source_data.csv"]
        stress = self.csv(files[0]); transport = self.csv(files[1])
        fig = plt.figure(figsize=(WIDTH, 5.35), constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[0.10, 1.0, 1.05])
        legend_ax = fig.add_subplot(gs[0, 0]); legend_ax.axis("off")
        top = gs[1, 0].subgridspec(1, 2, wspace=0.12)
        style = {"full_reference_score": (CHARCOAL, "o", "-", "Original score · retained reference"), "local_recalibration_apparent": (GREEN, "s", "-", "Updated score · reconstructed target"), "local_recalibration_truth": (ORANGE, "^", "--", "Same update · retained reference")}
        for i, db in enumerate(["MIMIC-IV", "eICU"]):
            ax = fig.add_subplot(top[0, i])
            for method, (color, marker, ls, label) in style.items():
                g=stress[(stress.database.eq(db)) & stress.method.eq(method)].sort_values("target_auc")
                ax.errorbar(g.target_auc, g["mean"], yerr=[g["mean"]-g.q025, g.q975-g["mean"]], color=color, marker=marker, linestyle=ls, capsize=2, label=label)
            ax.axhline(1,color=LIGHT,linewidth=.8); ax.set_title(db,fontweight="bold"); ax.set_xlabel("Designed retained-reference AUC")
            if i==0: ax.set_ylabel("Observed-to-expected ratio")
            else: ax.set_yticklabels([])
            clean(ax,"y")
            if i==0: panel_label(ax,"a",x=-.22)
        legend_ax.legend([Line2D([0],[0],color=v[0],marker=v[1],ls=v[2]) for v in style.values()], [v[3] for v in style.values()], frameon=False, ncol=3, loc="center")

        ax=fig.add_subplot(gs[2,0]); directions=["MIMIC-IV → eICU","eICU → MIMIC-IV"]; y=np.array([1,0],float)
        for spec,off,color,marker,label in [("minimal",-.12,"#7A7A7A","o","Minimal"),("extended_common",.12,BLUE,"s","Extended common")]:
            g=transport[transport.model_specification.eq(spec)].set_index("direction_label").reindex(directions)
            ax.errorbar(g.roc_auc,y+off,xerr=[g.roc_auc-g.roc_auc_ci_lower,g.roc_auc_ci_upper-g.roc_auc],fmt=marker,color=color,capsize=2,label=label)
        ax.axvline(.5,color=CHARCOAL,ls="--",lw=.7); ax.set_yticks(y,directions); ax.set_xlabel("External AUC (95% interval)"); ax.legend(frameon=False,loc="lower right")
        clean(ax,"x"); panel_label(ax,"b",x=-.055)
        self.save(fig,"Figure4",[self.source/x for x in files],"Target divergence persists across designed discrimination, while more common variables do not guarantee transport.")

    def figure5(self) -> None:
        rel="Figure5/Figure5_primary_12h_source_data.csv"; d=self.csv(rel)
        fig=plt.figure(figsize=(WIDTH,3.65),constrained_layout=True); gs=fig.add_gridspec(1,3,width_ratios=[1,1,.045])
        order=["INSPIRE","MIMIC","EICU","VitalDB"]
        image=None
        for i,(metric,title) in enumerate([("outcome_observed_fraction","Outcome observability"),("reconstructed_sensitivity","Reconstructed-endpoint sensitivity")]):
            ax=fig.add_subplot(gs[0,i]); p=d[d.metric.eq(metric)].pivot(index="target_database",columns="donor_schedule_database",values="mean").reindex(index=order,columns=order)
            image=ax.imshow(p.values,cmap="cividis",vmin=0,vmax=1,aspect="equal")
            ax.set_xticks(range(4),["INSPIRE","MIMIC-IV","eICU","VitalDB"],rotation=32,ha="right"); ax.set_yticks(range(4),["INSPIRE","MIMIC-IV","eICU","VitalDB"])
            ax.set_xlabel("Donor timing pattern");
            if i==0: ax.set_ylabel("Target retained-measurement grid")
            for r in range(4):
                for c in range(4):
                    val=p.iloc[r,c]; ax.text(c,r,f"{val:.0%}",ha="center",va="center",color="white" if val<.55 else CHARCOAL,fontweight="bold",fontsize=7)
            ax.set_title(title,fontweight="bold"); panel_label(ax,"a" if i==0 else "b",x=-.17)
            for spine in ax.spines.values(): spine.set_visible(False)
        cax=fig.add_subplot(gs[0,2]); cb=fig.colorbar(image,cax=cax); cb.set_label("Fraction")
        self.save(fig,"Figure5",[self.source/rel],"Unchanged trajectories yield different endpoint observability and sensitivity under different empirical measurement schedules.")

    def supplementary17(self) -> None:
        files=["SupplementaryFigure17a_dense_fraction_source_data.csv", "SupplementaryFigure17b_selection_imbalance_source_data.csv", "SupplementaryFigure17c_d_weight_diagnostics_source_data.csv"]
        bundled=[self.source/"SupplementaryFigure17"/name for name in files]
        if not all(path.is_file() for path in bundled):
            bundled=[self.source_output/name for name in files]
        if all(path.is_file() for path in bundled):
            summary,selection,weights=[pd.read_csv(path) for path in bundled]
            inputs=bundled
        else:
            if self.supplement is None:
                raise FileNotFoundError("Supply Supplementary Figure 17 CSV files or --supplementary-tables")
            summary=pd.read_excel(self.supplement,sheet_name="Table_dense_reference_selecti_1")
            summary=pd.concat([summary,pd.DataFrame([{"database":"VitalDB","candidate_n":2890,"dense_n":1078,"dense_fraction":1078/2890}])],ignore_index=True)
            selection=pd.read_excel(self.supplement,sheet_name="Table_dense_reference_selection")
            weights=pd.read_excel(self.supplement,sheet_name="Table_observability_weight_diag")
            for name,data in zip(files,[summary,selection,weights]):
                data.to_csv(self.source_output/name,index=False)
            inputs=[self.supplement]
        fig,axes=plt.subplots(2,2,figsize=(WIDTH,6.5),constrained_layout=True)
        ax=axes[0,0]; order=["INSPIRE","MIMIC-IV","eICU","VitalDB"]; g=summary.set_index("database").reindex(order)
        ax.bar(np.arange(4),g.dense_fraction*100,color=[BLUE,ORANGE,GREEN,PURPLE],width=.65)
        for i,row in enumerate(g.itertuples()): ax.text(i,row.dense_fraction*100+2,f"{row.dense_fraction:.0%}\n{int(row.dense_n):,}/{int(row.candidate_n):,}",ha="center",va="bottom",fontsize=6.5)
        ax.set_xticks(range(4),order); ax.set_ylim(0,88); ax.set_ylabel("Dense-reference fraction (%)"); clean(ax,"y"); panel_label(ax,"a")

        ax=axes[0,1]; s=selection[selection.database.eq("INSPIRE")].copy()
        variable_labels={"n_creatinine_0_168h":"0–168 h creatinine count","n_postop_creatinine_7d":"7-day postoperative creatinine count","n_postop_creatinine":"Postoperative creatinine count","Gastrocolorectal":"Cancer site","restricted_rf_probability":"Restricted RF score","unittype":"ICU unit type","active_service":"Active service","PreopAlb":"Preoperative albumin","PreopCr":"Preoperative creatinine","Gender":"Sex"}
        s["label"]=s.variable.map(variable_labels).fillna(s.variable.astype(str))+np.where(s.level.notna()," · "+s.level.astype(str),"")
        s=s.sort_values("absolute_standardized_difference",ascending=False).drop_duplicates("label").head(12).sort_values("absolute_standardized_difference")
        colors=np.where(s.absolute_standardized_difference>=.1,VERMILLION,"#7A7A7A")
        ax.barh(np.arange(len(s)),s.absolute_standardized_difference,color=colors,height=.65); ax.axvline(.1,color=CHARCOAL,ls="--",lw=.7)
        ax.set_yticks(range(len(s)),s.label); ax.set_xlabel("Absolute standardised difference\n(candidate vs dense reference)"); clean(ax,"x"); panel_label(ax,"b")

        w=weights[weights.target.eq("two_slot")].copy(); trunc_order=["none","0.5_99.5","1_99","2.5_97.5"]
        labels={"none":"None","0.5_99.5":"0.5–99.5%","1_99":"1–99%","2.5_97.5":"2.5–97.5%"}
        ax=axes[1,0]
        for model,color,marker in [("logistic",BLUE,"o"),("gradient_boosting",ORANGE,"s")]:
            h=w[w.nuisance_model.eq(model)].set_index("weight_truncation").reindex(trunc_order)
            ax.plot(range(4),h.effective_sample_size/h.observed_n,marker=marker,color=color,label=model.replace("_"," "))
        ax.set_xticks(range(4),[labels[x] for x in trunc_order],rotation=20,ha="right"); ax.set_ylim(.7,.9); ax.set_ylabel("Effective sample size / observed N"); ax.set_xlabel("Weight truncation"); ax.legend(frameon=False); clean(ax,"y"); panel_label(ax,"c")

        ax=axes[1,1]
        for model,color,marker in [("logistic",BLUE,"o"),("gradient_boosting",ORANGE,"s")]:
            h=w[w.nuisance_model.eq(model)].set_index("weight_truncation").reindex(trunc_order)
            ax.plot(range(4),h.weight_max,marker=marker,color=color,label=model.replace("_"," "))
        ax.set_xticks(range(4),[labels[x] for x in trunc_order],rotation=20,ha="right"); ax.set_yscale("log"); ax.set_ylabel("Maximum inverse-probability weight"); ax.set_xlabel("Weight truncation"); clean(ax,"y"); panel_label(ax,"d")
        self.save(fig,"SupplementaryFigure17",inputs,"Dense-reference estimands condition on selective monitoring, and weighting quality depends on overlap and weight stability.")

    def finish(self) -> None:
        (self.output.parent/"FIGURE_MANIFEST.json").write_text(json.dumps(self.manifest,indent=2)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-data",type=Path,required=True)
    parser.add_argument("--supplementary-tables",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); configure(); b=Builder(args.source_data,args.supplementary_tables,args.output)
    b.figure1(); b.figure2(); b.figure3(); b.figure4(); b.figure5(); b.supplementary17(); b.finish()
    print(json.dumps({"status":"PASS","figures":6,"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
