#!/usr/bin/env python
"""Build manuscript-facing tables and figures from direct-RDoC outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = ROOT / "validation" / "outputs" / "direct_rdoc_gaussian_publication_20260805_matched_heldout_v4"
DEFAULT_CONTROLS = ROOT / "validation" / "outputs" / "direct_rdoc_negative_controls_publication_20260805_matched_heldout_v4"
DEFAULT_ADAPTIVE = ROOT / "validation" / "outputs" / "direct_rdoc_adaptive_publication_20260805_matched_heldout_v4"
DEFAULT_IRT = ROOT / "validation" / "outputs" / "direct_rdoc_irt_publication_20260805_matched_heldout_v4"
DEFAULT_CALIBRATION = ROOT / "simulations" / "paper" / "outputs" / "publication_calibration_20260805_v4"
DEFAULT_EMPIRICAL = ROOT / "empirical" / "derived_phi" / "empirical_fit_runs" / "publication_ready_20260805_v4"
DEFAULT_OUT = ROOT / "simulations" / "paper" / "outputs" / "publication_ready_20260805_v4"

METHOD_LABELS = {
    "ball_student_causal": "BALL causal student",
    "ball_teacher_smoother": "BALL teacher smoother",
    "ball_direct_causal": "Direct causal transformer",
    "ball_direct_transformer": "BALL direct transformer",
    "gp_causal_filter": "Causal Gaussian-process filter",
    "exponential_decay_gru": "Exponential-decay GRU",
    "s0_direct_lgssm": "LGSSM",
    "markov_direct_transition": "Markov pattern-mixture model",
    "ball_direct_causal_compute_matched": "Compute-matched direct causal transformer",
    "ball_inherited_dynamics_only": "Inherited dynamics and questionnaires",
    "ball_teacher_matching_only": "Teacher matching only",
    "ball_full_decomposition": "Full BALL decomposition arm",
    "ball_student_transition_only": "BALL with the observed RDoC profile restricted to the transition",
    "ball_student_transition_only_strict": "BALL with RDoC and its correlated proxies restricted to the transition",
}
PRIMARY_BALL_METHOD = "ball_student_causal"
TEACHER_METHOD = "ball_teacher_smoother"
LEGACY_BALL_METHOD = "ball_direct_transformer"
BALL_TRANSITION_LABEL = "BALL shared generative transition"
NOT_SEPARATELY_ESTIMATED = "Reported in shared BALL row"
NO_EXPLICIT_COEFFICIENT = "—"
NOT_APPLICABLE = "—"
METHOD_ORDER = [
    PRIMARY_BALL_METHOD,
    TEACHER_METHOD,
    "ball_direct_causal",
    "gp_causal_filter",
    "exponential_decay_gru",
    "s0_direct_lgssm",
    "markov_direct_transition",
]
COEFFICIENT_METHOD_ORDER = [TEACHER_METHOD, "ball_direct_causal", "s0_direct_lgssm", "markov_direct_transition"]
COEFFICIENT_LABELS = {
    TEACHER_METHOD: "BALL shared transition",
    "ball_direct_causal": "Direct causal transition",
    "s0_direct_lgssm": "LGSSM",
    "markov_direct_transition": "Markov transition",
}
COLORS = {
    "ball_student_causal": "#2f6db3",
    "ball_teacher_smoother": "#6aa6d8",
    "ball_direct_causal": "#1b9e77",
    "ball_direct_transformer": "#2f6db3",
    "gp_causal_filter": "#7b3294",
    "exponential_decay_gru": "#e6ab02",
    "s0_direct_lgssm": "#8f8f8f",
    "markov_direct_transition": "#d07c2c",
}
CELL_LABELS = {
    "linear": "Linear",
    "interaction": "Interaction",
    "nonlinear": "Nonlinear",
    "heterogeneous": "Heterogeneous",
    "missingness": "Missingness",
}
CONTROL_LABELS = {
    "positive": "Positive",
    "null": "Null signal",
    "permuted_proxy": "Permuted proxy",
    "noise_proxy": "Noise proxy",
}


def fmt(mean: float, se: float | None = None, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "n/a"
    if se is None or not np.isfinite(se):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({se:.{digits}f})"


def fmt_panel_range(mean: float, low: float, high: float, digits: int = 3) -> str:
    """Describe heterogeneous simulation conditions without treating them as draws."""
    if not all(np.isfinite(v) for v in (mean, low, high)):
        return "n/a"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def panel_summary(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = [m for m in METHOD_ORDER if m in set(agg["method"])]
    for method in methods:
        g = agg.loc[agg["method"] == method].copy()
        row = {
            "method": method,
            "label": METHOD_LABELS[method],
            "scenario_panels": int(len(g)),
        }
        for metric in ["beta_cosine", "beta_topk_f1", "latent_rmse", "val_anchor_rmse"]:
            vals = g[f"{metric}_mean"].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{metric}_se"] = (
                float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                if len(finite) > 1
                else (0.0 if len(finite) == 1 else np.nan)
            )
            row[f"{metric}_min"] = float(np.min(finite)) if len(finite) else np.nan
            row[f"{metric}_max"] = float(np.max(finite)) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def manuscript_summary_table(
    summary: pd.DataFrame,
    trajectory_metrics: list[tuple[str, str]],
) -> pd.DataFrame:
    """Separate the shared BALL coefficient from teacher and student trajectories."""

    rows: list[dict[str, object]] = []
    for item in summary.itertuples(index=False):
        method = str(item.method)
        row: dict[str, object] = {
            "method": str(item.label),
            "scenario_panels": int(item.scenario_panels),
            "beta_cosine": (
                NOT_SEPARATELY_ESTIMATED
                if method in {PRIMARY_BALL_METHOD, TEACHER_METHOD}
                else NO_EXPLICIT_COEFFICIENT
                if method in {"gp_causal_filter", "exponential_decay_gru"}
                else fmt_panel_range(item.beta_cosine_mean, item.beta_cosine_min, item.beta_cosine_max)
            ),
            "beta_topk_f1": (
                NOT_SEPARATELY_ESTIMATED
                if method in {PRIMARY_BALL_METHOD, TEACHER_METHOD}
                else NO_EXPLICIT_COEFFICIENT
                if method in {"gp_causal_filter", "exponential_decay_gru"}
                else fmt_panel_range(item.beta_topk_f1_mean, item.beta_topk_f1_min, item.beta_topk_f1_max)
            ),
        }
        for output_name, source_name in trajectory_metrics:
            row[output_name] = fmt_panel_range(
                getattr(item, f"{source_name}_mean"),
                getattr(item, f"{source_name}_min"),
                getattr(item, f"{source_name}_max"),
            )
        rows.append(row)
        if method == TEACHER_METHOD:
            transition_row: dict[str, object] = {
                "method": BALL_TRANSITION_LABEL,
                "scenario_panels": int(item.scenario_panels),
                "beta_cosine": fmt_panel_range(
                    item.beta_cosine_mean, item.beta_cosine_min, item.beta_cosine_max
                ),
                "beta_topk_f1": fmt_panel_range(
                    item.beta_topk_f1_mean, item.beta_topk_f1_min, item.beta_topk_f1_max
                ),
            }
            for output_name, _ in trajectory_metrics:
                transition_row[output_name] = NOT_APPLICABLE
            rows.append(transition_row)
    return pd.DataFrame(rows)


def build_table1(agg: pd.DataFrame, out: Path) -> pd.DataFrame:
    summary = panel_summary(agg)
    table = manuscript_summary_table(
        summary,
        [("latent_rmse", "latent_rmse"), ("val_anchor_rmse", "val_anchor_rmse")],
    )
    table.to_csv(out / "table1_primary_comparison.csv", index=False)
    summary.to_csv(out / "table1_primary_comparison_numeric.csv", index=False)
    return summary


def sensitivity_panel_summary(agg: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    methods = [m for m in METHOD_ORDER if m in set(agg["method"])]
    for method in methods:
        g = agg.loc[agg["method"] == method].copy()
        row = {
            "method": method,
            "label": METHOD_LABELS[method],
            "scenario_panels": int(len(g)),
        }
        for metric in metrics:
            vals = g[f"{metric}_mean"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(np.nanmean(vals)) if len(vals) else np.nan
            row[f"{metric}_se"] = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
            row[f"{metric}_min"] = float(np.nanmin(vals)) if len(vals) else np.nan
            row[f"{metric}_max"] = float(np.nanmax(vals)) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_sensitivity_tables(adaptive: pd.DataFrame, irt: pd.DataFrame, out: Path) -> None:
    adaptive_metrics = ["beta_cosine", "beta_topk_f1", "latent_rmse", "val_anchor_rmse"]
    adaptive_summary = sensitivity_panel_summary(adaptive, adaptive_metrics)
    adaptive_table = manuscript_summary_table(
        adaptive_summary,
        [("latent_rmse", "latent_rmse"), ("val_anchor_rmse", "val_anchor_rmse")],
    )
    adaptive_table.to_csv(out / "supp_table6_adaptive_lasso_sensitivity.csv", index=False)
    adaptive_summary.to_csv(out / "supp_table6_adaptive_lasso_sensitivity_numeric.csv", index=False)

    irt_metrics = ["beta_cosine", "beta_topk_f1", "latent_rmse_trait", "irt_item_nll", "irt_expected_total_rmse"]
    irt_summary = sensitivity_panel_summary(irt, irt_metrics)
    irt_table = manuscript_summary_table(
        irt_summary,
        [
            ("trait_latent_rmse", "latent_rmse_trait"),
            ("irt_item_nll", "irt_item_nll"),
            ("irt_expected_total_rmse", "irt_expected_total_rmse"),
        ],
    )
    irt_table.to_csv(out / "supp_table7_irt_sensitivity.csv", index=False)
    irt_summary.to_csv(out / "supp_table7_irt_sensitivity_numeric.csv", index=False)


def write_supp_table1(agg: pd.DataFrame, out: Path) -> None:
    rows = []
    shared_transition_trajectory_rows = {
        PRIMARY_BALL_METHOD,
        TEACHER_METHOD,
        "ball_inherited_dynamics_only",
        "ball_teacher_matching_only",
        "ball_full_decomposition",
    }
    for row in agg.itertuples(index=False):
        coefficient_owner = row.method not in {
            *shared_transition_trajectory_rows,
            "gp_causal_filter",
            "exponential_decay_gru",
        }
        coefficient_value = (
            NOT_SEPARATELY_ESTIMATED
            if row.method in shared_transition_trajectory_rows
            else NO_EXPLICIT_COEFFICIENT
            if row.method in {"gp_causal_filter", "exponential_decay_gru"}
            else None
        )
        rows.append(
            {
                "cell": CELL_LABELS.get(row.cell, row.cell),
                "share": f"{row.share:.2f}",
                "method": METHOD_LABELS.get(row.method, row.method),
                "beta_cosine": fmt(row.beta_cosine_mean, row.beta_cosine_mcse) if coefficient_owner else coefficient_value,
                "beta_topk_f1": fmt(row.beta_topk_f1_mean, row.beta_topk_f1_mcse) if coefficient_owner else coefficient_value,
                "latent_rmse": fmt(row.latent_rmse_mean, row.latent_rmse_mcse),
                "val_anchor_rmse": fmt(row.val_anchor_rmse_mean, row.val_anchor_rmse_mcse),
            }
        )
        if row.method == TEACHER_METHOD:
            rows.append(
                {
                    "cell": CELL_LABELS.get(row.cell, row.cell),
                    "share": f"{row.share:.2f}",
                    "method": BALL_TRANSITION_LABEL,
                    "beta_cosine": fmt(row.beta_cosine_mean, row.beta_cosine_mcse),
                    "beta_topk_f1": fmt(row.beta_topk_f1_mean, row.beta_topk_f1_mcse),
                    "latent_rmse": NOT_APPLICABLE,
                    "val_anchor_rmse": NOT_APPLICABLE,
                }
            )
    pd.DataFrame(rows).to_csv(out / "supp_table1_direct_rdoc_cells.csv", index=False)


def write_distillation_ablation(per_run: pd.DataFrame, out: Path) -> None:
    keys = ["cell", "share", "seed"]
    metrics = ["latent_rmse", "val_anchor_rmse"]
    student = per_run.loc[per_run["method"] == PRIMARY_BALL_METHOD, keys + metrics].copy()
    direct = per_run.loc[per_run["method"] == "ball_direct_causal", keys + metrics].copy()
    paired = student.merge(direct, on=keys, suffixes=("_ball", "_direct"), validate="one_to_one")
    if len(paired) != 100:
        raise ValueError(f"Expected 100 matched distillation-ablation runs; found {len(paired)}")
    paired["latent_rmse_difference"] = paired["latent_rmse_ball"] - paired["latent_rmse_direct"]
    paired["anchor_rmse_difference"] = paired["val_anchor_rmse_ball"] - paired["val_anchor_rmse_direct"]

    rows: list[dict[str, object]] = []

    def add_row(scope: str, level: str, frame: pd.DataFrame) -> None:
        row: dict[str, object] = {"scope": scope, "level": level, "matched_runs": int(len(frame))}
        for metric in ("latent_rmse_difference", "anchor_rmse_difference"):
            values = frame[metric].to_numpy(dtype=float)
            mcse = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            mean = float(np.mean(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_mcse"] = mcse
            row[f"{metric}_ci_low"] = mean - 1.96 * mcse
            row[f"{metric}_ci_high"] = mean + 1.96 * mcse
            row[f"{metric}_ball_wins"] = int(np.sum(values < 0))
        rows.append(row)

    add_row("overall", "All ten simulation conditions", paired)
    for share, frame in paired.groupby("share", sort=True):
        add_row("direct_drift_share", f"{float(share):.2f}", frame)
    for cell, frame in paired.groupby("cell", sort=True):
        add_row("stress_regime", CELL_LABELS.get(cell, cell), frame)

    pd.DataFrame(rows).to_csv(out / "supp_table1b_distillation_ablation.csv", index=False)

    decomposition_methods = [
        PRIMARY_BALL_METHOD,
        "ball_direct_causal",
        "ball_direct_causal_compute_matched",
        "ball_inherited_dynamics_only",
        "ball_teacher_matching_only",
        "ball_full_decomposition",
        "ball_student_transition_only",
        "ball_student_transition_only_strict",
    ]
    decomposition = per_run[per_run["method"].isin(decomposition_methods)].copy()
    if set(decomposition["method"]) != set(decomposition_methods):
        raise ValueError("The simulation distillation and transition-identification arms are incomplete")
    decomposition_rows = []
    for (scope, level, method), frame in pd.concat(
        [
            decomposition.assign(scope="overall", level="All ten simulation conditions"),
            decomposition.assign(scope="direct_drift_share", level=decomposition["share"].map(lambda value: f"{float(value):.2f}")),
            decomposition.assign(scope="stress_regime", level=decomposition["cell"].map(lambda value: CELL_LABELS.get(value, value))),
        ],
        ignore_index=True,
    ).groupby(["scope", "level", "method"], sort=True):
        decomposition_rows.append(
            {
                "scope": scope,
                "level": level,
                "training_arm": METHOD_LABELS.get(method, method),
                "matched_runs": int(len(frame)),
                "latent_rmse_mean": float(frame["latent_rmse"].mean()),
                "latent_rmse_mcse": float(frame["latent_rmse"].std(ddof=1) / np.sqrt(len(frame))),
                "questionnaire_rmse_mean": float(frame["val_anchor_rmse"].mean()),
                "questionnaire_rmse_mcse": float(frame["val_anchor_rmse"].std(ddof=1) / np.sqrt(len(frame))),
            }
        )
    pd.DataFrame(decomposition_rows).to_csv(
        out / "supp_table1c_distillation_decomposition.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "training_arm": "Direct causal transformer",
                "teacher_parameters": "none",
                "generative_model": "jointly fitted with the causal encoder",
                "teacher_distribution_matching": "no",
                "questionnaire_likelihood": "yes",
                "updated_parameters": "causal encoder and generative model",
                "optimizer_update_budget": "direct causal training schedule",
            },
            {
                "training_arm": "Inherited dynamics and questionnaires",
                "teacher_parameters": "frozen after teacher fitting",
                "generative_model": "frozen teacher-fitted model",
                "teacher_distribution_matching": "no",
                "questionnaire_likelihood": "yes",
                "updated_parameters": "causal encoder",
                "optimizer_update_budget": "student training schedule",
            },
            {
                "training_arm": "Teacher matching only",
                "teacher_parameters": "frozen after teacher fitting",
                "generative_model": "frozen teacher-fitted model",
                "teacher_distribution_matching": "yes",
                "questionnaire_likelihood": "no",
                "updated_parameters": "causal encoder",
                "optimizer_update_budget": "student training schedule",
            },
            {
                "training_arm": "Full BALL student",
                "teacher_parameters": "frozen after teacher fitting",
                "generative_model": "frozen teacher-fitted model",
                "teacher_distribution_matching": "yes",
                "questionnaire_likelihood": "yes",
                "updated_parameters": "causal encoder",
                "optimizer_update_budget": "student training schedule after the common teacher fit",
            },
            {
                "training_arm": "Compute-matched direct causal transformer",
                "teacher_parameters": "none",
                "generative_model": "jointly fitted with the causal encoder",
                "teacher_distribution_matching": "no",
                "questionnaire_likelihood": "yes",
                "updated_parameters": "causal encoder and generative model",
                "optimizer_update_budget": "every questionnaire warm-up, teacher-variational, and student-stage update in the full BALL system",
            },
        ]
    ).assign(
        common_teacher_prerequisite=lambda frame: frame["training_arm"].isin(
            [
                "Inherited dynamics and questionnaires",
                "Teacher matching only",
                "Full BALL student",
            ]
        ).map({True: "same fitted bidirectional teacher", False: "none"})
    ).to_csv(out / "supp_table1d_distillation_arm_specification.csv", index=False)


def figure1(out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax in axes:
        ax.set_axis_off()

    def box(ax, xy, text, fc, w=0.34, h=0.16):
        patch = plt.Rectangle(xy, w, h, transform=ax.transAxes, facecolor=fc, edgecolor="#333333", linewidth=1.5)
        ax.add_patch(patch)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10, transform=ax.transAxes)

    ax = axes[0]
    ax.set_title("A. Observed data streams", loc="left", fontsize=14)
    box(ax, (0.04, 0.66), "Sparse recall-windowed\nanchors", "#d9ead3")
    box(ax, (0.04, 0.40), "Longitudinal structured\ncovariates", "#cfe2f3")
    box(ax, (0.04, 0.14), "Lagged treatment and\noptional auxiliary profile", "#fce5cd")
    box(ax, (0.55, 0.40), "Session-level latent\ntrajectory", "#eeeeee")
    for y in [0.74, 0.48, 0.22]:
        ax.annotate("", xy=(0.55, 0.48), xytext=(0.38, y), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})

    ax = axes[1]
    ax.set_title("B. Inference and generative pathways", loc="left", fontsize=14)
    box(ax, (0.04, 0.62), "Bidirectional teacher\ntraining-only smoother", "#d9eaf7")
    box(ax, (0.58, 0.62), "Shared generative\ntransition model", "#f4cccc")
    box(ax, (0.30, 0.18), "Causal student\nprospective filter", "#cfe2f3", w=0.40)
    ax.annotate("", xy=(0.42, 0.35), xytext=(0.22, 0.62), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.annotate("", xy=(0.58, 0.35), xytext=(0.72, 0.62), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.annotate("", xy=(0.38, 0.70), xytext=(0.58, 0.70), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.24, 0.47, "Distillation", ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.text(0.76, 0.47, "Shared dynamics", ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.text(0.48, 0.76, "Joint training", ha="center", va="center", fontsize=9, transform=ax.transAxes)

    ax = axes[2]
    ax.set_title("C. Shared benchmark", loc="left", fontsize=14)
    box(ax, (0.04, 0.72), "BALL teacher and\ncausal student", "#cfe2f3", h=0.14)
    box(ax, (0.04, 0.52), "Direct causal\ntransformer", "#d9ead3", h=0.14)
    box(ax, (0.04, 0.32), "Gaussian process and\nexponential-decay GRU", "#eadcf2", h=0.14)
    box(ax, (0.04, 0.12), "LGSSM and Markov\nmodels", "#eeeeee", h=0.14)
    box(ax, (0.58, 0.36), "Common inputs and\nevaluation targets", "#fce5cd", w=0.36)
    for y in [0.79, 0.59, 0.39, 0.19]:
        ax.annotate("", xy=(0.58, 0.44), xytext=(0.40, y), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})

    ax = axes[3]
    ax.set_title("D. Distinct evaluation targets", loc="left", fontsize=14)
    box(ax, (0.03, 0.62), "Latent trajectory\nerror", "#cfe2f3", w=0.28)
    box(ax, (0.36, 0.62), "Observable anchor\nerror", "#d9ead3", w=0.28)
    box(ax, (0.69, 0.62), "Transition direction\nrecovery", "#f4cccc", w=0.28)
    box(ax, (0.30, 0.20), "Complementary measures\nof performance", "#eeeeee", w=0.40, h=0.22)
    for x in [0.17, 0.50, 0.83]:
        ax.annotate("", xy=(0.50, 0.40), xytext=(x, 0.62), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})

    fig.tight_layout()
    fig.savefig(out / "figure1_schematic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure2(agg: pd.DataFrame, delta: pd.DataFrame, out: Path) -> None:
    summary = panel_summary(agg)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    short_labels = {
        PRIMARY_BALL_METHOD: "BALL\nstudent",
        TEACHER_METHOD: "BALL\nteacher",
        "ball_direct_causal": "Direct causal\ntransformer",
        "gp_causal_filter": "Causal GP\nfilter",
        "exponential_decay_gru": "Exponential-decay\nGRU",
        "s0_direct_lgssm": "LGSSM",
        "markov_direct_transition": "Markov\nmixture",
    }
    labels = [short_labels.get(m, METHOD_LABELS[m]) for m in methods]

    means = [summary.loc[summary["method"] == m, "latent_rmse_mean"].iloc[0] for m in methods]
    axes[0, 0].bar(x, means, color=[COLORS[m] for m in methods])
    axes[0, 0].set_xticks(x, labels, rotation=0, fontsize=9)
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].set_title("A. Latent trajectory error", loc="left")

    coefficient_methods = [m for m in COEFFICIENT_METHOD_ORDER if m in set(summary["method"])]
    coefficient_x = np.arange(len(coefficient_methods))
    coefficient_means = [
        summary.loc[summary["method"] == m, "beta_cosine_mean"].iloc[0]
        for m in coefficient_methods
    ]
    axes[0, 1].bar(coefficient_x, coefficient_means, color=[COLORS[m] for m in coefficient_methods])
    axes[0, 1].set_xticks(
        coefficient_x,
        [COEFFICIENT_LABELS[m].replace(" ", "\n") for m in coefficient_methods],
        rotation=0,
    )
    axes[0, 1].set_ylabel("Cosine")
    axes[0, 1].set_title("B. Shared transition-coefficient recovery", loc="left")

    groups = [f"{CELL_LABELS.get(r.cell, r.cell)}\n{r.share:.2f}" for r in delta.itertuples(index=False)]
    xpos = np.arange(len(groups))
    width = 0.36
    axes[1, 0].bar(xpos - width / 2, delta["ball_minus_s0_latent_rmse_mean"], width, label="BALL minus LGSSM", color="#7aa6d8")
    axes[1, 0].bar(xpos + width / 2, delta["ball_minus_markov_latent_rmse_mean"], width, label="BALL minus Markov", color="#f0a35e")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("C. Paired latent RMSE differences", loc="left")
    axes[1, 0].set_ylabel("Latent RMSE difference")
    axes[1, 0].set_xticks(xpos, groups, rotation=60, ha="right", fontsize=8)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(xpos - width / 2, delta["ball_minus_s0_beta_cosine_mean"], width, label="BALL minus LGSSM", color="#7aa6d8")
    axes[1, 1].bar(xpos + width / 2, delta["ball_minus_markov_beta_cosine_mean"], width, label="BALL minus Markov", color="#f0a35e")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("D. Paired shared-coefficient differences", loc="left")
    axes[1, 1].set_ylabel("Coefficient cosine difference")
    axes[1, 1].set_xticks(xpos, groups, rotation=60, ha="right", fontsize=8)
    axes[1, 1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out / "figure2_recoverability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _heat(ax, values: pd.DataFrame, title: str, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    im = ax.imshow(values.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, loc="left")
    ax.set_xticks(np.arange(values.shape[1]), [f"{float(c):.2f}" for c in values.columns])
    ax.set_yticks(np.arange(values.shape[0]), [CELL_LABELS.get(i, i) for i in values.index])
    ax.set_xlabel("Direct RDoC drift share")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values.iloc[i, j]
            red, green, blue, _ = im.cmap(im.norm(float(val)))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "white" if luminance < 0.45 else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def figure3(delta: pd.DataFrame, out: Path) -> None:
    cells = ["linear", "interaction", "nonlinear", "heterogeneous", "missingness"]
    shares = sorted(delta["share"].unique())
    latent_panels = [
        ("ball_minus_direct_latent_rmse_mean", "A. BALL minus direct causal transformer"),
        ("ball_minus_gp_causal_latent_rmse_mean", "B. BALL minus causal Gaussian-process filter"),
        ("ball_minus_ode_rnn_latent_rmse_mean", "C. BALL minus exponential-decay GRU"),
        ("ball_minus_s0_latent_rmse_mean", "D. BALL minus LGSSM"),
        ("ball_minus_markov_latent_rmse_mean", "E. BALL minus Markov transition model"),
    ]
    anchor_panel = (
        "ball_minus_direct_val_anchor_rmse_mean",
        "F. BALL minus direct causal transformer\nHeld-out anchor RMSE",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 9.4))
    flat = axes.ravel()
    latent_finite = np.concatenate(
        [delta[col].to_numpy(dtype=float) for col, _ in latent_panels if col in delta.columns]
    )
    latent_finite = latent_finite[np.isfinite(latent_finite)]
    latent_bound = float(np.max(np.abs(latent_finite))) if len(latent_finite) else 1.0
    for ax, (metric, title) in zip(flat, latent_panels):
        table = delta.pivot(index="cell", columns="share", values=metric).reindex(index=cells, columns=shares)
        _heat(ax, table, title, "RdBu_r", -latent_bound, latent_bound)
    anchor_metric, anchor_title = anchor_panel
    anchor_table = delta.pivot(index="cell", columns="share", values=anchor_metric).reindex(
        index=cells, columns=shares
    )
    anchor_finite = anchor_table.to_numpy(dtype=float)
    anchor_finite = anchor_finite[np.isfinite(anchor_finite)]
    anchor_bound = float(np.max(np.abs(anchor_finite))) if len(anchor_finite) else 1.0
    _heat(flat[-1], anchor_table, anchor_title, "RdBu_r", -anchor_bound, anchor_bound)
    fig.text(
        0.5,
        0.005,
        "Panels A-E show paired latent RMSE differences. Panel F shows the paired held-out anchor RMSE difference. Negative values favor BALL.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "figure3_stress_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure4(ctrl: pd.DataFrame, out: Path) -> None:
    controls = ["positive", "null", "permuted_proxy", "noise_proxy"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    methods = [m for m in METHOD_ORDER if m in set(ctrl["method"])]
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(controls))
    for idx, (metric, title, ylabel) in enumerate(
        [
            ("beta_cosine_mean", "A. Coefficient-template\nalignment", "Template cosine"),
            ("beta_topk_f1_mean", "B. Active-dimension\ntemplate overlap", "Active-dimension F1"),
            ("latent_rmse_mean", "C. Latent trajectory\nrecovery", "RMSE"),
        ]
    ):
        ax = axes[idx]
        offset0 = -0.5 * width * (len(methods) - 1)
        metric_methods = (
            [m for m in COEFFICIENT_METHOD_ORDER if m in set(ctrl["method"])]
            if metric in {"beta_cosine_mean", "beta_topk_f1_mean"}
            else methods
        )
        width = 0.8 / max(len(metric_methods), 1)
        offset0 = -0.5 * width * (len(metric_methods) - 1)
        for j, method in enumerate(metric_methods):
            vals = []
            for control in controls:
                row = ctrl.loc[(ctrl["control"] == control) & (ctrl["method"] == method)]
                vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            label = (
                COEFFICIENT_LABELS.get(method, METHOD_LABELS[method])
                if metric in {"beta_cosine_mean", "beta_topk_f1_mean"}
                else METHOD_LABELS[method]
            )
            ax.bar(x + offset0 + j * width, vals, width, label=label, color=COLORS[method])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [CONTROL_LABELS[c] for c in controls], rotation=30, ha="right")
        if idx == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        elif idx == 2:
            ax.legend(
                frameon=False,
                fontsize=8,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
            )
    fig.tight_layout()
    fig.savefig(out / "figure4_identifiability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def empirical_outputs(
    emp: Path,
    out: Path,
    *,
    stability: Path | None = None,
    rdoc_scorer: Path | None = None,
) -> None:
    def result_path(name: str) -> Path:
        source = Path(name)
        complete_name = f"{source.stem}_complete{source.suffix}"
        for candidate in (emp / complete_name, emp / name):
            if candidate.exists():
                return candidate
        return emp / complete_name

    static_empirical = ROOT / "empirical" / "derived"
    required_files = [
        "empirical_recovery_overall.csv",
        "empirical_recovery_by_instrument.csv",
        "empirical_generalization.csv",
        "empirical_calibration.csv",
        "empirical_calibration_conditional.csv",
        "empirical_bootstrap_diff.csv",
        "empirical_assessment_burden.csv",
        "empirical_workload_classification.csv",
        "empirical_rdoc_transition.csv",
        "empirical_rdoc_transition_coefficients.csv",
        "empirical_rdoc_transition_full_record_sensitivity.csv",
        "empirical_hyperparameter_selection.csv",
        "empirical_pairwise_bootstrap.csv",
        "empirical_recovery_by_context.csv",
        "empirical_uncertainty_workload.csv",
        "empirical_input_contract.csv",
        "empirical_leakage_audit.csv",
        "empirical_cohort_characteristics.csv",
        "empirical_fit_timing.csv",
    ]
    missing_files = [name for name in required_files if not result_path(name).exists()]
    for static_name in ("build_manifest.json", "rdoc_proxy_manifest.json"):
        if not (static_empirical / static_name).exists():
            missing_files.append(static_name)
    if missing_files:
        raise FileNotFoundError(
            "Publication empirical outputs are incomplete: " + ", ".join(missing_files)
        )

    overall = pd.read_csv(result_path("empirical_recovery_overall.csv"))
    by_instrument = pd.read_csv(result_path("empirical_recovery_by_instrument.csv"))
    generalization = pd.read_csv(result_path("empirical_generalization.csv"))
    calib = pd.read_csv(result_path("empirical_calibration.csv"))
    conditional = pd.read_csv(result_path("empirical_calibration_conditional.csv"))
    boot = pd.read_csv(result_path("empirical_bootstrap_diff.csv"))
    burden = pd.read_csv(result_path("empirical_assessment_burden.csv"))
    workload = pd.read_csv(result_path("empirical_workload_classification.csv"))
    pairwise = pd.read_csv(result_path("empirical_pairwise_bootstrap.csv"))
    context = pd.read_csv(result_path("empirical_recovery_by_context.csv"))
    uncertainty_workload = pd.read_csv(result_path("empirical_uncertainty_workload.csv"))
    input_contract = pd.read_csv(result_path("empirical_input_contract.csv"))
    input_contract = input_contract.rename(columns={"target_embargo": "recall_window_rule"})
    if "prediction_target" in input_contract.columns:
        input_contract["prediction_target"] = (
            "withheld questionnaire total after the fixed instrument-range conversion"
        )
    if "recall_window_rule" in input_contract.columns:
        input_contract["recall_window_rule"] = (
            "retained and withheld questionnaires summarize distinct calendar days"
        )
    leakage_audit = pd.read_csv(result_path("empirical_leakage_audit.csv"))
    cohort_characteristics = pd.read_csv(result_path("empirical_cohort_characteristics.csv"))
    analytic_sessions = ROOT / "empirical" / "data" / "rtms_paper_analytic_sessions.csv"
    if analytic_sessions.is_file() and {"item", "value", "unit"}.issubset(
        cohort_characteristics.columns
    ):
        available_columns = set(pd.read_csv(analytic_sessions, nrows=0).columns)
        patient_column = "PatientFID"
        category_columns = [
            column
            for column in ("protocol_category", "protocol", "is_bilateral", "bilateral_str", "intensity_level")
            if column in available_columns
        ]
        if patient_column in available_columns and category_columns:
            treatment = pd.read_csv(
                analytic_sessions,
                usecols=[patient_column, *category_columns],
            )
            selected_categories = []
            for label, candidates in (
                ("Protocol", ("protocol_category", "protocol")),
                ("Laterality", ("is_bilateral", "bilateral_str")),
                ("Intensity", ("intensity_level",)),
            ):
                column = next((name for name in candidates if name in treatment.columns), None)
                if column is not None:
                    selected_categories.append((label, column))
            existing_items = set(cohort_characteristics["item"].astype(str))
            added_rows = []
            for label, column in selected_categories:
                for category, group in treatment.groupby(column, dropna=False, sort=True):
                    category_label = "unrecorded" if pd.isna(category) else str(category)
                    session_item = f"{label} {category_label} sessions"
                    patient_item = f"{label} {category_label} patients"
                    if session_item not in existing_items:
                        added_rows.append((session_item, int(len(group)), "sessions"))
                    if patient_item not in existing_items:
                        added_rows.append(
                            (patient_item, int(group[patient_column].nunique()), "patients")
                        )
            if added_rows:
                cohort_characteristics = pd.concat(
                    [
                        cohort_characteristics,
                        pd.DataFrame(added_rows, columns=["item", "value", "unit"]),
                    ],
                    ignore_index=True,
                )
    fit_timing = pd.read_csv(result_path("empirical_fit_timing.csv"))
    manifest = json.loads((static_empirical / "build_manifest.json").read_text(encoding="utf-8"))
    proxy_manifest = json.loads((static_empirical / "rdoc_proxy_manifest.json").read_text(encoding="utf-8"))
    transition = pd.read_csv(result_path("empirical_rdoc_transition.csv"))
    transition_coefficients = pd.read_csv(
        result_path("empirical_rdoc_transition_coefficients.csv")
    )
    transition_full_record = pd.read_csv(
        result_path("empirical_rdoc_transition_full_record_sensitivity.csv")
    )
    hyperparameter_selection = pd.read_csv(
        result_path("empirical_hyperparameter_selection.csv")
    )
    fit_manifest_path = emp / "fit_run_manifest.json"
    if not fit_manifest_path.exists():
        raise FileNotFoundError(f"Publication fit manifest is absent: {fit_manifest_path}")
    fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
    if fit_manifest.get("status") != "complete":
        raise ValueError("Publication empirical fit is not complete")
    model_manifest_path = emp / "empirical_teacher_student_manifest_14d.json"
    if not model_manifest_path.exists():
        raise FileNotFoundError(f"The 14-day empirical model manifest is absent: {model_manifest_path}")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))

    frames = {
        "overall recovery": overall,
        "instrument recovery": by_instrument,
        "generalization": generalization,
        "calibration": calib,
        "conditional calibration": conditional,
        "bootstrap comparisons": boot,
        "assessment burden": burden,
        "workload classification": workload,
        "RDoC transition": transition,
        "RDoC transition coefficients": transition_coefficients,
        "RDoC full-record mean sensitivity": transition_full_record,
        "hyperparameter selection": hyperparameter_selection,
        "pairwise comparisons": pairwise,
        "context recovery": context,
        "uncertainty workload": uncertainty_workload,
        "input contract": input_contract,
        "leakage audit": leakage_audit,
        "cohort characteristics": cohort_characteristics,
        "fit timing": fit_timing,
    }
    empty_frames = [name for name, frame in frames.items() if frame.empty]
    if empty_frames:
        raise ValueError(
            "Publication empirical outputs contain empty tables: " + ", ".join(empty_frames)
        )

    ball_key = "ball" if "ball" in set(overall["method"]) else "barl"
    interp_col = "ball_minus_interp_rmse" if "ball_minus_interp_rmse" in boot.columns else "barl_minus_interp_rmse"
    markov_col = "ball_minus_markov_rmse" if "ball_minus_markov_rmse" in boot.columns else "barl_minus_markov_rmse"
    method_labels = {
        ball_key: "BALL causal student",
        "ball_teacher": "BALL teacher smoother",
        "ball_direct_causal": "Direct causal transformer",
        "ball_no_dense_ehr": "BALL using session number and treatment context",
        "ball_anchor_only": "BALL using questionnaires and calendar gaps",
        "gp_causal_filter": "Causal Gaussian-process filter",
        "exponential_decay_gru": "Exponential-decay GRU",
        "s0_direct_lgssm": "LGSSM",
        "markov_direct_transition": "Markov pattern-mixture model",
        "interpolation": "Interpolation",
        "locf": "Last observation carried forward",
        "causal_anchor_mean": "Prior-anchor mean",
        "locf_reference_origin": "Last observation carried forward with reference-origin cold start",
        "causal_anchor_mean_reference_origin": "Prior-anchor mean with reference-origin cold start",
        "ball_inherited_dynamics_only": "BALL inherited dynamics only",
        "ball_teacher_matching_only": "BALL teacher matching only",
        "ball_full_decomposition": "BALL full decomposition arm",
        "ball_direct_compute_matched": "Compute-matched direct causal transformer",
        "ball_session_balanced": "BALL with session-balanced questionnaire weight",
    }
    required_methods = {
        ball_key,
        "ball_teacher",
        "ball_direct_causal",
        "ball_no_dense_ehr",
        "ball_anchor_only",
        "gp_causal_filter",
        "exponential_decay_gru",
        "s0_direct_lgssm",
        "markov_direct_transition",
        "interpolation",
        "locf",
        "causal_anchor_mean",
        "locf_reference_origin",
        "causal_anchor_mean_reference_origin",
        "ball_inherited_dynamics_only",
        "ball_teacher_matching_only",
        "ball_full_decomposition",
        "ball_direct_compute_matched",
        "ball_session_balanced",
    }
    observed_methods = set(overall["method"])
    if observed_methods != required_methods:
        raise ValueError(
            "Publication empirical recovery has an unexpected method set: "
            + ", ".join(sorted(observed_methods))
        )
    cadences_present = {int(value) for value in overall["cadence"].unique()}
    if cadences_present != {14, 21, 28}:
        raise ValueError(f"Expected empirical cadences 14, 21, and 28; found {sorted(cadences_present)}")
    if "forward_2025" not in set(generalization["validation"].astype(str)):
        raise ValueError("The forward-2025 generalization sensitivity is absent")
    method_roles = {
        ball_key: "Prospective estimator trained by teacher-student distillation",
        "ball_direct_causal": "Prospective estimator trained with the causal variational objective",
        "ball_no_dense_ehr": "Prospective estimator using session number and treatment context",
        "ball_anchor_only": "Prospective estimator using questionnaires and calendar gaps",
        "gp_causal_filter": "Prospective nonlinear probabilistic benchmark",
        "exponential_decay_gru": "Prospective recurrent benchmark with learned elapsed-time decay",
        "locf": "Prospective questionnaire-history reference",
        "causal_anchor_mean": "Prospective questionnaire-history reference",
        "ball_teacher": "Retrospective estimator using the complete record",
        "s0_direct_lgssm": "Retrospective state-space reference",
        "markov_direct_transition": "Retrospective Markov reference",
        "interpolation": "Retrospective interpolation using questionnaires on both sides",
        "ball_inherited_dynamics_only": "Prospective decomposition arm using inherited dynamics and questionnaires",
        "ball_teacher_matching_only": "Prospective decomposition arm using teacher distributions",
        "ball_full_decomposition": "Prospective decomposition arm using teacher distributions and questionnaires",
        "ball_direct_compute_matched": "Prospective direct model trained for the full teacher-plus-student update budget",
        "ball_session_balanced": "Prospective BALL sensitivity with total questionnaire weight balanced by session",
    }
    comparison_columns = {
        "ball_direct_causal": ("ball_minus_direct_causal_rmse", "direct_causal_ci_lo", "direct_causal_ci_hi"),
        "ball_no_dense_ehr": ("ball_minus_no_dense_ehr_rmse", "no_dense_ehr_ci_lo", "no_dense_ehr_ci_hi"),
        "ball_anchor_only": ("ball_minus_anchor_only_ball_rmse", "anchor_only_ball_ci_lo", "anchor_only_ball_ci_hi"),
        "locf": ("ball_minus_locf_rmse", "locf_ci_lo", "locf_ci_hi"),
        "causal_anchor_mean": ("ball_minus_causal_anchor_mean_rmse", "causal_anchor_mean_ci_lo", "causal_anchor_mean_ci_hi"),
        "markov_direct_transition": (markov_col, "markov_ci_lo", "markov_ci_hi"),
        "interpolation": (interp_col, "ci_lo", "ci_hi"),
    }
    rows = []
    for cadence in sorted(overall["cadence"].unique()):
        cadence_calibration = calib.loc[calib["cadence"] == cadence]
        c = cadence_calibration.mean(numeric_only=True)
        b = boot.loc[boot["cadence"] == cadence].iloc[0]
        for method, label in method_labels.items():
            val = overall.loc[(overall["cadence"] == cadence) & (overall["method"] == method), "rmse"]
            if not len(val):
                continue
            comparison = ""
            if method in comparison_columns:
                point_col, low_col, high_col = comparison_columns[method]
                if all(column in b.index and np.isfinite(float(b[column])) for column in (point_col, low_col, high_col)):
                    comparison = (
                        f"{float(b[point_col]):.3f} "
                        f"[{float(b[low_col]):.3f}, {float(b[high_col]):.3f}]"
                    )
            if not comparison:
                paired = pairwise.loc[
                    (pairwise["cadence"].astype(int) == int(cadence))
                    & (pairwise["comparator"].astype(str) == str(method))
                ]
                if len(paired):
                    paired_row = paired.iloc[0]
                    comparison = (
                        f"{float(paired_row.rmse_difference):.3f} "
                        f"[{float(paired_row.ci_lo):.3f}, {float(paired_row.ci_hi):.3f}]"
                    )
            rows.append(
                {
                    "cadence": f"{int(cadence)} days",
                    "method": label,
                    "role": method_roles.get(method, ""),
                    "rmse": f"{float(val.iloc[0]):.3f}",
                    "BALL_minus_method_rmse_95_ci": comparison,
                    "patient_balanced_coverage": f"{float(c.coverage):.3f}" if method == ball_key else "",
                    "measurement_coverage": f"{float(c.measurement_coverage):.3f}" if method == ball_key and "measurement_coverage" in c.index else "",
                    "mean_clipped_width_fraction_of_legal_range": (
                        f"{float(c.mean_width_fraction_of_legal_range):.3f}"
                        if method == ball_key else ""
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(out / "supp_table3a_empirical_recovery.csv", index=False)

    burden_out = burden.copy()
    burden_out["cadence"] = burden_out["cadence"].astype(int).map(lambda value: f"{value} days")
    for column in ("retained_fraction", "assessment_reduction_fraction"):
        burden_out[column] = burden_out[column].map(lambda value: f"{float(value):.3f}")
    burden_out.to_csv(out / "supp_table3d_assessment_burden.csv", index=False)

    workload_out = workload.copy()
    workload_out["cadence"] = workload_out["cadence"].astype(int).map(
        lambda value: f"{value} days"
    )
    workload_out["method"] = workload_out["method"].map(
        lambda value: method_labels.get(str(value), str(value))
    )
    for column in (
        "patient_balanced_accuracy",
        "patient_balanced_accuracy_ci_low",
        "patient_balanced_accuracy_ci_high",
        "patient_balanced_prevalence",
        "patient_balanced_sensitivity",
        "patient_balanced_specificity",
        "patient_balanced_balanced_accuracy",
    ):
        workload_out[column] = workload_out[column].map(
            lambda value: f"{float(value):.3f}" if np.isfinite(float(value)) else ""
        )
    workload_out.to_csv(out / "supp_table3e_workload_classification.csv", index=False)

    context_out = context.copy()
    context_out["cadence"] = context_out["cadence"].astype(int).map(lambda value: f"{value} days")
    context_out["method"] = context_out["method"].map(
        lambda value: method_labels.get(str(value), str(value))
    )
    context_out["rmse"] = context_out["rmse"].map(lambda value: f"{float(value):.3f}")
    context_out.to_csv(out / "supp_table3f_empirical_context.csv", index=False)

    pairwise_out = pairwise.copy()
    pairwise_out["cadence"] = pairwise_out["cadence"].astype(int).map(lambda value: f"{value} days")
    pairwise_out["reference"] = pairwise_out["reference"].map(
        lambda value: method_labels.get(str(value), str(value))
    )
    pairwise_out["comparator"] = pairwise_out["comparator"].map(
        lambda value: method_labels.get(str(value), str(value))
    )
    for column in ("rmse_difference", "ci_lo", "ci_hi"):
        pairwise_out[column] = pairwise_out[column].map(lambda value: f"{float(value):.3f}")
    pairwise_out.to_csv(out / "supp_table3g_empirical_pairwise_comparisons.csv", index=False)

    input_contract.to_csv(out / "supp_table3h_empirical_input_contract.csv", index=False)
    leakage_audit.to_csv(out / "supp_table14_empirical_leakage_audit.csv", index=False)
    uncertainty_workload.to_csv(out / "supp_table15_uncertainty_guided_measurement.csv", index=False)
    fit_timing.to_csv(out / "supp_table16_compute_time.csv", index=False)

    if not by_instrument.empty:
        instrument_labels = {"PHQ9": "PHQ-9", "GAD7": "GAD-7", "BDI": "BDI-II"}
        prospective_methods = [
            ball_key,
            "ball_direct_causal",
            "ball_no_dense_ehr",
            "ball_anchor_only",
            "gp_causal_filter",
            "exponential_decay_gru",
            "locf",
            "causal_anchor_mean",
        ]
        instrument_rows = []
        for row in by_instrument.loc[by_instrument["method"].isin(prospective_methods)].itertuples(index=False):
            instrument_rows.append(
                {
                    "cadence": f"{int(row.cadence)} days",
                    "instrument": instrument_labels.get(str(row.instrument), str(row.instrument)),
                    "method": method_labels.get(str(row.method), str(row.method)),
                    "rmse": f"{float(row.rmse):.3f}",
                    "measurements": f"{int(row.n):,}",
                    "patients": f"{int(row.n_patients):,}",
                }
            )
        pd.DataFrame(instrument_rows).to_csv(
            out / "supp_table3b_empirical_by_instrument.csv", index=False
        )

    if not generalization.empty:
        generalization_rows = []
        for row in generalization.itertuples(index=False):
            generalization_rows.append(
                {
                    "validation": str(row.validation).replace("_", " "),
                    "cadence": f"{int(row.cadence)} days",
                    "method": method_labels.get(str(row.method), str(row.method)),
                    "rmse": f"{float(row.rmse):.3f}",
                    "measurements": f"{int(row.n):,}",
                    "training_patients": f"{int(row.n_train_patients):,}",
                    "test_patients": f"{int(row.n_test_patients):,}",
                    "patients_spanning_cutoff_excluded": f"{int(row.spanning_patients_excluded):,}",
                    "training_period": str(row.training_time_rule),
                    "test_period": str(row.test_time_rule),
                }
            )
        pd.DataFrame(generalization_rows).to_csv(
            out / "supp_table3c_empirical_generalization.csv", index=False
        )

    if not transition.empty:
        trows = []
        for row in transition.itertuples(index=False):
            trows.append(
                {
                    "cadence": f"{int(row.cadence)} days",
                    "channel": row.channel,
                    "transition_rows": f"{int(row.n):,}",
                    "patients": f"{int(row.n_patients):,}",
                    "nuisance_rmse": fmt(row.rmse_nuisance),
                    "rdoc_rmse": fmt(row.rmse_rdoc),
                    "rmse_improvement": fmt(row.rmse_improvement),
                    "incremental_r2": fmt(row.incremental_r2),
                    "beta_norm": fmt(row.beta_norm),
                    "top_domain": row.top_domain,
                    "permutation_p": fmt(row.permutation_p),
                    "holm_adjusted_permutation_p": fmt(row.permutation_p_holm),
                }
            )
        pd.DataFrame(trows).to_csv(out / "supp_table5_empirical_rdoc_transition.csv", index=False)

    domain_labels = {
        "B0": "Negative valence",
        "B1": "Positive valence",
        "B2": "Cognitive systems",
        "B3": "Social processes",
        "B4": "Arousal and regulatory systems",
        "B5": "Sensorimotor systems",
    }
    coefficient_table = transition_coefficients.copy()
    domain_column = "domain" if "domain" in coefficient_table.columns else "domain_code"
    coefficient_table["domain_name"] = coefficient_table[domain_column].map(domain_labels)
    coefficient_table.to_csv(
        out / "supp_table5b_empirical_rdoc_coefficients.csv", index=False
    )
    transition_full_record.to_csv(
        out / "supp_table5c_empirical_rdoc_full_record_mean_sensitivity.csv",
        index=False,
    )
    hyperparameter_selection.to_csv(
        out / "supp_table17_empirical_hyperparameter_selection.csv",
        index=False,
    )

    cohort_out = cohort_characteristics.rename(
        columns={"characteristic": "item"}
    ).copy()
    cohort_out["item"] = cohort_out["item"].replace(
        {
            "Patients": "Clinic-exclusive analytic patients",
            "Treatment sessions": "Clinic-exclusive analytic treatment sessions",
            "Clinics": "Canonical analytic clinics",
        }
    )
    additional_cohort_rows = pd.DataFrame(
        [
            {"item": "Patients in source extract", "value": fit_manifest["n_patients"], "unit": "patients"},
            {"item": "Treatment sessions in source extract", "value": fit_manifest["n_sessions"], "unit": "sessions"},
            {"item": "Source network clinics", "value": fit_manifest["source_network_clinic_count"], "unit": "clinics"},
            {"item": "Observed facility strings", "value": fit_manifest["observed_facility_string_count"], "unit": "labels"},
            {"item": "Cross-partition patients excluded", "value": fit_manifest["cross_partition_patients_excluded"], "unit": "patients"},
            {"item": "RDoC proxy session coverage", "value": proxy_manifest["proxy_coverage"], "unit": "fraction"},
            {"item": "Patients with any RDoC proxy", "value": proxy_manifest["patients_with_any_proxy"], "unit": "patients"},
            {"item": "Mean within-patient RDoC proxy standard deviation", "value": proxy_manifest["within_patient_proxy_sd_mean"], "unit": "score"},
            {"item": "Mean between-patient RDoC proxy standard deviation", "value": proxy_manifest["between_patient_proxy_sd_mean"], "unit": "score"},
            {"item": "Calendar window", "value": "January 2023 to June 2025", "unit": "calendar"},
            {
                "item": "Empirical validation target",
                "value": "Withheld questionnaire totals whose recall periods summarize distinct calendar days from retained questionnaires",
                "unit": "definition",
            },
        ]
    )
    pd.concat([cohort_out, additional_cohort_rows], ignore_index=True).to_csv(
        out / "supp_table2a_empirical_cohort.csv", index=False
    )

    comorbidity_labels = {
        "dx_depression": "Depressive disorder",
        "dx_anxiety": "Anxiety disorder",
        "dx_ptsd": "Post-traumatic stress disorder",
        "dx_bipolar": "Bipolar disorder",
        "dx_ocd": "Obsessive-compulsive disorder",
        "dx_adhd": "Attention-deficit/hyperactivity disorder",
        "dx_psychotic_spectrum": "Psychotic-spectrum disorder",
        "dx_substance_use": "Substance-use disorder",
        "dx_eating_disorder": "Eating disorder",
        "dx_personality_disorder": "Personality disorder",
        "dx_autism_spectrum": "Autism-spectrum disorder",
        "dx_sleep_disorder": "Sleep disorder",
    }
    comorbidity_prevalence = fit_manifest.get("comorbidity_prevalence", {})
    missing_comorbidity = sorted(set(comorbidity_labels).difference(comorbidity_prevalence))
    if missing_comorbidity:
        raise ValueError(
            "Publication fit manifest lacks comorbidity prevalence for: "
            + ", ".join(missing_comorbidity)
        )
    comorbidity_rows = [
        {
            "feature": "Prior diagnosis history available",
            "definition": "At least one diagnosis list recorded strictly before the modeled session",
            "active_sessions": f"{int(fit_manifest['comorbidity_prior_list_sessions']):,}",
            "patients_ever_active": f"{int(fit_manifest['comorbidity_patients_with_prior_list']):,}",
        },
        {
            "feature": "Non-depression psychiatric comorbidity count",
            "definition": "Cumulative count across the eleven non-depression categories below",
            "active_sessions": "—",
            "patients_ever_active": "—",
        },
    ]
    for column, label in comorbidity_labels.items():
        counts = comorbidity_prevalence[column]
        comorbidity_rows.append(
            {
                "feature": label,
                "definition": "Cumulative indicator from diagnosis lists recorded strictly before the modeled session",
                "active_sessions": f"{int(counts['sessions']):,}",
                "patients_ever_active": f"{int(counts['patients']):,}",
            }
        )
    pd.DataFrame(comorbidity_rows).to_csv(
        out / "supp_table2b_empirical_comorbidity_features.csv", index=False
    )

    feature_definitions = {
        "treatment_session_number": "Chronological treatment-session count through the current visit",
        "dx_comorbidity_count": "Count of documented psychiatric comorbidity categories other than depression",
        "dx_depression": "Documented depressive disorder",
        "dx_anxiety": "Documented anxiety disorder",
        "dx_ptsd": "Documented post-traumatic stress disorder",
        "dx_bipolar": "Documented bipolar disorder",
        "dx_ocd": "Documented obsessive-compulsive disorder",
        "dx_adhd": "Documented attention-deficit or hyperactivity disorder",
        "dx_psychotic_spectrum": "Documented psychotic-spectrum disorder",
        "dx_substance_use": "Documented substance-use disorder",
        "dx_eating_disorder": "Documented eating disorder",
        "dx_personality_disorder": "Documented personality disorder",
        "dx_autism_spectrum": "Documented autism-spectrum disorder",
        "dx_sleep_disorder": "Documented sleep disorder",
    }
    feature_rows = []
    for name in model_manifest.get("empirical_feature_names", []):
        feature_rows.append(
            {
                "feature": name,
                "definition": feature_definitions.get(name, name.replace("_", " ")),
                "value_scale": "continuous count" if name in {"treatment_session_number", "dx_comorbidity_count"} else "binary indicator",
                "availability": (
                    "known at the current visit"
                    if name == "treatment_session_number"
                    else "diagnosis records dated before the current visit"
                ),
                "preprocessing": "raw value with an explicit availability mask",
            }
        )
    feature_rows.extend(
        [
            {
                "feature": "previous_session_treatment",
                "definition": "Treatment category delivered at the previous session",
                "value_scale": "categorical embedding or one-hot representation",
                "availability": "known before the current questionnaire",
                "preprocessing": "development-patient vocabulary with an unknown category",
            },
            {
                "feature": "elapsed_calendar_days",
                "definition": "Calendar days since the preceding modeled session",
                "value_scale": "continuous days",
                "availability": "known before the current questionnaire",
                "preprocessing": "raw nonnegative count",
            },
        ]
    )
    pd.DataFrame(feature_rows).to_csv(
        out / "supp_table2c_empirical_feature_dictionary.csv", index=False
    )

    has_conditional = not conditional.empty
    fig, axes = plt.subplots(2, 3 if has_conditional else 2, figsize=(15 if has_conditional else 12, 8))
    axes = np.asarray(axes)
    cadences = sorted(overall["cadence"].unique())
    x = np.arange(len(cadences))
    width = 0.16
    empirical_methods = [m for m in [ball_key, "ball_direct_causal", "gp_causal_filter",
                                     "exponential_decay_gru", "locf", "causal_anchor_mean"]
                         if m in set(overall["method"])]
    width = min(0.8 / max(len(empirical_methods), 1), 0.16)
    offset0 = -0.5 * width * (len(empirical_methods) - 1)
    for j, method in enumerate(empirical_methods):
        vals = [overall.loc[(overall["cadence"] == c) & (overall["method"] == method), "rmse"].iloc[0] for c in cadences]
        axes[0, 0].bar(x + offset0 + j * width, vals, width, label=method_labels[method])
    axes[0, 0].set_title("A. Held-out questionnaire RMSE", loc="left")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].set_xticks(x, [f"{int(c)}d" for c in cadences])
    axes[0, 0].legend(frameon=False, fontsize=8)

    comparison_series = [
        ("ball_minus_direct_causal_rmse", "direct_causal_ci_lo", "direct_causal_ci_hi", "o-", "BALL minus direct causal"),
        ("ball_minus_locf_rmse", "locf_ci_lo", "locf_ci_hi", "^-", "BALL minus LOCF"),
        (markov_col, "markov_ci_lo", "markov_ci_hi", "s-", "BALL minus Markov"),
        (interp_col, "ci_lo", "ci_hi", "D-", "BALL minus interpolation"),
    ]
    for point_col, low_col, high_col, marker, label in comparison_series:
        if not all(column in boot.columns for column in (point_col, low_col, high_col)):
            continue
        axes[0, 1].errorbar(
            boot["cadence"],
            boot[point_col],
            yerr=[boot[point_col] - boot[low_col], boot[high_col] - boot[point_col]],
            fmt=marker,
            label=label,
        )
    for comparator, marker in (("gp_causal_filter", "v-"), ("exponential_decay_gru", "P-")):
        subset = pairwise.loc[pairwise["comparator"].astype(str) == comparator].sort_values("cadence")
        if subset.empty:
            continue
        axes[0, 1].errorbar(
            subset["cadence"],
            subset["rmse_difference"],
            yerr=[
                subset["rmse_difference"] - subset["ci_lo"],
                subset["ci_hi"] - subset["rmse_difference"],
            ],
            fmt=marker,
            label=f"BALL minus {method_labels[comparator]}",
        )
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("B. Patient-clustered bootstrap differences", loc="left")
    axes[0, 1].set_ylabel("RMSE difference")
    axes[0, 1].set_xlabel("Cadence, days")
    axes[0, 1].legend(frameon=False, fontsize=8)

    calibration_by_cadence = calib.groupby("cadence", as_index=False).agg(
        coverage=("coverage", "mean"),
        width_fraction=("mean_width_fraction_of_legal_range", "mean"),
    )
    axes[1, 0].plot(calibration_by_cadence["cadence"], calibration_by_cadence["coverage"], "o-", color="#2f6db3")
    axes[1, 0].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_ylim(0.90, 1.00)
    axes[1, 0].set_title("C. Observable-measurement coverage", loc="left")
    axes[1, 0].set_ylabel("Coverage")
    axes[1, 0].set_xlabel("Cadence, days")

    axes[1, 1].plot(calibration_by_cadence["cadence"], calibration_by_cadence["width_fraction"], "s-", color="#d07c2c")
    axes[1, 1].set_title("D. Observable-measurement interval width", loc="left")
    axes[1, 1].set_ylabel("Clipped width / legal score range")
    axes[1, 1].set_xlabel("Cadence, days")

    if has_conditional:
        stratum_types = [
            value
            for value in ["instrument", "gap", "cold_start", "prior_anchor_count", "severity", "treatment_week", "sex", "age", "clinic"]
            if value in set(conditional["stratum_type"])
        ]
        palette = {14: "#2f6db3", 21: "#d07c2c", 28: "#1b9e77"}
        for index, stratum_type in enumerate(stratum_types):
            sub = conditional.loc[conditional["stratum_type"] == stratum_type].copy()
            for cadence in sorted(sub["cadence"].unique()):
                group = sub.loc[sub["cadence"] == cadence]
                jitter = (int(cadence) - 21) / 35.0
                axes[0, 2].scatter(
                    np.full(len(group), index + jitter),
                    group["coverage"],
                    s=np.clip(group["n_patients"].to_numpy(dtype=float), 12, 90),
                    alpha=0.62,
                    color=palette.get(int(cadence), "#666666"),
                    label=f"{int(cadence)} days" if index == 0 else None,
                )
                axes[1, 2].scatter(
                    np.full(len(group), index + jitter),
                    group["mean_width_fraction_of_legal_range"],
                    s=np.clip(group["n_patients"].to_numpy(dtype=float), 12, 90),
                    alpha=0.62,
                    color=palette.get(int(cadence), "#666666"),
                )
        axes[0, 2].axhline(0.95, color="black", linestyle="--", linewidth=1)
        axes[0, 2].set_title("E. Conditional measurement coverage", loc="left")
        axes[0, 2].set_ylabel("Patient-balanced coverage")
        axes[0, 2].legend(frameon=False, fontsize=8)
        axes[1, 2].set_title("F. Conditional interval width", loc="left")
        axes[1, 2].set_ylabel("Clipped width / legal score range")
        for ax in (axes[0, 2], axes[1, 2]):
            ax.set_xticks(np.arange(len(stratum_types)), [value.replace("_", " ") for value in stratum_types], rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(out / "figure5_empirical.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    prospective_display = [
        method
        for method in (
            ball_key,
            "ball_direct_causal",
            "gp_causal_filter",
            "exponential_decay_gru",
            "locf",
            "causal_anchor_mean",
        )
        if method in set(by_instrument["method"])
    ]
    instrument_order = [
        item for item in ("PHQ9", "BDI", "GAD7") if item in set(by_instrument["instrument"])
    ]
    if prospective_display and instrument_order:
        fig, axes = plt.subplots(
            1, len(instrument_order), figsize=(4.6 * len(instrument_order), 3.8), sharey=True
        )
        axes = np.atleast_1d(axes)
        for ax, instrument in zip(axes, instrument_order, strict=True):
            panel = by_instrument[
                (by_instrument["instrument"] == instrument)
                & by_instrument["method"].isin(prospective_display)
            ]
            cadences_panel = sorted(panel["cadence"].astype(int).unique())
            x_panel = np.arange(len(cadences_panel))
            panel_width = min(0.82 / max(len(prospective_display), 1), 0.16)
            start = -0.5 * panel_width * (len(prospective_display) - 1)
            for index, method in enumerate(prospective_display):
                values = []
                for cadence in cadences_panel:
                    selected = panel[
                        (panel["cadence"].astype(int) == int(cadence))
                        & (panel["method"] == method)
                    ]
                    values.append(float(selected["rmse"].iloc[0]) if len(selected) else np.nan)
                ax.bar(
                    x_panel + start + index * panel_width,
                    values,
                    panel_width,
                    label=method_labels[method],
                )
            ax.set_title({"PHQ9": "PHQ-9", "BDI": "BDI-II", "GAD7": "GAD-7"}.get(instrument, instrument))
            ax.set_xticks(x_panel, [f"{value} days" for value in cadences_panel])
            ax.set_xlabel("Assessment cadence")
        axes[0].set_ylabel("Patient-balanced RMSE")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        fig.savefig(out / "supp_figure_empirical_by_instrument.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    context_types = [
        value
        for value in ("cold_start", "prior_anchor_count", "severity", "treatment_week")
        if value in set(context["stratum_type"])
    ]
    if context_types:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
        axes = axes.ravel()
        context_methods = [
            method
            for method in (ball_key, "ball_direct_causal", "gp_causal_filter", "exponential_decay_gru", "locf")
            if method in set(context["method"])
        ]
        for ax, stratum_type in zip(axes, context_types, strict=False):
            panel = context[context["stratum_type"] == stratum_type].copy()
            strata = list(dict.fromkeys(panel["stratum"].astype(str)))
            for method in context_methods:
                method_panel = panel[panel["method"] == method]
                for cadence, linestyle in ((14, "-"), (21, "--"), (28, ":")):
                    cadence_panel = method_panel[method_panel["cadence"].astype(int) == cadence]
                    values = [
                        float(cadence_panel.loc[cadence_panel["stratum"].astype(str) == level, "rmse"].iloc[0])
                        if len(cadence_panel.loc[cadence_panel["stratum"].astype(str) == level])
                        else np.nan
                        for level in strata
                    ]
                    ax.plot(
                        np.arange(len(strata)),
                        values,
                        marker="o",
                        linestyle=linestyle,
                        linewidth=1.1,
                        markersize=3.5,
                        alpha=0.85,
                        label=f"{method_labels[method]}, {cadence} days",
                    )
            ax.set_title(stratum_type.replace("_", " ").title(), loc="left")
            ax.set_xticks(np.arange(len(strata)), [value.replace("_", " ") for value in strata], rotation=30, ha="right")
            ax.set_ylabel("Patient-balanced RMSE")
        for ax in axes[len(context_types):]:
            ax.set_axis_off()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=7)
        fig.tight_layout(rect=(0, 0.20, 1, 1))
        fig.savefig(out / "supp_figure_empirical_context.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if not uncertainty_workload.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
        for cadence in sorted(uncertainty_workload["cadence"].astype(int).unique()):
            cadence_rows = uncertainty_workload[
                uncertainty_workload["cadence"].astype(int) == int(cadence)
            ]
            for width_type, linestyle in (("raw", "-"), ("support_clipped", "--")):
                panel = cadence_rows[
                    cadence_rows["width_type"].astype(str).eq(width_type)
                ].sort_values("interval_width_fraction_threshold")
                label = f"{cadence} days, {width_type.replace('_', ' ')}"
                axes[0].plot(panel["interval_width_fraction_threshold"], panel["additional_measurement_fraction"], marker="o", linestyle=linestyle, label=label)
                axes[1].plot(panel["interval_width_fraction_threshold"], panel["rmse_when_prediction_retained_development_sd_units"], marker="o", linestyle=linestyle, label=label)
                axes[2].plot(panel["interval_width_fraction_threshold"], panel["coverage_when_prediction_retained"], marker="o", linestyle=linestyle, label=label)
        axes[0].set_title("A. Additional measurements", loc="left")
        axes[0].set_ylabel("Fraction of held-out visits")
        axes[1].set_title("B. Error among retained estimates", loc="left")
        axes[1].set_ylabel("Patient-balanced RMSE")
        axes[2].set_title("C. Coverage among retained estimates", loc="left")
        axes[2].set_ylabel("Patient-balanced coverage")
        axes[2].axhline(0.95, color="black", linestyle="--", linewidth=0.9)
        for ax in axes:
            ax.set_xlabel("Maximum clipped width / legal score range")
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "supp_figure_uncertainty_guided_measurement.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if not generalization.empty:
        prospective_generalization = generalization[
            generalization["method"].isin(
                [ball_key, "ball_direct_causal", "locf", "causal_anchor_mean"]
            )
        ].copy()
        if not prospective_generalization.empty:
            fig, ax = plt.subplots(figsize=(7.4, 4.0))
            labels = [method_labels.get(str(value), str(value)) for value in prospective_generalization["method"]]
            ax.barh(np.arange(len(labels)), prospective_generalization["rmse"], color="#4C78A8")
            ax.set_yticks(np.arange(len(labels)), labels)
            ax.invert_yaxis()
            ax.set_xlabel("Patient-balanced RMSE")
            ax.set_title("Forward temporal validation using patients first observed in 2025", loc="left")
            fig.tight_layout()
            fig.savefig(out / "supp_figure_forward_temporal_validation.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    if not transition_coefficients.empty:
        cadences_coef = sorted(transition_coefficients["cadence"].astype(int).unique())
        channels_coef = [
            value for value in ("depression", "anxiety") if value in set(transition_coefficients["channel"])
        ]
        fig, axes = plt.subplots(
            len(channels_coef), len(cadences_coef),
            figsize=(4.2 * len(cadences_coef), 3.2 * len(channels_coef)),
            sharex=True,
        )
        axes = np.asarray(axes).reshape(len(channels_coef), len(cadences_coef))
        for row_index, channel in enumerate(channels_coef):
            for column_index, cadence in enumerate(cadences_coef):
                ax = axes[row_index, column_index]
                panel = transition_coefficients[
                    (transition_coefficients["channel"] == channel)
                    & (transition_coefficients["cadence"].astype(int) == cadence)
                ].copy()
                panel["domain_name"] = panel[domain_column].map(domain_labels)
                panel = panel.set_index(domain_column).reindex(list(domain_labels)).reset_index()
                y = np.arange(len(panel))
                ax.barh(y, panel["coefficient"], color="#4C78A8")
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_yticks(y, panel["domain_name"] if column_index == 0 else [])
                ax.invert_yaxis()
                ax.set_title(f"{channel.capitalize()}, {cadence} days", loc="left")
                ax.set_xlabel("Ridge coefficient for subsequent estimated change")
        fig.tight_layout()
        fig.savefig(out / "supp_figure_empirical_rdoc_coefficients.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if stability is not None:
        stability_table = stability / "ball_stability_results.csv"
        stability_figure = stability / "supplementary_figure_ball_stability.png"
        if not stability_table.is_file() or not stability_figure.is_file():
            raise FileNotFoundError("Protected BALL stability outputs are incomplete")
        pd.read_csv(stability_table).to_csv(out / "supp_table8_ball_stability.csv", index=False)
        shutil.copy2(stability_figure, out / "supp_figure_ball_stability.png")

    if rdoc_scorer is not None:
        scorer_table = rdoc_scorer / "supp_table_single_scorer_transition.csv"
        agreement_table = rdoc_scorer / "scorer_agreement.csv"
        coefficient_sensitivity = rdoc_scorer / "single_scorer_transition_coefficients.csv"
        for path in (scorer_table, agreement_table, coefficient_sensitivity):
            if not path.is_file():
                raise FileNotFoundError(f"RDoC scorer sensitivity output is absent: {path}")
        pd.read_csv(scorer_table).to_csv(out / "supp_table9_rdoc_scorer_sensitivity.csv", index=False)
        pd.read_csv(agreement_table).to_csv(out / "supp_table10_rdoc_scorer_agreement.csv", index=False)
        pd.read_csv(coefficient_sensitivity).to_csv(
            out / "supp_table11_rdoc_scorer_coefficients.csv", index=False
        )


def uncertainty_outputs(calibration: Path, out: Path) -> None:
    """Summarize the manuscript-grade simulation interval comparison."""

    arms_path = calibration / "fig3_arms.csv"
    if not arms_path.exists():
        raise FileNotFoundError(f"Publication calibration output is absent: {arms_path}")
    arms = pd.read_csv(arms_path)
    expected = {
        "student raw",
        "student anchor conformal",
        "student oracle",
        "Markov model-based",
    }
    if set(arms["arm"]) != expected or arms["replicate"].nunique() != 10:
        raise ValueError("Publication interval comparison is incomplete")
    labels = {
        "student raw": "BALL ensemble plus Laplace",
        "student anchor conformal": "BALL conformalized ensemble plus Laplace",
        "student oracle": "Conformal interval calibrated to simulated latent states",
        "Markov model-based": "Markov model-based interval",
    }
    rows = []
    for arm in (
        "student raw",
        "student anchor conformal",
        "student oracle",
        "Markov model-based",
    ):
        group = arms.loc[arms["arm"] == arm]
        n = int(len(group))
        observable_coverage = (
            pd.to_numeric(group["observable_anchor_coverage"], errors="coerce").dropna()
            if "observable_anchor_coverage" in group.columns
            else pd.Series(dtype=float)
        )
        observable_width = (
            pd.to_numeric(group["observable_anchor_rel_width"], errors="coerce").dropna()
            if "observable_anchor_rel_width" in group.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "interval_arm": labels[arm],
                "latent_coverage_mean": f"{float(group['coverage'].mean()):.3f}",
                "latent_coverage_mcse": f"{float(group['coverage'].std(ddof=1) / np.sqrt(n)):.3f}",
                "relative_width_mean": f"{float(group['rel_width'].mean()):.3f}",
                "relative_width_mcse": f"{float(group['rel_width'].std(ddof=1) / np.sqrt(n)):.3f}",
                "observable_anchor_coverage_mean": (
                    f"{float(observable_coverage.mean()):.3f}"
                    if len(observable_coverage)
                    else ""
                ),
                "observable_anchor_coverage_mcse": (
                    f"{float(observable_coverage.std(ddof=1) / np.sqrt(len(observable_coverage))):.3f}"
                    if len(observable_coverage) > 1
                    else ""
                ),
                "observable_anchor_relative_width_mean": (
                    f"{float(observable_width.mean()):.3f}"
                    if len(observable_width)
                    else ""
                ),
                "observable_anchor_relative_width_mcse": (
                    f"{float(observable_width.std(ddof=1) / np.sqrt(len(observable_width))):.3f}"
                    if len(observable_width) > 1
                    else ""
                ),
                "replicates": n,
            }
        )
    pd.DataFrame(rows).to_csv(
        out / "supp_table13_uncertainty_calibration.csv", index=False
    )


def write_control_table(ctrl: pd.DataFrame, out: Path) -> None:
    rows = []
    for row in ctrl.itertuples(index=False):
        coefficient_owner = row.method not in {PRIMARY_BALL_METHOD, TEACHER_METHOD}
        rows.append(
            {
                "control": CONTROL_LABELS.get(row.control, row.control),
                "method": METHOD_LABELS.get(row.method, row.method),
                "template_cosine": fmt(row.beta_cosine_mean, row.beta_cosine_mcse) if coefficient_owner else NOT_SEPARATELY_ESTIMATED,
                "template_topk_f1": fmt(row.beta_topk_f1_mean, row.beta_topk_f1_mcse) if coefficient_owner else NOT_SEPARATELY_ESTIMATED,
                "latent_rmse": fmt(row.latent_rmse_mean, row.latent_rmse_mcse),
                "val_anchor_rmse": fmt(row.val_anchor_rmse_mean, row.val_anchor_rmse_mcse),
            }
        )
        if row.method == TEACHER_METHOD:
            rows.append(
                {
                    "control": CONTROL_LABELS.get(row.control, row.control),
                    "method": BALL_TRANSITION_LABEL,
                    "template_cosine": fmt(row.beta_cosine_mean, row.beta_cosine_mcse),
                    "template_topk_f1": fmt(row.beta_topk_f1_mean, row.beta_topk_f1_mcse),
                    "latent_rmse": NOT_APPLICABLE,
                    "val_anchor_rmse": NOT_APPLICABLE,
                }
            )
    pd.DataFrame(rows).to_csv(out / "supp_table4_negative_controls.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build direct-RDoC manuscript tables and figures.")
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--adaptive", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--irt", type=Path, default=DEFAULT_IRT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--empirical", type=Path, default=DEFAULT_EMPIRICAL)
    parser.add_argument(
        "--stability",
        type=Path,
        default=None,
        help="Protected directory containing aggregate BALL stability outputs.",
    )
    parser.add_argument(
        "--rdoc-scorer",
        type=Path,
        default=None,
        help="Directory containing aggregate RDoC scorer-sensitivity outputs.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    primary_per_run_path = args.primary / "per_run.csv"
    primary_manifest_path = args.primary / "manifest.json"
    if not primary_per_run_path.exists() or not primary_manifest_path.exists():
        raise FileNotFoundError(
            "The final corrected benchmark requires per_run.csv and manifest.json"
        )
    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    snapshot_dir = args.primary / "source_snapshot"
    for source_name, manifest_key in (
        ("BALL.py", "ball_py_sha256"),
        ("direct_rdoc_benchmark.py", "benchmark_py_sha256"),
        ("direct_rdoc_fair_comparator.py", "fair_comparator_py_sha256"),
        ("direct_rdoc_common.py", "common_py_sha256"),
        ("ball_validation_harness.py", "harness_py_sha256"),
    ):
        source_path = snapshot_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Simulation source snapshot is absent: {source_path}")
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if primary_manifest.get(manifest_key) != source_hash:
            raise ValueError(f"Simulation manifest hash does not match {source_name}")
    primary_per_run = pd.read_csv(primary_per_run_path)
    primary_counts = primary_per_run.groupby("method").size().to_dict()
    expected_primary_methods = {
        "ball_teacher_smoother",
        "ball_student_causal",
        "ball_direct_causal",
        "gp_causal_filter",
        "exponential_decay_gru",
        "s0_direct_lgssm",
        "markov_direct_transition",
        "ball_direct_causal_compute_matched",
        "ball_inherited_dynamics_only",
        "ball_teacher_matching_only",
        "ball_full_decomposition",
        "ball_student_transition_only",
        "ball_student_transition_only_strict",
    }
    if len(primary_per_run) != 1300 or set(primary_counts) != expected_primary_methods:
        raise ValueError(
            f"Expected 1,300 rows from thirteen primary methods; found {len(primary_per_run)} rows and "
            f"methods {sorted(primary_counts)}"
        )
    bad_counts = {method: count for method, count in primary_counts.items() if int(count) != 100}
    if bad_counts:
        raise ValueError(f"Each primary method must have 100 matched fits; found {bad_counts}")

    agg = pd.read_csv(args.primary / "aggregate.csv")
    delta = pd.read_csv(args.primary / "paired_delta_aggregate.csv")
    ctrl = pd.read_csv(args.controls / "aggregate.csv")
    adaptive = pd.read_csv(args.adaptive / "aggregate.csv")
    irt = pd.read_csv(args.irt / "aggregate.csv")
    # pandas parses the literal control name "null" as NaN; restore it so the
    # null-control rows keep their label in Supplementary Table 5 and Figure 4.
    ctrl["control"] = ctrl["control"].fillna("null")
    # This build requires the teacher/student outputs; refuse to silently fall
    # back to the legacy single-transformer artifacts.
    for _name, _frame in (("benchmark", agg), ("negative-controls", ctrl), ("adaptive-sensitivity", adaptive), ("irt-sensitivity", irt)):
        if PRIMARY_BALL_METHOD not in set(_frame["method"]):
            raise SystemExit(
                f"Primary method '{PRIMARY_BALL_METHOD}' is absent from the {_name} "
                f"aggregate (methods present: {sorted(set(_frame['method']))}). "
                f"Refusing to fall back to legacy '{LEGACY_BALL_METHOD}'."
            )
    aggregate_counts = agg.groupby("method").size().to_dict()
    if set(aggregate_counts) != expected_primary_methods or any(int(count) != 10 for count in aggregate_counts.values()):
        raise ValueError(
            "Primary aggregate must contain ten matched scenario panels for each of thirteen methods; "
            f"found {aggregate_counts}"
        )
    if len(delta) != 10:
        raise ValueError(f"Expected ten paired scenario rows; found {len(delta)}")

    summary = build_table1(agg, args.out)
    write_supp_table1(agg, args.out)
    write_distillation_ablation(primary_per_run, args.out)
    write_sensitivity_tables(adaptive, irt, args.out)
    write_control_table(ctrl, args.out)
    figure1(args.out)
    figure2(agg, delta, args.out)
    figure3(delta, args.out)
    figure4(ctrl, args.out)
    uncertainty_outputs(args.calibration, args.out)
    empirical_outputs(
        args.empirical,
        args.out,
        stability=args.stability,
        rdoc_scorer=args.rdoc_scorer,
    )

    manifest = {
        "primary_source": str(args.primary),
        "control_source": str(args.controls),
        "adaptive_lasso_sensitivity_source": str(args.adaptive),
        "irt_sensitivity_source": str(args.irt),
        "uncertainty_calibration_source": str(args.calibration),
        "empirical_source": str(args.empirical),
        "stability_source": str(args.stability) if args.stability is not None else None,
        "rdoc_scorer_source": str(args.rdoc_scorer) if args.rdoc_scorer is not None else None,
        "outputs": [
            "figure1_schematic.png",
            "figure2_recoverability.png",
            "figure3_stress_comparison.png",
            "figure4_identifiability.png",
            "figure5_empirical.png",
            "table1_primary_comparison.csv",
            "table1_primary_comparison_numeric.csv",
            "supp_table1_direct_rdoc_cells.csv",
            "supp_table1b_distillation_ablation.csv",
            "supp_table1c_distillation_decomposition.csv",
            "supp_table1d_distillation_arm_specification.csv",
            "supp_table2a_empirical_cohort.csv",
            "supp_table2b_empirical_comorbidity_features.csv",
            "supp_table2c_empirical_feature_dictionary.csv",
            "supp_table3a_empirical_recovery.csv",
            "supp_table3b_empirical_by_instrument.csv",
            "supp_table3c_empirical_generalization.csv",
            "supp_table3d_assessment_burden.csv",
            "supp_table3e_workload_classification.csv",
            "supp_table3f_empirical_context.csv",
            "supp_table3g_empirical_pairwise_comparisons.csv",
            "supp_table3h_empirical_input_contract.csv",
            "supp_table4_negative_controls.csv",
            "supp_table5_empirical_rdoc_transition.csv",
            "supp_table5b_empirical_rdoc_coefficients.csv",
            "supp_table5c_empirical_rdoc_full_record_mean_sensitivity.csv",
            "supp_table6_adaptive_lasso_sensitivity.csv",
            "supp_table6_adaptive_lasso_sensitivity_numeric.csv",
            "supp_table7_irt_sensitivity.csv",
            "supp_table7_irt_sensitivity_numeric.csv",
            "supp_table13_uncertainty_calibration.csv",
            "supp_table14_empirical_leakage_audit.csv",
            "supp_table15_uncertainty_guided_measurement.csv",
            "supp_table16_compute_time.csv",
            "supp_table17_empirical_hyperparameter_selection.csv",
            "supp_figure_empirical_by_instrument.png",
            "supp_figure_empirical_context.png",
            "supp_figure_uncertainty_guided_measurement.png",
            "supp_figure_forward_temporal_validation.png",
            "supp_figure_empirical_rdoc_coefficients.png",
        ],
    }
    if args.stability is not None:
        manifest["outputs"].extend(
            ["supp_table8_ball_stability.csv", "supp_figure_ball_stability.png"]
        )
    if args.rdoc_scorer is not None:
        manifest["outputs"].extend(
            [
                "supp_table9_rdoc_scorer_sensitivity.csv",
                "supp_table10_rdoc_scorer_agreement.csv",
                "supp_table11_rdoc_scorer_coefficients.csv",
            ]
        )
    missing_declared_outputs = [
        name for name in manifest["outputs"] if not (args.out / name).is_file()
    ]
    if missing_declared_outputs:
        raise FileNotFoundError(
            "The publication builder failed to create declared outputs: "
            + ", ".join(missing_declared_outputs)
        )
    (args.out / "direct_rdoc_manuscript_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manuscript outputs to {args.out}")
    print(summary[["label", "beta_cosine_mean", "latent_rmse_mean", "val_anchor_rmse_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
