"""Within-patient stability versus time-matched between-patient separation.

The input is the all-session causal-student transition export produced by
``BALL.py empirical-fit``. Patient-level values remain in memory only. The
script writes an aggregate table, a publication figure, and a provenance
manifest to a protected output directory outside every Git working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260804
N_BOOT = 5000


def _git_ancestor(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def patient_face_validity(
    transitions: pd.DataFrame,
    cadence: int,
    channel: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    frame = transitions[
        transitions["cadence"].eq(cadence)
        & transitions["channel"].eq(channel)
        & transitions["day"].between(0, 83)
        & transitions["next_day"].between(1, 84)
    ].copy()
    frame["treatment_week"] = (frame["day"] // 7).astype(int)

    patient_week = (
        frame.groupby(["id", "treatment_week"], as_index=False)
        .agg(
            latent_state=("L_current", "mean"),
            within_change=("delta", lambda values: np.abs(values).mean()),
            n_transitions=("delta", "size"),
        )
    )
    week_counts = patient_week.groupby("treatment_week")["id"].transform("nunique")
    patient_week = patient_week[week_counts.ge(10)].copy()
    patient_week["between_difference"] = np.nan
    for _, indices in patient_week.groupby("treatment_week").groups.items():
        values = patient_week.loc[indices, "latent_state"].to_numpy(dtype=float)
        distances = np.abs(values[:, None] - values[None, :])
        np.fill_diagonal(distances, np.nan)
        patient_week.loc[indices, "between_difference"] = np.nanmedian(distances, axis=1)

    patient = (
        patient_week.groupby("id")
        .agg(
            within_change=("within_change", "mean"),
            between_difference=("between_difference", "mean"),
            n_weeks=("treatment_week", "nunique"),
            n_transitions=("n_transitions", "sum"),
        )
        .dropna()
    )
    patient = patient[patient["n_transitions"].ge(2)].copy()
    if patient.empty:
        raise ValueError(f"No eligible patients for {cadence}-day {channel} analysis")
    patient["paired_difference"] = patient["between_difference"] - patient["within_change"]
    patient["cadence"] = cadence
    patient["channel"] = channel

    values = patient[["within_change", "between_difference", "paired_difference"]].to_numpy()
    boot = np.empty((N_BOOT, 4), dtype=float)
    n = len(patient)
    for replicate in range(N_BOOT):
        sampled = values[rng.integers(0, n, n)]
        boot[replicate, 0] = np.median(sampled[:, 0])
        boot[replicate, 1] = np.median(sampled[:, 1])
        boot[replicate, 2] = np.median(sampled[:, 2])
        boot[replicate, 3] = np.mean(sampled[:, 2] > 0)

    within_ci = percentile_ci(boot[:, 0])
    between_ci = percentile_ci(boot[:, 1])
    paired_ci = percentile_ci(boot[:, 2])
    proportion_ci = percentile_ci(boot[:, 3])
    summary = {
        "cadence_days": cadence,
        "channel": channel,
        "patients": n,
        "patient_weeks": int(len(patient_week)),
        "median_within_change": float(patient["within_change"].median()),
        "median_within_change_ci_lo": within_ci[0],
        "median_within_change_ci_hi": within_ci[1],
        "median_between_difference": float(patient["between_difference"].median()),
        "median_between_difference_ci_lo": between_ci[0],
        "median_between_difference_ci_hi": between_ci[1],
        "median_paired_difference": float(patient["paired_difference"].median()),
        "median_paired_difference_ci_lo": paired_ci[0],
        "median_paired_difference_ci_hi": paired_ci[1],
        "proportion_between_gt_within": float(np.mean(patient["paired_difference"] > 0)),
        "proportion_between_gt_within_ci_lo": proportion_ci[0],
        "proportion_between_gt_within_ci_hi": proportion_ci[1],
    }
    return patient.reset_index(drop=True), summary


def make_figure(primary: pd.DataFrame, summaries: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
        }
    )
    colors = {"depression": "#0072B2", "anxiety": "#D55E00"}
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True, sharey=True)
    ticks = [0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    for panel, (axis, channel) in enumerate(zip(axes, ["depression", "anxiety"], strict=True)):
        data = primary[primary["channel"].eq(channel)]
        row = summaries[
            summaries["cadence_days"].eq(14) & summaries["channel"].eq(channel)
        ].iloc[0]
        axis.scatter(
            data["within_change"],
            data["between_difference"],
            s=10,
            alpha=0.28,
            color=colors[channel],
            edgecolors="none",
            rasterized=True,
        )
        axis.plot([0.025, 2.1], [0.025, 2.1], color="#4D4D4D", linewidth=0.9, linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(0.025, 2.1)
        axis.set_ylim(0.025, 2.1)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.set_title(channel.capitalize())
        axis.set_xlabel("Within-patient change (standardized latent units)")
        if panel == 0:
            axis.set_ylabel("Between-patient difference (standardized latent units)")
        axis.text(
            0.04,
            0.94,
            f"n = {int(row['patients']):,}\n{100 * row['proportion_between_gt_within']:.1f}% above identity",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BDBDBD"},
        )
        axis.text(-0.12, 1.04, chr(ord("A") + panel), transform=axis.transAxes, fontweight="bold")
        axis.grid(False)
    figure.tight_layout(w_pad=1.6)
    figure.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    transitions_path = args.transitions.expanduser().resolve()
    output_dir = args.out.expanduser().resolve()
    if not transitions_path.is_file():
        raise FileNotFoundError(f"Transition export not found: {transitions_path}")
    if _git_ancestor(output_dir) is not None:
        raise ValueError("Patient-derived stability artifacts must be written outside every Git working tree")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_columns = pd.read_csv(transitions_path, nrows=0).columns
    if any(str(column).startswith("B") for column in source_columns):
        raise ValueError("The stability source must not be selected on RDoC availability")
    transitions = pd.read_csv(
        transitions_path,
        usecols=["cadence", "channel", "id", "day", "next_day", "dt", "L_current", "delta"],
    )
    if not transitions["dt"].gt(0).all():
        raise ValueError("Transition gaps must be strictly positive")

    rng = np.random.default_rng(SEED)
    patient_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for cadence in (14, 21, 28):
        for channel in ("depression", "anxiety"):
            patient, summary = patient_face_validity(transitions, cadence, channel, rng)
            patient_frames.append(patient)
            summaries.append(summary)
    patient_summary = pd.concat(patient_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "ball_stability_results.csv", index=False)
    make_figure(
        patient_summary[patient_summary["cadence"].eq(14)],
        summary_frame,
        output_dir / "supplementary_figure_ball_stability.png",
    )
    manifest = {
        "analysis": "BALL within-patient stability versus time-matched between-patient separation",
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "primary_cadence_days": 14,
        "sensitivity_cadences_days": [21, 28],
        "treatment_day_window": [0, 84],
        "minimum_patients_per_week_bin": 10,
        "minimum_transitions_per_patient": 2,
        "rdoc_selection": "none; all positive-gap adjacent modeled sessions are eligible",
        "same_day_policy": "zero-day pairs are excluded because reliable within-day ordering is unavailable",
        "patient_level_output_written": False,
        "source_sha256": sha256(transitions_path),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
