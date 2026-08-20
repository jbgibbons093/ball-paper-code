'''Summarize frozen empirical RDoC transition results.'''

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RDOC_COLS = [f"B{j}" for j in range(6)]
RDOC_NUISANCE_COLS = ["L_current", "dt", "rdoc_days_stale"]
RDOC_RIDGE = 1.0
RDOC_FOLDS = 5
RDOC_PERMUTATIONS = 200
MAX_HELDOUT_PATIENTS = 345


def _design_matrix(
    frame: pd.DataFrame,
    features: list[str],
    norm: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    values = frame[features].to_numpy(dtype=float)
    if norm is None:
        mean = np.nanmean(values, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        filled = np.where(np.isfinite(values), values, mean)
        standard_deviation = np.nanstd(filled, axis=0)
        standard_deviation = np.where(
            (standard_deviation > 1e-8) & np.isfinite(standard_deviation),
            standard_deviation,
            1.0,
        )
    else:
        mean, standard_deviation = norm
        filled = np.where(np.isfinite(values), values, mean)
    standardized = (filled - mean) / standard_deviation
    return np.column_stack([np.ones(len(frame)), standardized]), (mean, standard_deviation)


def _ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_design, norm = _design_matrix(train, features)
    train_outcome = train["delta"].to_numpy(dtype=float)
    test_design, _ = _design_matrix(test, features, norm)
    penalty = np.eye(train_design.shape[1])
    penalty[0, 0] = 0.0
    left = train_design.T @ train_design + RDOC_RIDGE * penalty
    right = train_design.T @ train_outcome
    try:
        coefficient = np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.pinv(left) @ right
    return test_design @ coefficient, coefficient


def _patient_folds(frame: pd.DataFrame) -> list[np.ndarray]:
    patient_ids = np.asarray(sorted(frame["id"].unique()), dtype=int)
    generator = np.random.default_rng(29)
    generator.shuffle(patient_ids)
    fold_count = min(RDOC_FOLDS, len(patient_ids))
    if fold_count < 2:
        return []
    return [fold for fold in np.array_split(patient_ids, fold_count) if len(fold)]


def _cross_validated_error(
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[float, float]:
    outcomes: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for test_ids in _patient_folds(frame):
        is_test = frame["id"].isin(test_ids)
        train = frame.loc[~is_test]
        test = frame.loc[is_test]
        if len(train) <= len(features) + 1 or test.empty:
            continue
        predicted, _ = _ridge_predict(train, test, features)
        outcomes.append(test["delta"].to_numpy(dtype=float))
        predictions.append(predicted)
    if not outcomes:
        return float("nan"), float("nan")
    outcome = np.concatenate(outcomes)
    prediction = np.concatenate(predictions)
    residual = outcome - prediction
    return float(np.sqrt(np.mean(residual**2))), float(np.sum(residual**2))


def _summarize_group(
    frame: pd.DataFrame,
    cadence: int,
    channel: str,
    patient_mean_mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    frame = frame.dropna(subset=["delta", *RDOC_NUISANCE_COLS, *RDOC_COLS]).copy()
    frame = frame.sort_values(["id", "day", "next_day"], kind="mergesort").reset_index(drop=True)
    patients_with_any_profile = int(frame["id"].nunique())
    if patients_with_any_profile > MAX_HELDOUT_PATIENTS:
        raise RuntimeError(
            f"RDoC analysis contains {patients_with_any_profile} patients, exceeding "
            f"the held-out-clinic maximum of {MAX_HELDOUT_PATIENTS}."
        )
    profile_counts = frame.groupby("id", sort=False).size()
    patients_with_two_profiles = int(profile_counts.ge(2).sum())

    between_columns: list[str] = []
    within_columns: list[str] = []
    for column in RDOC_COLS:
        between_column = f"{column}_between"
        within_column = f"{column}_within"
        if patient_mean_mode == "expanding_strictly_earlier":
            cumulative = frame.groupby("id", sort=False)[column].cumsum() - frame[column]
            prior_count = frame.groupby("id", sort=False).cumcount()
            frame[between_column] = cumulative / prior_count.replace(0, np.nan)
        elif patient_mean_mode == "full_record_retrospective":
            frame[between_column] = frame.groupby("id")[column].transform("mean")
        else:
            raise ValueError(f"Unknown patient mean mode: {patient_mean_mode}")
        frame[within_column] = frame[column] - frame[between_column]
        between_columns.append(between_column)
        within_columns.append(within_column)

    frame = frame.dropna(subset=between_columns + within_columns).reset_index(drop=True)
    base_features = [*RDOC_NUISANCE_COLS, *between_columns]
    full_features = [*base_features, *within_columns]
    base_rmse, base_sse = _cross_validated_error(frame, base_features)
    full_rmse, full_sse = _cross_validated_error(frame, full_features)
    improvement = base_rmse - full_rmse
    incremental_r_squared = 1.0 - full_sse / base_sse

    _, fitted_coefficient = _ridge_predict(frame, frame, full_features)
    coefficient_map = dict(zip(["intercept", *full_features], fitted_coefficient))
    within_coefficient = np.asarray(
        [coefficient_map.get(column, 0.0) for column in within_columns],
        dtype=float,
    )

    generator = np.random.default_rng(1100 + cadence + (0 if channel == "depression" else 100))
    permutation_at_least_as_large = 0
    completed_permutations = 0
    for _ in range(RDOC_PERMUTATIONS):
        permuted = frame.copy()
        for indices in permuted.groupby("id", sort=False).groups.values():
            indices = np.asarray(list(indices), dtype=int)
            if len(indices) < 2:
                continue
            shift = int(generator.integers(1, len(indices)))
            permuted.loc[indices, within_columns] = np.roll(
                permuted.loc[indices, within_columns].to_numpy(dtype=float),
                shift=shift,
                axis=0,
            )
        permuted_rmse, _ = _cross_validated_error(permuted, full_features)
        if np.isfinite(permuted_rmse):
            completed_permutations += 1
            permutation_at_least_as_large += int(base_rmse - permuted_rmse >= improvement)
    permutation_probability = (
        (permutation_at_least_as_large + 1.0) / (completed_permutations + 1.0)
        if completed_permutations
        else float("nan")
    )

    estimand = (
        "current profile deviation from the mean of strictly earlier profiles"
        if patient_mean_mode == "expanding_strictly_earlier"
        else "current profile deviation from the full-record patient mean"
    )
    summary: dict[str, object] = {
        "cadence": cadence,
        "channel": channel,
        "n": int(len(frame)),
        "n_patients": int(frame["id"].nunique()),
        "patients_with_any_profile": patients_with_any_profile,
        "patients_with_at_least_two_profiles": patients_with_two_profiles,
        "eligible_transitions_after_prior_profile_requirement": int(len(frame)),
        "rmse_nuisance": base_rmse,
        "rmse_rdoc": full_rmse,
        "rmse_improvement": improvement,
        "incremental_r2": incremental_r_squared,
        "beta_norm": float(np.linalg.norm(within_coefficient)),
        "top_domain": RDOC_COLS[int(np.argmax(np.abs(within_coefficient)))],
        "permutation_p": permutation_probability,
        "permutation_scheme": "within-patient circular time shift",
        "rdoc_estimand": estimand,
        "patient_mean_mode": patient_mean_mode,
        "current_profile_in_reference_mean": patient_mean_mode == "full_record_retrospective",
    }
    coefficients = [
        {
            "cadence": cadence,
            "channel": channel,
            "domain": domain,
            "coefficient": float(coefficient_map.get(within_column, 0.0)),
            "estimand": estimand,
            "patient_mean_mode": patient_mean_mode,
        }
        for domain, within_column in zip(RDOC_COLS, within_columns)
    ]
    return summary, coefficients


def _holm(values: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(raw), np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(raw))
    if finite_positions.size:
        ordered_positions = finite_positions[np.argsort(raw[finite_positions], kind="mergesort")]
        ordered_raw = raw[ordered_positions]
        ordered_adjusted = np.maximum.accumulate(
            (len(ordered_raw) - np.arange(len(ordered_raw))) * ordered_raw
        )
        adjusted[ordered_positions] = np.minimum(ordered_adjusted, 1.0)
    return adjusted


def summarize(transitions: pd.DataFrame, patient_mean_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    for (cadence, channel), group in transitions.groupby(["cadence", "channel"], sort=True):
        summary, coefficient_rows = _summarize_group(
            group,
            int(cadence),
            str(channel),
            patient_mean_mode,
        )
        summaries.append(summary)
        coefficients.extend(coefficient_rows)
    summary_frame = pd.DataFrame(summaries)
    summary_frame["permutation_p_holm"] = _holm(summary_frame["permutation_p"])
    summary_frame["multiplicity_family"] = (
        "three questionnaire schedules crossed with two latent channels"
    )
    return summary_frame, pd.DataFrame(coefficients)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output_directory = arguments.output_dir or arguments.input_dir

    transition_frames = [
        pd.read_csv(arguments.input_dir / f"empirical_rdoc_transition_rows_{cadence}d.csv")
        for cadence in (14, 21, 28)
    ]
    transitions = pd.concat(transition_frames, ignore_index=True)
    if transitions["id"].nunique() > MAX_HELDOUT_PATIENTS:
        raise RuntimeError("Combined transition rows exceed the held-out-clinic patient limit.")

    primary, primary_coefficients = summarize(transitions, "expanding_strictly_earlier")
    retrospective, retrospective_coefficients = summarize(
        transitions,
        "full_record_retrospective",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    primary.to_csv(output_directory / "empirical_rdoc_transition_final.csv", index=False)
    primary_coefficients.to_csv(
        output_directory / "empirical_rdoc_transition_coefficients_final.csv",
        index=False,
    )
    retrospective.to_csv(
        output_directory / "empirical_rdoc_transition_full_record_sensitivity_final.csv",
        index=False,
    )
    retrospective_coefficients.to_csv(
        output_directory / "empirical_rdoc_transition_full_record_coefficients_final.csv",
        index=False,
    )

    manifest = pd.DataFrame(
        [
            {
                "input_patient_count": int(transitions["id"].nunique()),
                "input_transition_count": int(len(transitions)),
                "primary_result_rows": int(len(primary)),
                "retrospective_result_rows": int(len(retrospective)),
                "maximum_allowed_patients": MAX_HELDOUT_PATIENTS,
                "patient_limit_passed": bool(transitions["id"].nunique() <= MAX_HELDOUT_PATIENTS),
            }
        ]
    )
    manifest.to_csv(output_directory / "empirical_rdoc_transition_postprocess_manifest.csv", index=False)


if __name__ == "__main__":
    main()
