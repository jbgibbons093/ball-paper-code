from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "validation" / "outputs" / "direct_rdoc_gaussian_publication_20260805_matched_heldout_v4"
DEFAULT_CALIBRATION = ROOT / "simulations" / "paper" / "outputs" / "publication_calibration_20260805_v4"
DEFAULT_EMPIRICAL = ROOT / "empirical" / "derived_phi" / "empirical_fit_runs" / "publication_ready_20260805_v4"
DEFAULT_CONTROLS = ROOT / "validation" / "outputs" / "direct_rdoc_negative_controls_publication_20260805_matched_heldout_v4"
DEFAULT_ADAPTIVE = ROOT / "validation" / "outputs" / "direct_rdoc_adaptive_publication_20260805_matched_heldout_v4"
DEFAULT_IRT = ROOT / "validation" / "outputs" / "direct_rdoc_irt_publication_20260805_matched_heldout_v4"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_ancestor(path: Path) -> Path | None:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def validate_benchmark(path: Path) -> dict:
    per_run = pd.read_csv(path / "per_run.csv")
    methods = {
        "ball_student_causal",
        "ball_teacher_smoother",
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
    key = ["cell", "share", "seed", "method"]
    require(set(per_run["method"]) == methods, f"Unexpected benchmark methods: {sorted(set(per_run['method']))}")
    require(len(per_run) == 1300, f"Expected 1,300 benchmark rows, found {len(per_run)}")
    require(not per_run.duplicated(key).any(), "Duplicate benchmark replicate keys")
    counts = per_run.groupby(["cell", "share", "method"]).size()
    require(len(counts) == 130 and counts.eq(10).all(), "Benchmark cells are not complete at ten seeds each")
    for metric in ("latent_rmse", "val_anchor_rmse"):
        values = pd.to_numeric(per_run[metric], errors="coerce").to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"Nonfinite benchmark {metric}")
    for metric in ("beta_cosine", "beta_topk_f1"):
        explicit = per_run[
            ~per_run["method"].isin(
                {"gp_causal_filter", "exponential_decay_gru"}
            )
        ]
        values = pd.to_numeric(explicit[metric], errors="coerce").to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"Nonfinite explicit-transition {metric}")
    aggregate = pd.read_csv(path / "aggregate.csv")
    paired = pd.read_csv(path / "paired_delta_aggregate.csv")
    require(len(aggregate) == 130, f"Expected 130 aggregate rows, found {len(aggregate)}")
    require(len(paired) == 10, f"Expected 10 paired-delta rows, found {len(paired)}")
    for column in (
        "ball_minus_direct_latent_rmse_mean",
        "ball_minus_direct_val_anchor_rmse_mean",
        "ball_minus_gp_causal_latent_rmse_mean",
        "ball_minus_ode_rnn_latent_rmse_mean",
        "ball_minus_s0_latent_rmse_mean",
        "ball_minus_markov_latent_rmse_mean",
    ):
        require(column in paired, f"Missing paired benchmark column {column}")
        require(np.isfinite(pd.to_numeric(paired[column], errors="coerce")).all(), f"Nonfinite {column}")
    student = per_run.loc[
        per_run["method"] == "ball_student_causal",
        ["cell", "share", "seed", "latent_rmse", "val_anchor_rmse"],
    ]
    direct = per_run.loc[
        per_run["method"] == "ball_direct_causal",
        ["cell", "share", "seed", "latent_rmse", "val_anchor_rmse"],
    ]
    distillation = student.merge(
        direct,
        on=["cell", "share", "seed"],
        suffixes=("_ball", "_direct"),
        validate="one_to_one",
    )
    require(len(distillation) == 100, f"Expected 100 matched distillation runs, found {len(distillation)}")
    latent_distillation_difference = float(
        (distillation["latent_rmse_ball"] - distillation["latent_rmse_direct"]).mean()
    )
    anchor_distillation_difference = float(
        (distillation["val_anchor_rmse_ball"] - distillation["val_anchor_rmse_direct"]).mean()
    )
    require(
        abs(latent_distillation_difference - float(paired["ball_minus_direct_latent_rmse_mean"].mean())) <= 1e-10,
        "Stored paired latent distillation difference does not reproduce",
    )
    require(
        abs(anchor_distillation_difference - float(paired["ball_minus_direct_val_anchor_rmse_mean"].mean())) <= 1e-10,
        "Stored paired anchor distillation difference does not reproduce",
    )
    for metric in ("latent_rmse", "val_anchor_rmse", "beta_cosine", "beta_topk_f1"):
        recalculated = (
            per_run.groupby(["cell", "share", "method"], as_index=False)[metric]
            .agg(recomputed_mean="mean", recomputed_sd="std", recomputed_n="size")
        )
        recalculated["recomputed_mcse"] = (
            recalculated["recomputed_sd"] / np.sqrt(recalculated["recomputed_n"])
        )
        compared = aggregate.merge(
            recalculated,
            on=["cell", "share", "method"],
            how="left",
            validate="one_to_one",
        )
        mean_delta = np.abs(
            pd.to_numeric(compared[f"{metric}_mean"], errors="coerce")
            - pd.to_numeric(compared["recomputed_mean"], errors="coerce")
        )
        finite_mean_delta = mean_delta[np.isfinite(mean_delta)]
        require(
            finite_mean_delta.le(1e-10).all(),
            f"Stored aggregate mean does not reproduce for {metric}",
        )
        mcse_delta = np.abs(
            pd.to_numeric(compared[f"{metric}_mcse"], errors="coerce")
            - pd.to_numeric(compared["recomputed_mcse"], errors="coerce")
        )
        finite_mcse_delta = mcse_delta[np.isfinite(mcse_delta)]
        require(
            finite_mcse_delta.le(1e-10).all(),
            f"Stored aggregate Monte Carlo standard error does not reproduce for {metric}",
        )

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    benchmark_ball_snapshot = path / "source_snapshot" / "BALL.py"
    require(
        benchmark_ball_snapshot.exists(),
        "The exact BALL.py benchmark source snapshot is absent",
    )
    require(
        manifest.get("ball_py_sha256") == sha256(benchmark_ball_snapshot),
        "The archived BALL.py source does not match the corrected benchmark provenance",
    )
    benchmark_script_snapshot = path / "source_snapshot" / "direct_rdoc_benchmark.py"
    require(
        benchmark_script_snapshot.exists(),
        "The exact benchmark-script source snapshot is absent",
    )
    require(
        manifest.get("benchmark_py_sha256") == sha256(benchmark_script_snapshot),
        "The archived benchmark script does not match the corrected benchmark provenance",
    )
    fair_snapshot = path / "source_snapshot" / "direct_rdoc_fair_comparator.py"
    require(fair_snapshot.exists(), "The exact classical-comparator source snapshot is absent")
    require(
        manifest.get("fair_comparator_py_sha256") == sha256(fair_snapshot),
        "The archived classical-comparator source does not match the benchmark manifest",
    )
    common_snapshot = path / "source_snapshot" / "direct_rdoc_common.py"
    require(common_snapshot.exists(), "The exact shared benchmark source snapshot is absent")
    require(
        manifest.get("common_py_sha256") == sha256(common_snapshot),
        "The archived shared benchmark source does not match the benchmark manifest",
    )
    harness_snapshot = path / "source_snapshot" / "ball_validation_harness.py"
    require(harness_snapshot.exists(), "The exact simulation harness snapshot is absent")
    require(
        manifest.get("harness_py_sha256") == sha256(harness_snapshot),
        "The archived simulation harness does not match the benchmark manifest",
    )
    expected_args = {
        "n": 150,
        "t": 84,
        "teacher_epochs": 300,
        "student_epochs": 300,
        "ode_epochs": 300,
        "ensemble_size": 5,
        "s0_basis": "matched",
        "anchor_observation": "gaussian",
    }
    require(
        all(manifest.get("args", {}).get(key) == value for key, value in expected_args.items()),
        "Primary simulation manifest does not match the manuscript-grade benchmark specification",
    )
    contract = manifest.get("input_contract", {})
    require(
        {"ball_student_causal", "ball_direct_causal", "gp_causal_filter", "exponential_decay_gru"}.issubset(
            set(contract.get("causal_estimators", []))
        )
        and set(contract.get("retrospective_smoothers", []))
        == {"ball_teacher_smoother", "s0_direct_lgssm", "markov_direct_transition"},
        "Primary simulation information sets are not declared by inferential timing",
    )
    return {
        "rows": len(per_run),
        "methods": sorted(methods),
        "ball_sha": manifest.get("ball_py_sha256"),
        "ball_minus_direct_latent_rmse": latent_distillation_difference,
        "ball_minus_direct_anchor_rmse": anchor_distillation_difference,
    }


def validate_calibration(path: Path) -> dict:
    arms = pd.read_csv(path / "fig3_arms.csv")
    distance = pd.read_csv(path / "fig3_by_anchor_distance.csv")
    missingness = pd.read_csv(path / "fig3_by_daily_missingness.csv")
    expected = {"student raw", "student anchor conformal", "student oracle", "Markov model-based"}
    require(set(arms["arm"]) == expected, f"Unexpected calibration arms: {sorted(set(arms['arm']))}")
    require(arms["replicate"].nunique() == 10 and len(arms) == 40, "Calibration replicate count is incomplete")
    require(
        arms.groupby("replicate")["arm"].nunique().eq(4).all(),
        "Each calibration replicate must contain all four interval arms",
    )
    for frame, name in ((arms, "arms"), (distance, "anchor distance"), (missingness, "daily missingness")):
        informative = frame
        if "n" in informative.columns:
            informative = informative[pd.to_numeric(informative["n"], errors="coerce").fillna(0).gt(0)]
        require(not informative.empty, f"No informative {name} rows")
        for column in ("coverage", "rel_width"):
            values = pd.to_numeric(informative[column], errors="coerce").to_numpy(dtype=float)
            require(np.isfinite(values).all(), f"Nonfinite {name} {column}")
        require(informative["coverage"].between(0, 1).all(), f"Out-of-range {name} coverage")
        require(informative["rel_width"].gt(0).all(), f"Nonpositive {name} width")
    for column in ("observable_anchor_coverage", "observable_anchor_rel_width"):
        require(column in arms, f"Calibration output lacks {column}")
    student_anchor = arms[arms["arm"].isin({"student raw", "student anchor conformal"})]
    require(
        np.isfinite(pd.to_numeric(student_anchor["observable_anchor_coverage"], errors="coerce")).all()
        and student_anchor["observable_anchor_coverage"].between(0, 1).all(),
        "Observable-anchor coverage is incomplete",
    )
    require(
        np.isfinite(pd.to_numeric(student_anchor["observable_anchor_rel_width"], errors="coerce")).all()
        and student_anchor["observable_anchor_rel_width"].gt(0).all(),
        "Observable-anchor width is incomplete",
    )
    conformal = arms[arms["arm"] == "student anchor conformal"]
    conformal_observable_coverage = float(conformal["observable_anchor_coverage"].mean())
    conformal_observable_width = float(conformal["observable_anchor_rel_width"].mean())
    require(
        0.93 <= conformal_observable_coverage <= 0.99,
        "Conformal observable-anchor coverage is not near the 0.95 target",
    )
    require(
        conformal_observable_width < 5.0,
        "Conformal observable-anchor intervals are implausibly wide",
    )
    oracle_latent_coverage = float(
        arms.loc[arms["arm"] == "student oracle", "coverage"].mean()
    )
    require(
        0.92 <= oracle_latent_coverage <= 0.98,
        "Simulation-only latent-oracle intervals are not near the 0.95 target",
    )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    require(
        manifest.get("analysis") == "manuscript-grade uncertainty calibration",
        "Calibration manifest does not identify the manuscript-grade analysis",
    )
    expected_args = {
        "replicates": 10,
        "n": 1000,
        "t": 84,
        "members": 5,
        "teacher_epochs": 300,
        "student_epochs": 300,
        "d_model": 96,
        "n_layers": 3,
        "batch_size": 32,
    }
    require(
        all(manifest.get("args", {}).get(key) == value for key, value in expected_args.items()),
        "Calibration manifest does not match the prespecified manuscript-grade fit",
    )
    require(
        manifest.get("conformal_score")
        == "equal-patient-weight absolute anchor residual divided by predictive SD",
        "Calibration did not use the patient-balanced cluster-weighted normalized conformal score",
    )
    require(
        manifest.get("coverage_target")
        == "patient-balanced marginal observable-measurement coverage",
        "Calibration coverage target is not declared",
    )
    require(
        set(manifest.get("interval_arms", [])) == expected,
        "Calibration manifest interval arms are incomplete",
    )
    source_snapshot = path / "source_snapshot" / "BALL.py"
    require(source_snapshot.exists(), "Calibration source snapshot is absent")
    require(
        manifest.get("ball_py_sha256") == sha256(source_snapshot),
        "Calibration source snapshot does not match the manifest hash",
    )
    return {"rows": len(arms), "ball_sha": manifest.get("ball_py_sha256")}


def _validate_stored_aggregate(
    per_run: pd.DataFrame,
    aggregate: pd.DataFrame,
    group_keys: list[str],
    metrics: tuple[str, ...],
) -> None:
    for metric in metrics:
        recalculated = (
            per_run.groupby(group_keys, as_index=False, dropna=False)[metric]
            .agg(recomputed_mean="mean", recomputed_sd="std", recomputed_n="size")
        )
        recalculated["recomputed_mcse"] = (
            recalculated["recomputed_sd"] / np.sqrt(recalculated["recomputed_n"])
        )
        compared = aggregate.merge(
            recalculated,
            on=group_keys,
            how="left",
            validate="one_to_one",
        )
        for stored_suffix, recomputed_column in (
            ("mean", "recomputed_mean"),
            ("mcse", "recomputed_mcse"),
        ):
            delta = np.abs(
                pd.to_numeric(compared[f"{metric}_{stored_suffix}"], errors="coerce")
                - pd.to_numeric(compared[recomputed_column], errors="coerce")
            )
            finite = delta[np.isfinite(delta)]
            require(
                finite.le(1e-10).all(),
                f"Stored sensitivity {metric} {stored_suffix} does not reproduce",
            )


def validate_sensitivity(
    path: Path,
    *,
    kind: str,
    group_keys: list[str],
    expected_groups: int,
    expected_methods: set[str],
    metrics: tuple[str, ...],
) -> dict:
    per_run = pd.read_csv(path / "per_run.csv")
    aggregate = pd.read_csv(path / "aggregate.csv")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    require(set(per_run["method"]) == expected_methods, f"Unexpected {kind} methods")
    key = [*group_keys[:-1], "seed", "method"]
    require(not per_run.duplicated(key).any(), f"Duplicate {kind} replicate keys")
    counts = per_run.groupby(group_keys, dropna=False).size()
    require(len(counts) == expected_groups, f"Unexpected {kind} group count")
    require(counts.eq(10).all(), f"{kind} groups are not complete at ten seeds")
    require(len(aggregate) == expected_groups, f"Unexpected {kind} aggregate row count")
    for metric in metrics:
        metric_rows = per_run
        if metric in {"beta_cosine", "beta_topk_f1"}:
            metric_rows = metric_rows[
                ~metric_rows["method"].isin(
                    {"gp_causal_filter", "exponential_decay_gru"}
                )
            ]
        values = pd.to_numeric(metric_rows[metric], errors="coerce").to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"Nonfinite {kind} {metric}")
    _validate_stored_aggregate(per_run, aggregate, group_keys, metrics)
    return {
        "rows": len(per_run),
        "groups": len(counts),
        "methods": sorted(expected_methods),
        "source_ball_sha": manifest.get("results_generated_with_ball_py_sha256", manifest.get("ball_py_sha256")),
    }


def validate_empirical(path: Path) -> dict:
    require(
        git_ancestor(path) is None,
        "The complete empirical run must remain outside every Git working tree",
    )
    suffix = "_complete"
    overall = pd.read_csv(path / f"empirical_recovery_overall{suffix}.csv")
    calibration = pd.read_csv(path / f"empirical_calibration{suffix}.csv")
    bootstrap = pd.read_csv(path / f"empirical_bootstrap_diff{suffix}.csv")
    transition = pd.read_csv(path / f"empirical_rdoc_transition{suffix}.csv")
    transition_sensitivity = pd.read_csv(
        path / f"empirical_rdoc_transition_full_record_sensitivity{suffix}.csv"
    )
    by_instrument = pd.read_csv(path / "empirical_recovery_by_instrument_complete.csv")
    by_subgroup = pd.read_csv(path / "empirical_recovery_by_subgroup_complete.csv")
    conditional = pd.read_csv(path / "empirical_calibration_conditional_complete.csv")
    generalization = pd.read_csv(path / "empirical_generalization_complete.csv")
    burden = pd.read_csv(path / "empirical_assessment_burden_complete.csv")
    workload = pd.read_csv(path / "empirical_workload_classification_complete.csv")
    pairwise = pd.read_csv(path / "empirical_pairwise_bootstrap_complete.csv")
    context = pd.read_csv(path / "empirical_recovery_by_context_complete.csv")
    uncertainty_workload = pd.read_csv(path / "empirical_uncertainty_workload_complete.csv")
    input_contract = pd.read_csv(path / "empirical_input_contract_complete.csv")
    leakage_audit = pd.read_csv(path / "empirical_leakage_audit_complete.csv")
    cohort = pd.read_csv(path / "empirical_cohort_characteristics_complete.csv")
    timing = pd.read_csv(path / "empirical_fit_timing_complete.csv")
    ball_transitions = pd.read_csv(path / "empirical_ball_transition_rows_complete.csv")
    rdoc_transition_rows = pd.read_csv(path / "empirical_rdoc_transition_rows_complete.csv")
    measurement_calibration = json.loads(
        (path / "measurement_calibration.json").read_text(encoding="utf-8")
    )
    hyperparameters = pd.read_csv(path / "empirical_hyperparameter_selection_complete.csv")
    manifest = json.loads((path / "fit_run_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("status") == "complete", "Empirical fit manifest is not complete")
    require(manifest.get("completed_cadences") == [14, 21, 28], "Empirical cadences are incomplete")
    source_snapshot = path / str(manifest.get("source_snapshot", ""))
    require(source_snapshot.is_file(), "Empirical BALL.py source snapshot is absent")
    require(
        manifest.get("ball_py_sha256") == sha256(source_snapshot),
        "Empirical BALL.py source snapshot does not match the fit manifest",
    )
    cadence_manifests = [
        json.loads((path / f"empirical_teacher_student_manifest_{cadence}d.json").read_text(encoding="utf-8"))
        for cadence in (14, 21, 28)
    ]
    require(
        all(item.get("same_session_cross_instrument_questionnaires_in_causal_encoder") is False for item in cadence_manifests)
        and all(item.get("questionnaire_event_accounting_identity_passed") is True for item in cadence_manifests),
        "Prospective questionnaire timing or event accounting failed",
    )
    require(manifest.get("canonical_clinic_count") == 19, "Empirical canonical clinic count is not 19")
    require(manifest.get("source_network_clinic_count") == 25, "Source network clinic count is not 25")
    require(manifest.get("observed_facility_string_count") == 23, "Observed facility-string count is not 23")
    require(
        manifest.get("source_patient_partitions") == {"train": 914, "val": 325, "test": 370},
        "Source patient partitions changed",
    )
    require(
        manifest.get("source_clinic_partitions") == {"train": 10, "val": 3, "test": 6},
        "Source clinic partitions changed",
    )
    require(manifest.get("multiple_clinic_patients") == 203, "Multiple-clinic patient count changed")
    require(manifest.get("cross_partition_patients_excluded") == 94, "Cross-partition exclusion count is not 94")
    require(
        manifest.get("clinic_exclusive_patient_partitions")
        == {"train": 872, "val": 298, "test": 345},
        "Clinic-exclusive patient partitions changed",
    )
    require(manifest.get("clinic_exclusive_patient_count") == 1515, "Clinic-exclusive patient count is not 1,515")
    require(manifest.get("clinic_exclusive_session_count") == 52493, "Clinic-exclusive session count is not 52,493")
    require(
        manifest.get("primary_model_partitions") == {"development": 1170, "heldout_clinic": 345},
        "Primary model partitions changed",
    )
    require(
        manifest.get("primary_interval_partitions") == {"calibration": 172, "test": 173},
        "Primary interval allocations changed",
    )
    require(
        manifest.get("comorbidity_feature_rows") == 52493
        and manifest.get("comorbidity_patients") == 1515,
        "Comorbidity manifest is not restricted to the clinic-exclusive analytic cohort",
    )
    require(
        manifest.get("comorbidity_source") == "protected external numeric feature table"
        and manifest.get("comorbidity_strictly_prior") is True,
        "Comorbidity provenance does not certify strictly prior external numeric features",
    )
    require(
        manifest.get("zero_day_adjacent_session_pairs") == 3648
        and manifest.get("elapsed_day_preprocessing")
        == "raw nonnegative calendar-day gaps with no division, binning, or clipping"
        and manifest.get("same_day_transition_policy")
        == "zero-day pairs are excluded from the dynamics likelihood because reliable within-day ordering is unavailable",
        "Elapsed-day handling is not the prespecified unclipped calendar-time analysis",
    )
    require(
        manifest.get("comorbidity_prior_list_sessions") == 29874
        and manifest.get("comorbidity_patients_with_prior_list") == 949,
        "Strictly prior diagnosis-history availability counts changed",
    )
    expected_comorbidity_prevalence = {
        "dx_depression": {"sessions": 29211, "patients": 924},
        "dx_anxiety": {"sessions": 14521, "patients": 454},
        "dx_ptsd": {"sessions": 3168, "patients": 100},
        "dx_bipolar": {"sessions": 229, "patients": 9},
        "dx_ocd": {"sessions": 1371, "patients": 47},
        "dx_adhd": {"sessions": 2484, "patients": 74},
        "dx_psychotic_spectrum": {"sessions": 80, "patients": 3},
        "dx_substance_use": {"sessions": 430, "patients": 18},
        "dx_eating_disorder": {"sessions": 119, "patients": 5},
        "dx_personality_disorder": {"sessions": 360, "patients": 13},
        "dx_autism_spectrum": {"sessions": 189, "patients": 5},
        "dx_sleep_disorder": {"sessions": 1312, "patients": 41},
    }
    require(
        manifest.get("comorbidity_prevalence") == expected_comorbidity_prevalence,
        "Strictly prior diagnosis-category prevalence changed",
    )
    require(
        manifest.get("interval_clinic_partitions") == {"calibration": 6, "test": 6},
        "Both interval roles do not cover all six held-out clinics",
    )
    require(set(overall["cadence"]) == {14, 21, 28}, "Unexpected empirical cadences")
    required_methods = {
        "ball",
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
    require(
        set(overall["method"]) == required_methods,
        f"Unexpected empirical method set: {sorted(set(overall['method']))}",
    )
    require(np.isfinite(pd.to_numeric(overall["rmse"], errors="coerce")).all(), "Nonfinite empirical RMSE")
    require(
        {"rmse_raw", "rmse_support_clipped"}.issubset(overall.columns)
        and np.isfinite(overall[["rmse_raw", "rmse_support_clipped"]].to_numpy(dtype=float)).all(),
        "Raw and support-clipped empirical point-prediction errors are incomplete",
    )
    require(
        set(calibration["cadence"]) == {14, 21, 28}
        and set(calibration["instrument"].astype(str)) == {"PHQ9", "BDI", "GAD7"},
        "Instrument-specific calibration rows are incomplete",
    )
    require(calibration["coverage"].between(0, 1).all(), "Out-of-range empirical coverage")
    require(calibration["mean_width_native_clipped"].gt(0).all(), "Nonpositive empirical interval width")
    require(calibration["mean_width_fraction_of_legal_range"].between(0, 1).all(), "Support-clipped interval width is invalid")
    require(
        set(calibration["score_unit"].astype(str))
        == {"patient-balanced absolute residual divided by BALL predictive standard deviation"}
        and set(calibration["coverage_target"].astype(str))
        == {"patient-balanced marginal observable-measurement coverage"},
        "Empirical calibration is not the patient-balanced cluster-weighted procedure",
    )
    require(set(transition["cadence"]) == {14, 21, 28} and len(transition) == 6, "Empirical RDoC transition summary is incomplete")
    require(
        {"permutation_p", "permutation_p_holm"}.issubset(transition.columns)
        and transition[["permutation_p", "permutation_p_holm"]].apply(
            pd.to_numeric, errors="coerce"
        ).notna().all().all()
        and transition["permutation_p_holm"].between(0, 1).all()
        and (transition["permutation_p_holm"] >= transition["permutation_p"]).all(),
        "The six empirical RDoC joint tests lack valid Holm-adjusted permutation probabilities",
    )
    require(
        set(transition["patient_mean_mode"].astype(str)) == {"expanding_strictly_earlier"}
        and not transition["current_profile_in_reference_mean"].astype(bool).any(),
        "Primary RDoC analysis does not use the strictly earlier expanding patient mean",
    )
    require(
        len(transition_sensitivity) == 6
        and set(transition_sensitivity["patient_mean_mode"].astype(str)) == {"full_record_retrospective"},
        "Retrospective full-record RDoC sensitivity is incomplete",
    )
    require(rdoc_transition_rows["id"].nunique() <= 345, "RDoC analysis exceeds the held-out-clinic population")
    require(
        {
            "patients_with_any_profile",
            "patients_with_at_least_two_profiles",
            "eligible_transitions_after_prior_profile_requirement",
        }.issubset(transition.columns)
        and (transition["patients_with_at_least_two_profiles"] <= transition["patients_with_any_profile"]).all(),
        "RDoC eligibility accounting is incomplete",
    )
    require("stratum_type" in by_subgroup and "clinic" in set(by_subgroup["stratum_type"]), "Empirical subgroup output lacks clinic strata")
    require("stratum_type" in conditional and "clinic" in set(conditional["stratum_type"]), "Conditional calibration lacks clinic strata")
    require(len(by_instrument) > 0 and len(bootstrap) == 3 and len(generalization) > 0, "Empirical sensitivity outputs are incomplete")
    require(
        set(input_contract["method"].astype(str)) == required_methods,
        "Empirical input contract does not cover every fitted method",
    )
    expected_pairwise_counts = {14: 18, 21: 12, 28: 12}
    pairwise_counts = pairwise.groupby(pairwise["cadence"].astype(int)).size().to_dict()
    require(
        pairwise_counts == expected_pairwise_counts,
        f"Patient-clustered comparisons are incomplete: {pairwise_counts}",
    )
    require(
        set(context["stratum_type"].astype(str))
        == {"cold_start", "prior_anchor_count", "severity", "treatment_week"},
        "Empirical context evaluation is incomplete",
    )
    require(
        set(uncertainty_workload["cadence"].astype(int)) == {14, 21, 28}
        and set(uncertainty_workload["interval_width_fraction_threshold"].astype(float))
        == {0.1, 0.2, 0.3, 0.4, 0.5}
        and set(uncertainty_workload["width_type"].astype(str)) == {"raw", "support_clipped"}
        and len(uncertainty_workload) == 30
        and uncertainty_workload.groupby(
            ["cadence", "interval_width_fraction_threshold"], sort=False
        )["width_type"].nunique().eq(2).all(),
        "Uncertainty-guided measurement analysis is incomplete",
    )
    require(
        set(leakage_audit["status"].astype(str)).issubset({"pass", "excluded"})
        and len(leakage_audit) >= 10,
        "Empirical leakage audit is incomplete",
    )
    require(
        {"Patients", "Treatment sessions", "Clinics"}.issubset(
            set(cohort["characteristic"].astype(str))
        ),
        "Empirical cohort summary is incomplete",
    )
    require(
        set(timing["cadence"].astype(int)) == {14, 21, 28}
        and {"teacher_student_ensemble", "direct_causal_ensemble", "gaussian_process", "ode_rnn_ensemble"}.issubset(
            set(timing["stage"].astype(str))
        ),
        "Empirical compute-time audit is incomplete",
    )
    require(
        set(generalization["validation"].astype(str)) == {"forward_2025"},
        "Generalization output is not the prespecified forward-2025 sensitivity",
    )
    require(
        generalization["n_train_source_eligible"].astype(int).gt(0).all()
        and generalization["n_test_source_eligible"].astype(int).gt(0).all()
        and generalization["spanning_patients_excluded"].astype(int).gt(0).all()
        and generalization["n_test_model_patients"].astype(int).ge(generalization["n_test_patients"].astype(int)).all()
        and generalization["n_temporal_calibration_patients"].astype(int).gt(0).all()
        and generalization["n_temporal_final_test_patients"].astype(int).gt(0).all(),
        "Strict temporal development, calibration, or final-test cohorts are invalid",
    )
    require(
        set(generalization["training_time_rule"].astype(str))
        == {"last observed session before 2025-01-01"}
        and set(generalization["test_time_rule"].astype(str))
        == {"first observed session on or after 2025-01-01"},
        "Strict temporal availability rules changed",
    )
    temporal_ball = generalization[generalization["method"].astype(str).eq("ball")]
    require(
        len(temporal_ball) == 1
        and temporal_ball["patient_balanced_conformal_coverage"].between(0, 1).all()
        and temporal_ball["mean_interval_width_fraction_of_legal_range"].between(0, 1).all(),
        "Temporal conformal calibration is incomplete",
    )
    require(
        set(calibration["n_test_patients"].astype(int)) == {173}
        and set(calibration["n_calibration_patients"].astype(int)) == {171},
        "Actual anchor-bearing calibration/test patient counts changed",
    )
    all_burden = burden[burden["instrument"].astype(str) == "all"].sort_values("cadence")
    require(
        all_burden[["cadence", "available_assessments", "retained_assessments"]]
        .astype(int)
        .to_records(index=False)
        .tolist()
        == [(14, 122791, 23611), (21, 122791, 17336), (28, 122791, 13839)],
        "Assessment-burden counts changed",
    )
    require(set(workload["cadence"].astype(int)) == {14, 21, 28}, "Workload classification cadences are incomplete")
    require(set(workload["instrument"].astype(str)) == {"PHQ9", "GAD7", "BDI"}, "Workload classification instruments are incomplete")
    require(set(workload["outcome"].astype(str)) == {"response", "remission"}, "Workload classification outcomes are incomplete")
    require(
        set(ball_transitions["cadence"].astype(int)) == {14, 21, 28}
        and set(ball_transitions["channel"].astype(str)) == {"depression", "anxiety"},
        "All-session BALL transition export is incomplete",
    )
    require(
        ball_transitions["dt"].gt(0).all()
        and ball_transitions["id"].nunique() == 344
        and set(ball_transitions["selection"].astype(str))
        == {"all positive-gap adjacent modeled sessions in clinic-held-out patients"},
        "All-session BALL transition export has an invalid scope",
    )
    require(
        not any(str(column).startswith("B") for column in ball_transitions.columns)
        and len(ball_transitions) > len(rdoc_transition_rows),
        "BALL face-validity source remains selected on RDoC availability",
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
        values = pd.to_numeric(workload[column], errors="coerce")
        require(values.dropna().between(0, 1).all(), f"Out-of-range workload metric {column}")
    require(
        (
            workload["patient_balanced_accuracy_ci_low"]
            <= workload["patient_balanced_accuracy"]
        ).all()
        and (
            workload["patient_balanced_accuracy"]
            <= workload["patient_balanced_accuracy_ci_high"]
        ).all(),
        "Workload accuracy lies outside its patient-clustered interval",
    )
    require(
        set(workload["accuracy_ci_method"].astype(str))
        == {"patient-clustered percentile bootstrap"}
        and set(workload["accuracy_ci_replicates"].astype(int)) == {2000},
        "Workload classification uncertainty is not the prespecified patient-clustered bootstrap",
    )
    require(
        set(measurement_calibration["instruments"]) == {"PHQ9", "BDI", "GAD7"}
        and measurement_calibration["instruments"]["PHQ9"]["intercept"] == 0.0
        and measurement_calibration["instruments"]["PHQ9"]["loading"] == 1.0
        and measurement_calibration["instruments"]["GAD7"]["intercept"] == 0.0
        and measurement_calibration["instruments"]["GAD7"]["loading"] == 1.0,
        "Frozen development measurement calibration violates its identification constraints",
    )
    require(
        measurement_calibration.get("fit_scope")
        == "all genuine questionnaire events from the ten clinic-exclusive training clinics before artificial sparsification"
        and measurement_calibration.get("exported_parameters")
        == "instrument intercepts, loadings, and observation standard deviations only"
        and "latent_process" not in measurement_calibration
        and set(measurement_calibration["development_data"]["normalized_score_mean_by_instrument"])
        == {"PHQ9", "BDI", "GAD7"},
        "Measurement calibration source or training means are incomplete",
    )
    require(
        set(hyperparameters["daily_persistence"].dropna().astype(float)).issuperset({0.1, 0.3, 0.6, 0.9})
        and set(hyperparameters["questionnaire_weight"].dropna().astype(float)).issuperset({10.0, 20.0, 40.0, 80.0})
        and len(hyperparameters) == 16
        and set(hyperparameters["selection_stage"].astype(str)) == {"joint_grid"},
        "Development-only core hyperparameter selection is incomplete",
    )

    expected_features = [
        "treatment_session_number",
        "dx_comorbidity_count",
        "dx_depression",
        "dx_anxiety",
        "dx_ptsd",
        "dx_bipolar",
        "dx_ocd",
        "dx_adhd",
        "dx_psychotic_spectrum",
        "dx_substance_use",
        "dx_eating_disorder",
        "dx_personality_disorder",
        "dx_autism_spectrum",
        "dx_sleep_disorder",
    ]
    for cadence in (14, 21, 28):
        cadence_manifest = json.loads(
            (path / f"empirical_teacher_student_manifest_{cadence}d.json").read_text(
                encoding="utf-8"
            )
        )
        require(cadence_manifest.get("n_patients") == 1512, f"{cadence}d tensor patient count changed")
        require(cadence_manifest.get("n_development_patients") == 1168, f"{cadence}d development count changed")
        require(cadence_manifest.get("n_heldout_clinic_patients") == 344, f"{cadence}d held-out count changed")
        require(cadence_manifest.get("empirical_feature_names") == expected_features, f"{cadence}d feature schema changed")
        require(cadence_manifest.get("uses_rdoc_proxy") is False, f"{cadence}d primary tensor uses RDoC")
        require(
            cadence_manifest.get("anchor_representation")
            == "one event per patient, session, and questionnaire instrument"
            and cadence_manifest.get("questionnaire_event_collisions") == 0,
            f"{cadence}d questionnaire event contract failed",
        )
        require(
            set(cadence_manifest.get("anchor_instrument_names", [])) == {"PHQ9", "BDI", "GAD7"},
            f"{cadence}d instrument-specific anchor tensor is incomplete",
        )
        require(cadence_manifest.get("same_session_narrative_inputs") is False, f"{cadence}d primary tensor uses narrative")
        require(cadence_manifest.get("target_fields_in_structured_feature_allowlist") is False, f"{cadence}d target field entered allowlist")
        require(cadence_manifest.get("action_vocabulary_scope") == "development partition only", f"{cadence}d action vocabulary scope changed")
        require(
            cadence_manifest.get("same_day_transition_count") == 3647
            and cadence_manifest.get("elapsed_day_preprocessing")
            == "raw nonnegative calendar-day gaps with no division, binning, or clipping",
            f"{cadence}d elapsed-day handling changed",
        )
    return {
        "overall_rows": len(overall),
        "test_patients": manifest.get("primary_interval_partitions", {}).get("test"),
        "clinics": manifest.get("canonical_clinic_count"),
        "workload_rows": len(workload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manuscript-grade BALL outputs.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--empirical", type=Path, default=DEFAULT_EMPIRICAL)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--adaptive", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--irt", type=Path, default=DEFAULT_IRT)
    args = parser.parse_args()
    result = {
        "benchmark": validate_benchmark(args.benchmark.resolve()),
        "negative_controls": validate_sensitivity(
            args.controls.resolve(),
            kind="negative control",
            group_keys=["control", "share", "method"],
            expected_groups=20,
            expected_methods={
                "ball_student_causal",
                "ball_teacher_smoother",
                "ball_direct_causal",
                "s0_direct_lgssm",
                "markov_direct_transition",
            },
            metrics=("latent_rmse", "val_anchor_rmse", "beta_cosine", "beta_topk_f1"),
        ),
        "adaptive_lasso": validate_sensitivity(
            args.adaptive.resolve(),
            kind="adaptive Lasso",
            group_keys=["cell", "share", "method"],
            expected_groups=110,
            expected_methods={
                "ball_student_causal",
                "ball_teacher_smoother",
                "ball_direct_causal",
                "ball_direct_causal_compute_matched",
                "ball_inherited_dynamics_only",
                "ball_teacher_matching_only",
                "ball_full_decomposition",
                "ball_student_transition_only",
                "ball_student_transition_only_strict",
                "s0_direct_lgssm",
                "markov_direct_transition",
            },
            metrics=("latent_rmse", "val_anchor_rmse", "beta_cosine", "beta_topk_f1"),
        ),
        "irt": validate_sensitivity(
            args.irt.resolve(),
            kind="IRT",
            group_keys=["cell", "share", "method"],
            expected_groups=130,
            expected_methods={
                "ball_student_causal",
                "ball_teacher_smoother",
                "ball_direct_causal",
                "ball_direct_causal_compute_matched",
                "ball_inherited_dynamics_only",
                "ball_teacher_matching_only",
                "ball_full_decomposition",
                "ball_student_transition_only",
                "ball_student_transition_only_strict",
                "gp_causal_filter",
                "exponential_decay_gru",
                "s0_direct_lgssm",
                "markov_direct_transition",
            },
            metrics=("latent_rmse_trait", "irt_item_nll", "irt_expected_total_rmse", "beta_cosine", "beta_topk_f1"),
        ),
        "calibration": validate_calibration(args.calibration.resolve()),
        "empirical": validate_empirical(args.empirical.resolve()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
