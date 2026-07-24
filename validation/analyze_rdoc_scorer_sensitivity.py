from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "RDoC_LLM_scorer.csv"
SESSIONS = ROOT / "empirical" / "data" / "rtms_paper_analytic_sessions.csv"
TRANSITIONS = ROOT / "empirical" / "derived" / "empirical_rdoc_transition_rows.csv"
ENSEMBLE_SUMMARY = ROOT / "empirical" / "derived" / "empirical_rdoc_transition.csv"
ENSEMBLE_COEFS = ROOT / "empirical" / "derived" / "empirical_rdoc_transition_coefficients.csv"
OUT = ROOT / "validation" / "outputs" / "rdoc_scorer_sensitivity_20260716"

DOMAIN_SOURCES = {
    "negative valence": (
        "negative_valence_systems_gemma-4-31B-it",
        "negative_valence_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
    "positive valence": (
        "positive_valence_systems_gemma-4-31B-it",
        "positive_valence_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
    "cognitive systems": (
        "cognitive_systems_gemma-4-31B-it",
        "cognitive_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
    "social processes": (
        "social_processes_gemma-4-31B-it",
        "social_processes_Qwen3-Next-80B-A3B-Instruct",
    ),
    "arousal/regulatory": (
        "arousal_and_regulatory_systems_gemma-4-31B-it",
        "arousal_and_regulatory_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
    "sensorimotor": (
        "sensorimotor_systems_gemma-4-31B-it",
        "sensorimotor_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
}
DOMAIN_NAMES = list(DOMAIN_SOURCES)
B_COLS = [f"B{i}" for i in range(6)]
NUISANCE = ["L_current", "dt", "rdoc_days_stale"]
RIDGE = 1.0
FOLDS = 5
PERMUTATIONS = 200

CLINICAL_BANDS = {
    "PHQ9": [
        ("Minimal/none", 0, 4),
        ("Mild", 5, 9),
        ("Moderate", 10, 14),
        ("Moderately severe", 15, 19),
        ("Severe", 20, 27),
    ],
    "BDI": [
        ("Minimal/none", 0, 13),
        ("Mild", 14, 19),
        ("Moderate", 20, 28),
        ("Severe", 29, 63),
    ],
    "GAD7": [
        ("Minimal/none", 0, 4),
        ("Mild", 5, 9),
        ("Moderate", 10, 14),
        ("Severe", 15, 21),
    ],
}


def _norm_field_name(value: object) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return value


def load_retained_raw_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = pd.read_csv(SESSIONS, low_memory=False)
    sessions["ServiceDate"] = pd.to_datetime(sessions["ServiceDate"], errors="coerce")
    sessions = sessions.dropna(subset=["PatientFID", "ServiceDate"]).copy()
    sessions["PatientFID"] = pd.to_numeric(sessions["PatientFID"], errors="coerce")
    sessions = sessions.dropna(subset=["PatientFID"]).copy()
    sessions["PatientFID"] = sessions["PatientFID"].astype(int)
    patients = set(sessions["PatientFID"].unique())
    first_session = sessions.groupby("PatientFID")["ServiceDate"].min()

    model_cols = [column for pair in DOMAIN_SOURCES.values() for column in pair]
    usecols = ["FieldValue_UID", "FieldName", "PatientFID", "ServiceDate"] + model_cols
    kept: list[pd.DataFrame] = []
    seen: set[object] = set()
    for chunk in pd.read_csv(RAW, usecols=usecols, chunksize=100_000, low_memory=False):
        chunk["PatientFID"] = pd.to_numeric(chunk["PatientFID"], errors="coerce")
        chunk = chunk[chunk["PatientFID"].isin(patients)].copy()
        if chunk.empty:
            continue
        normalized = chunk["FieldName"].map(_norm_field_name)
        excluded = normalized.isin({"dx", "dx_diagnosis"}) | normalized.str.contains("diagnosis", na=False)
        chunk = chunk.loc[~excluded].copy()
        chunk = chunk.loc[~chunk["FieldValue_UID"].isin(seen)].copy()
        seen.update(chunk["FieldValue_UID"].dropna().tolist())
        kept.append(chunk)

    raw = pd.concat(kept, ignore_index=True)
    raw["PatientFID"] = raw["PatientFID"].astype(int)
    raw["ServiceDate"] = pd.to_datetime(raw["ServiceDate"], errors="coerce")
    raw = raw.dropna(subset=["ServiceDate"]).copy()
    raw["first_session_date"] = raw["PatientFID"].map(first_session)
    raw["day"] = (raw["ServiceDate"] - raw["first_session_date"]).dt.days.astype(int)
    for column in model_cols:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").clip(0.0, 1.0)
    raw = raw.dropna(subset=model_cols).copy()

    session_axis = sessions.sort_values(["PatientFID", "ServiceDate"]).copy()
    session_axis["session"] = session_axis.groupby("PatientFID").cumcount()
    first = session_axis.groupby("PatientFID")["ServiceDate"].transform("min")
    session_axis["day"] = (session_axis["ServiceDate"] - first).dt.days.astype(int)
    session_axis = session_axis[["PatientFID", "session", "day"]].rename(columns={"PatientFID": "id"})
    return raw, session_axis


def scorer_agreement(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agreement_rows: list[dict] = []
    distribution_rows: list[dict] = []
    for domain, (gemma_col, qwen_col) in DOMAIN_SOURCES.items():
        pair = raw[[gemma_col, qwen_col]].dropna()
        gemma = pair[gemma_col].to_numpy(dtype=float)
        qwen = pair[qwen_col].to_numpy(dtype=float)
        n = len(pair)
        pearson = float(pearsonr(gemma, qwen).statistic)
        spearman = float(spearmanr(gemma, qwen).statistic)
        fisher_z = np.arctanh(np.clip(pearson, -0.999999, 0.999999))
        fisher_se = 1.0 / math.sqrt(max(n - 3, 1))
        pearson_lo, pearson_hi = np.tanh([fisher_z - 1.96 * fisher_se, fisher_z + 1.96 * fisher_se])
        weighted_kappa = float(
            cohen_kappa_score((gemma * 2).round().astype(int), (qwen * 2).round().astype(int), weights="quadratic")
        )
        agreement_rows.append(
            {
                "domain": domain,
                "n": n,
                "pearson_r": pearson,
                "pearson_ci_low": float(pearson_lo),
                "pearson_ci_high": float(pearson_hi),
                "spearman_rho": spearman,
                "exact_agreement": float(np.mean(gemma == qwen)),
                "quadratic_weighted_kappa": weighted_kappa,
                "mean_absolute_difference": float(np.mean(np.abs(gemma - qwen))),
            }
        )
        ensemble = (gemma + qwen) / 2.0
        for score in (0.0, 0.25, 0.5, 0.75, 1.0):
            distribution_rows.append(
                {
                    "domain": domain,
                    "ensemble_score": score,
                    "n": int(np.sum(ensemble == score)),
                    "proportion": float(np.mean(ensemble == score)),
                }
            )
    return pd.DataFrame(agreement_rows), pd.DataFrame(distribution_rows)


def clinical_anchor_crosswalk() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_rows: list[dict] = []
    crosswalk_rows: list[dict] = []
    stats_by_cadence: dict[int, dict[str, tuple[float, float]]] = {}
    for cadence in (14, 21, 28):
        anchors = pd.read_csv(ROOT / "empirical" / "derived" / f"anchors_sparse_{cadence}d.csv")
        retained = anchors[anchors["role"] == "anchor"].copy()
        stats_by_cadence[cadence] = {}
        for scale, values in retained.groupby("anchor")["value"]:
            array = values.to_numpy(dtype=float)
            mean = float(np.mean(array))
            sd = float(np.std(array)) or 1.0
            stats_by_cadence[cadence][str(scale)] = (mean, sd)
            stats_rows.append(
                {
                    "cadence_days": cadence,
                    "instrument": str(scale),
                    "retained_anchors": int(len(array)),
                    "mean": mean,
                    "sd": sd,
                }
            )

    for instrument, bands in CLINICAL_BANDS.items():
        channel = "Depression" if instrument in {"PHQ9", "BDI"} else "Anxiety"
        for severity, raw_low, raw_high in bands:
            row = {
                "latent_channel": channel,
                "severity_label": severity,
                "instrument": instrument,
                "raw_score_range": f"{raw_low}-{raw_high}",
            }
            for cadence in (14, 21, 28):
                mean, sd = stats_by_cadence[cadence][instrument]
                lower = (raw_low - mean) / sd
                upper = (raw_high - mean) / sd
                row[f"latent_z_range_{cadence}d"] = f"{lower:.2f} to {upper:.2f}"
            crosswalk_rows.append(row)
    return pd.DataFrame(crosswalk_rows), pd.DataFrame(stats_rows)


def build_session_proxy(raw: pd.DataFrame, session_axis: pd.DataFrame, scorer: str) -> pd.DataFrame:
    index = 0 if scorer == "Gemma" else 1
    score_frame = pd.DataFrame({"id": raw["PatientFID"].astype(int), "day": raw["day"].astype(int)})
    for j, (_, source_pair) in enumerate(DOMAIN_SOURCES.items()):
        values = raw[source_pair[index]].astype(float)
        score_frame[f"B{j}"] = (values - values.mean()) / (values.std() or 1.0)
    notes = score_frame.groupby(["id", "day"], as_index=False)[B_COLS].mean()
    notes["note_day"] = notes["day"]
    notes = notes.sort_values("day").reset_index(drop=True)
    sessions = session_axis.sort_values("day").reset_index(drop=True)
    merged = pd.merge_asof(sessions, notes, on="day", by="id", direction="backward")
    merged["rdoc_observed"] = merged["note_day"].notna()
    merged["rdoc_days_stale"] = merged["day"] - merged["note_day"]
    for column in B_COLS:
        merged[column] = merged[column].fillna(0.0)
    return merged[["id", "session", "day", "rdoc_observed", "rdoc_days_stale", *B_COLS]]


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    train_fill = np.where(np.isfinite(train), train, mean)
    test_fill = np.where(np.isfinite(test), test, mean)
    sd = np.nanstd(train_fill, axis=0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    return (train_fill - mean) / sd, (test_fill - mean) / sd


def _ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_std, test_std = _standardize(train_x, test_x)
    train_design = np.column_stack([np.ones(len(train_std)), train_std])
    test_design = np.column_stack([np.ones(len(test_std)), test_std])
    penalty = np.eye(train_design.shape[1])
    penalty[0, 0] = 0.0
    lhs = train_design.T @ train_design + RIDGE * penalty
    rhs = train_design.T @ train_y
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(lhs) @ rhs
    return test_design @ coef, coef


def _folds(ids: np.ndarray) -> list[np.ndarray]:
    unique = np.asarray(sorted(np.unique(ids)), dtype=int)
    rng = np.random.default_rng(29)
    rng.shuffle(unique)
    return [fold for fold in np.array_split(unique, min(FOLDS, len(unique))) if len(fold)]


def _cv_error(ids: np.ndarray, y: np.ndarray, x: np.ndarray, folds: list[np.ndarray]) -> tuple[float, float]:
    outcomes: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for fold in folds:
        test = np.isin(ids, fold)
        train = ~test
        pred, _ = _ridge_fit_predict(x[train], y[train], x[test])
        outcomes.append(y[test])
        predictions.append(pred)
    observed = np.concatenate(outcomes)
    predicted = np.concatenate(predictions)
    residual = observed - predicted
    return float(np.sqrt(np.mean(residual**2))), float(np.sum(residual**2))


def transition_sensitivity(
    transitions: pd.DataFrame,
    profile: pd.DataFrame,
    scorer: str,
    ensemble_coef: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_day = profile.sort_values("session").drop_duplicates(["id", "day"], keep="last")
    renamed = profile_day[["id", "day", "rdoc_observed", "rdoc_days_stale", *B_COLS]].rename(
        columns={"rdoc_days_stale": "rdoc_days_stale_new", **{column: f"{column}_new" for column in B_COLS}}
    )
    merged = transitions.drop(columns=[*B_COLS]).merge(renamed, on=["id", "day"], how="left")
    merged = merged[merged["rdoc_observed"].fillna(False)].copy()
    merged["rdoc_days_stale"] = merged["rdoc_days_stale_new"]
    for column in B_COLS:
        merged[column] = merged[f"{column}_new"]

    summaries: list[dict] = []
    coefficients: list[dict] = []
    for (cadence, channel), frame in merged.groupby(["cadence", "channel"], sort=True):
        frame = frame.dropna(subset=["delta_per_day", *NUISANCE, *B_COLS]).copy()
        ids = frame["id"].to_numpy(dtype=int)
        y = frame["delta_per_day"].to_numpy(dtype=float)
        nuisance = frame[NUISANCE].to_numpy(dtype=float)
        rdoc = frame[B_COLS].to_numpy(dtype=float)
        folds = _folds(ids)
        base_rmse, base_sse = _cv_error(ids, y, nuisance, folds)
        full_x = np.column_stack([nuisance, rdoc])
        full_rmse, full_sse = _cv_error(ids, y, full_x, folds)
        improvement = base_rmse - full_rmse
        incremental_r2 = 1.0 - full_sse / base_sse
        _, coef = _ridge_fit_predict(full_x, y, full_x)
        beta = coef[1 + len(NUISANCE) :]

        ensemble_group = ensemble_coef[
            (ensemble_coef["cadence"] == int(cadence)) & (ensemble_coef["channel"] == str(channel))
        ]
        ensemble_beta = np.array(
            [
                float(ensemble_group.loc[ensemble_group["domain"] == column, "coefficient"].iloc[0])
                for column in B_COLS
            ],
            dtype=float,
        )
        denominator = np.linalg.norm(beta) * np.linalg.norm(ensemble_beta)
        cosine = float(np.dot(beta, ensemble_beta) / denominator) if denominator > 0 else float("nan")

        rng = np.random.default_rng(1100 + int(cadence) + (0 if channel == "depression" else 100))
        perm_better = 0
        for _ in range(PERMUTATIONS):
            permuted = rdoc[rng.permutation(len(rdoc))]
            perm_rmse, _ = _cv_error(ids, y, np.column_stack([nuisance, permuted]), folds)
            if base_rmse - perm_rmse >= improvement:
                perm_better += 1
        permutation_p = (perm_better + 1.0) / (PERMUTATIONS + 1.0)

        summaries.append(
            {
                "scorer": scorer,
                "cadence": int(cadence),
                "channel": str(channel),
                "n": int(len(frame)),
                "n_patients": int(frame["id"].nunique()),
                "rmse_nuisance": base_rmse,
                "rmse_rdoc": full_rmse,
                "rmse_improvement": improvement,
                "incremental_r2": incremental_r2,
                "beta_norm": float(np.linalg.norm(beta)),
                "top_domain": DOMAIN_NAMES[int(np.argmax(np.abs(beta)))],
                "beta_cosine_vs_ensemble": cosine,
                "permutation_p": permutation_p,
            }
        )
        coefficients.extend(
            {
                "scorer": scorer,
                "cadence": int(cadence),
                "channel": str(channel),
                "domain": DOMAIN_NAMES[j],
                "coefficient": float(beta[j]),
            }
            for j in range(6)
        )
    return pd.DataFrame(summaries), pd.DataFrame(coefficients)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, session_axis = load_retained_raw_rows()
    agreement, distribution = scorer_agreement(raw)
    crosswalk, anchor_stats = clinical_anchor_crosswalk()
    transitions = pd.read_csv(TRANSITIONS, low_memory=False)
    ensemble_coef = pd.read_csv(ENSEMBLE_COEFS)

    sensitivity_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    proxy_manifest: dict[str, dict] = {}
    for scorer in ("Gemma", "Qwen"):
        proxy = build_session_proxy(raw, session_axis, scorer)
        summary, coefficients = transition_sensitivity(transitions, proxy, scorer, ensemble_coef)
        sensitivity_frames.append(summary)
        coefficient_frames.append(coefficients)
        observed = proxy[proxy["rdoc_observed"]]
        proxy_manifest[scorer] = {
            "session_rows": int(len(proxy)),
            "session_rows_with_proxy": int(len(observed)),
            "patients_with_proxy": int(observed["id"].nunique()),
            "median_days_stale": float(observed["rdoc_days_stale"].median()),
        }

    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    agreement.to_csv(OUT / "scorer_agreement.csv", index=False)
    distribution.to_csv(OUT / "ensemble_score_distribution.csv", index=False)
    crosswalk.to_csv(OUT / "clinical_anchor_crosswalk.csv", index=False)
    anchor_stats.to_csv(OUT / "clinical_anchor_standardizers.csv", index=False)
    sensitivity.to_csv(OUT / "single_scorer_transition_sensitivity.csv", index=False)
    coefficients.to_csv(OUT / "single_scorer_transition_coefficients.csv", index=False)

    ensemble = pd.read_csv(ENSEMBLE_SUMMARY).rename(columns={"n": "n"})
    ensemble.insert(0, "scorer", "Ensemble")
    ensemble["top_domain"] = ensemble["top_domain"].map({f"B{i}": DOMAIN_NAMES[i] for i in range(6)})
    ensemble["beta_cosine_vs_ensemble"] = 1.0
    combined = pd.concat([ensemble, sensitivity], ignore_index=True, sort=False)
    combined.to_csv(OUT / "supp_table_single_scorer_transition.csv", index=False)

    manifest = {
        "analysis": "Exploratory scorer-robustness sensitivity requested during coauthor revision",
        "raw_source": str(RAW.relative_to(ROOT)),
        "transition_source": str(TRANSITIONS.relative_to(ROOT)),
        "retained_note_fields": int(len(raw)),
        "domains": DOMAIN_NAMES,
        "scorers": ["Gemma", "Qwen"],
        "scoring_scale": [0.0, 0.5, 1.0],
        "ridge_penalty": RIDGE,
        "patient_folds": FOLDS,
        "permutations": PERMUTATIONS,
        "proxy_summary": proxy_manifest,
        "outputs": [
            "scorer_agreement.csv",
            "ensemble_score_distribution.csv",
            "clinical_anchor_crosswalk.csv",
            "clinical_anchor_standardizers.csv",
            "single_scorer_transition_sensitivity.csv",
            "single_scorer_transition_coefficients.csv",
            "supp_table_single_scorer_transition.csv",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(agreement.to_string(index=False))
    print()
    print(crosswalk.to_string(index=False))
    print()
    print(sensitivity.to_string(index=False))
    print(f"\nWrote sensitivity outputs to {OUT}")


if __name__ == "__main__":
    main()
