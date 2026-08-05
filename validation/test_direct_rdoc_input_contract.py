from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import torch

from validation.direct_rdoc_benchmark import (
    _ode_rnn_tensors,
    fit_gp_causal_filter,
    make_data,
    strict_transition_only_data,
)
from simulations.src.methods.ball_ssm import build_ssm_batch
from validation.direct_rdoc_fair_comparator import (
    causal_encoder_action_array,
    fit_daily_readout,
    observed_treatment_history_arrays,
    person_arrays,
)


def _args(anchor_observation: str = "gaussian") -> SimpleNamespace:
    return SimpleNamespace(
        n=80,
        t=84,
        slow_fraction=0.75,
        delta_ar=0.3,
        rdoc_active_dims=3,
        rdoc_beta_seed=31011,
        rdoc_min_abs=0.35,
        interaction_strength=1.5,
        nonlinear_strength=0.75,
        heterogeneity_strength=0.5,
        n_subtypes_for_scale=3,
        missing_daily_probability=0.55,
        missing_mnar_gamma=0.7,
        missing_proxy_noise=0.4,
        missing_density_multiplier=1.5,
        anchor_observation=anchor_observation,
        irt_n_items=9,
        irt_discrimination=1.5,
        daily_ridge=1.0,
    )


def test_validation_measurements_are_hidden_with_recall_window_embargo() -> None:
    data = make_data(_args(), 1729, 0.25, "nonlinear")
    targets = data.anchors.loc[data.anchors["measurement_eval"].astype(bool)]
    assert len(targets) > 0
    assert not targets["observed"].astype(bool).any()
    for target in targets.itertuples(index=False):
        retained = data.anchors.loc[
            data.anchors["id"].astype(int).eq(int(target.id))
            & data.anchors["observed"].astype(bool)
        ]
        overlap = retained.loc[
            retained["window_start"].le(int(target.window_end))
            & retained["window_end"].ge(int(target.window_start))
        ]
        assert overlap.empty


def test_item_response_targets_remain_available_only_for_scoring() -> None:
    data = make_data(_args("irt"), 1729, 0.25, "linear")
    targets = data.anchors.loc[data.anchors["measurement_eval"].astype(bool)]
    assert len(targets) > 0
    assert not targets["observed"].astype(bool).any()
    assert np.isfinite(targets["irt_total"].to_numpy(dtype=float)).all()
    assert targets["irt_items"].apply(lambda values: np.isfinite(values).all()).all()


def test_classical_treatment_burden_uses_observed_daily_proxy() -> None:
    data = make_data(_args(), 1729, 0.25, "linear")
    train_ids = set(
        data.individuals.loc[data.individuals["split"] == "train", "id"].astype(int)
    )
    daily_readout = fit_daily_readout(data, train_ids)
    pid = min(train_ids)
    comp_i, _, _, _, _, _, _, recent, burden = person_arrays(data, pid, daily_readout)
    daily_i = data.daily.loc[data.daily["id"].astype(int).eq(pid)].sort_values("t").reset_index(drop=True)
    altered = comp_i.copy()
    altered["treatment_burden"] = 1e9
    altered_recent, altered_burden = observed_treatment_history_arrays(altered, daily_i, data.config)
    assert np.allclose(recent, altered_recent)
    assert np.allclose(burden, altered_burden)
    assert np.max(np.abs(altered_burden)) < 1e6


def test_ode_rnn_uses_previous_gap_and_causal_treatment_context() -> None:
    data = make_data(_args(), 1729, 0.25, "linear")
    pid = int(data.individuals.iloc[0]["id"])
    mask = data.components["id"].astype(int).eq(pid)
    ordered_index = data.components.loc[mask].sort_values("t").index
    gap_pattern = np.arange(1, len(ordered_index) + 1, dtype=float)
    data.components.loc[ordered_index, "dt"] = gap_pattern
    data.components.loc[ordered_index, "encoder_a"] = -1
    data.components.loc[ordered_index[1:], "encoder_a"] = data.components.loc[
        ordered_index[:-1], "a"
    ].to_numpy()

    x, previous_gap, _, _, _ = _ode_rnn_tensors(data, [pid], torch.device("cpu"))
    observed = previous_gap.detach().cpu().numpy()[0]
    assert observed[0] == 0.0
    assert np.allclose(observed[1:], gap_pattern[:-1])

    comp_i = data.components.loc[ordered_index].sort_values("t").reset_index(drop=True)
    assert np.array_equal(
        causal_encoder_action_array(comp_i),
        comp_i["encoder_a"].to_numpy(dtype=int),
    )
    assert x.shape[1] == len(ordered_index)


def test_gp_filter_excludes_future_questionnaires() -> None:
    args = _args()
    args.n = 40
    data = make_data(args, 1729, 0.25, "linear")
    test_ids = set(
        data.individuals.loc[data.individuals["split"] == "test", "id"].astype(int)
    )
    observed = data.anchors.loc[
        data.anchors["observed"].astype(bool)
        & data.anchors["id"].astype(int).isin(test_ids)
    ].copy()
    counts = observed.groupby("id").size()
    pid = int(counts.loc[counts >= 2].index[0])
    patient_anchors = observed.loc[observed["id"].astype(int).eq(pid)].sort_values("t")
    future_index = int(patient_anchors.index[-1])
    available_at = int(patient_anchors.iloc[-1]["t"])

    altered_anchors = data.anchors.copy(deep=True)
    altered_anchors.loc[future_index, "value"] = (
        float(altered_anchors.loc[future_index, "value"]) + 25.0
    )
    altered = dataclasses.replace(data, anchors=altered_anchors)

    causal_original = fit_gp_causal_filter(data, args)
    causal_altered = fit_gp_causal_filter(altered, args)

    causal_before_altered = causal_altered.loc[
        causal_altered["id"].astype(int).eq(pid) & causal_altered["t"].le(available_at),
        "L_hat",
    ].to_numpy(dtype=float)
    causal_before = causal_original.loc[
        causal_original["id"].astype(int).eq(pid) & causal_original["t"].le(available_at),
        "L_hat",
    ].to_numpy(dtype=float)
    assert np.allclose(causal_before, causal_before_altered, atol=1e-10, rtol=0.0)


def test_strict_transition_arm_removes_proxies_from_all_structured_paths() -> None:
    data = make_data(_args(), 1729, 0.25, "linear")
    strict = strict_transition_only_data(data)
    proxy_columns = [f"X{index}" for index in range(8, 12)]
    assert (strict.daily[proxy_columns].to_numpy(dtype=float) == 0.0).all()
    input_columns = [f"input_{column}" for column in proxy_columns]
    if set(input_columns).issubset(strict.daily.columns):
        assert not strict.daily[input_columns].to_numpy(dtype=bool).any()
    assert not strict.daily[[f"obs_{column}" for column in proxy_columns]].to_numpy(dtype=bool).any()
    batch, _ = build_ssm_batch(
        strict,
        "train",
        torch.device("cpu"),
        max_individuals=20,
        include_rdoc_in_encoder=False,
    )
    assert torch.count_nonzero(batch.ehr[:, :, 8:12]) == 0
    assert torch.count_nonzero(batch.x_raw[:, :, 8:12]) == 0
    assert torch.count_nonzero(batch.x_obs[:, :, 8:12]) == 0
    assert strict.metadata["structured_reconstruction_proxy_columns"] == []
