"""Tests for empirical questionnaire range and measurement-status checks."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BALL

BALL._install_virtual_package()
from empirical.build_sparse_anchors import _build_anchor_set, _genuine_masks  # noqa: E402
from empirical.fit_empirical import _channel_eval, _empirical_ode_tensors  # noqa: E402


def test_out_of_range_questionnaire_totals_are_excluded() -> None:
    frame = pd.DataFrame(
        {
            "PatientFID": [1, 1, 1],
            "session": [0, 1, 2],
            "ServiceDate": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "day": [0, 1, 2],
            "phq9": [27.0, 28.0, 10.0],
            "phq9_is_imputed": [0, 0, 1],
            "phq9_is_interpolated": [0, 0, 0],
            "gad_filled": [21.0, 23.0, 7.0],
            "gad_measured": [1, 1, 0],
            "bdi_filled": [63.0, 64.0, 12.0],
            "bdi_measured": [1, 1, 0],
        }
    )
    masks = _genuine_masks(frame)
    assert masks["PHQ9"].tolist() == [True, False, False]
    assert masks["GAD7"].tolist() == [True, False, False]
    assert masks["BDI"].tolist() == [True, False, False]

    anchors = _build_anchor_set(frame, masks, cadence_days=14)
    assert set(anchors["value"].astype(float)) == {21.0, 27.0, 63.0}
    assert set(anchors["anchor"]) == {"PHQ9", "GAD7", "BDI"}


def test_sparsification_and_event_accounting_are_instrument_specific() -> None:
    frame = pd.DataFrame(
        {
            "PatientFID": [1, 1],
            "session": [0, 1],
            "ServiceDate": pd.to_datetime(["2025-01-01", "2025-01-20"]),
            "day": [0, 19],
            "phq9": [12.0, 11.0],
            "phq9_is_imputed": [0, 0],
            "phq9_is_interpolated": [0, 0],
            "gad_filled": [8.0, 7.0],
            "gad_measured": [1, 1],
            "bdi_filled": [24.0, 22.0],
            "bdi_measured": [1, 1],
        }
    )
    anchors = _build_anchor_set(frame, _genuine_masks(frame), cadence_days=28)
    first_session = anchors[anchors["session"].eq(0)]
    assert set(first_session.loc[first_session["role"].eq("anchor"), "anchor"]) == {
        "PHQ9",
        "GAD7",
        "BDI",
    }
    counts = anchors["evaluation_role"].value_counts()
    assert len(anchors) == int(
        counts.get("retained", 0)
        + counts.get("withheld_eligible", 0)
        + counts.get("withheld_ineligible_overlap", 0)
    )
    assert not anchors.duplicated(["id", "session", "anchor"]).any()


def test_cross_instrument_baselines_use_the_frozen_measurement_link() -> None:
    anchors = pd.DataFrame(
        [
            {
                "id": 1,
                "session": 0,
                "day": 0,
                "anchor": "PHQ9",
                "value": 13.5,
                "role": "anchor",
                "evaluation_role": "retained",
                "window_start_day": -13,
                "window_end_day": 0,
            },
            {
                "id": 1,
                "session": 1,
                "day": 10,
                "anchor": "BDI",
                "value": 31.5,
                "role": "anchor",
                "evaluation_role": "retained",
                "window_start_day": -3,
                "window_end_day": 10,
            },
            {
                "id": 1,
                "session": 2,
                "day": 30,
                "anchor": "PHQ9",
                "value": 13.5,
                "role": "withheld",
                "evaluation_role": "withheld_eligible",
                "window_start_day": 17,
                "window_end_day": 30,
            },
        ]
    )
    sessions = pd.DataFrame({"id": [1, 1, 1], "day": [0, 10, 30], "session": [0, 1, 2]})
    measurement = {
        "instruments": {
            "PHQ9": {"intercept": 0.0, "loading": 1.0, "observation_sd": 0.2},
            "BDI": {"intercept": -0.2, "loading": 0.5, "observation_sd": 0.2},
            "GAD7": {"intercept": 0.0, "loading": 1.0, "observation_sd": 0.2},
        },
        "development_data": {
            "normalized_score_mean_by_instrument": {"PHQ9": 0.0, "BDI": 0.0, "GAD7": 0.0},
            "normalized_score_sd_by_instrument": {"PHQ9": 1.0, "BDI": 1.0, "GAD7": 1.0},
        },
    }
    zero = {1: np.zeros(3, dtype=float)}
    result = _channel_eval(
        anchors,
        sessions,
        {"PHQ9": (13.5, 13.5), "BDI": (31.5, 31.5), "GAD7": (10.5, 10.5)},
        neural_predictions={
            "student": zero,
            "teacher": zero,
            "s0": zero,
            "student_sd": {1: np.full(3, 0.1, dtype=float)},
        },
        session_days_by_id={1: np.array([0, 10, 30], dtype=int)},
        measurement_calibration=measurement,
    )
    assert len(result) == 1
    # Normalized BDI=0 corresponds to latent depression (0 - -0.2) / 0.5 = 0.4,
    # which maps to normalized PHQ-9=0.4 through its reference link.
    assert float(result.iloc[0]["locf"]) == 0.4
    # The cumulative mean averages the earlier PHQ-9 latent value 0.0 and the
    # later BDI-II latent value 0.4 on the common depression scale.
    assert float(result.iloc[0]["causal_anchor_mean"]) == 0.2


def test_recurrent_comparator_lags_consecutive_instrument_events_without_false_duplicates() -> None:
    config = SimpleNamespace(
        t=3,
        p_daily=1,
        n_treatment_types=1,
        anchor_instrument_names=("PHQ9", "BDI", "GAD7"),
    )
    data = SimpleNamespace(
        config=config,
        components=pd.DataFrame(
            {
                "id": [1, 1, 1],
                "t": [0, 1, 2],
                    "session_observed": [True, True, True],
                    "dt": [1.0, 1.0, 1.0],
                    "a": [0, 0, 0],
            }
        ),
        daily=pd.DataFrame(
            {
                "id": [1, 1, 1],
                "t": [0, 1, 2],
                "X0": [0.0, 0.0, 0.0],
                "input_X0": [True, True, True],
            }
        ),
        anchors=pd.DataFrame(
            {
                "id": [1, 1],
                "t": [0, 1],
                "observed": [True, True],
                "anchor": ["Y1", "Y1"],
                "instrument": ["PHQ9", "PHQ9"],
                "value": [0.1, 0.2],
                "window_start": [0, 0],
                "window_end": [0, 1],
                "intercept": [0.0, 0.0],
                "loading": [1.0, 1.0],
            }
        ),
    )
    x, _, _, anchor_rows = _empirical_ode_tensors(data, [1], torch.device("cpu"))
    # The first questionnaire becomes available at the next recorded session,
    # while the second remains a separate event at its own source session.
    assert np.isclose(float(x[0, 1, -6]), 0.1)
    assert len(anchor_rows) == 2
