#!/usr/bin/env python
"""Deterministic tests for the calibrated graded-response IRT observation model.

Covers the properties Codex required before any IRT smoke: the DGP emits item-level
responses on the calibrated trait scale and masks unobserved anchors; the responses
reach the BALL batch; the neural model carries FIXED (non-learnable) item parameters;
the exact per-item likelihood matches a brute-force category-probability sum; BALL and
the comparators receive the IDENTICAL frozen calibration; and the teacher/student
neural path executes under IRT and produces finite recovery and held-out metrics.

Run: python validation/test_irt_calibration.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direct_rdoc_benchmark as B          # noqa: E402
import direct_rdoc_fair_comparator as F    # noqa: E402
import ball_validation_harness as H        # noqa: E402

_ssm = sys.modules["simulations.src.methods.ball_ssm"]
SSMConfig = _ssm.SSMConfig
build_ssm_batch = _ssm.build_ssm_batch
fit_ball_ssm = _ssm.fit_ball_ssm
_make_models = _ssm._make_models
N_ITEMS = 9
MAX_TOTAL = 9 * 3


def _args(obs="irt", n=80):
    return types.SimpleNamespace(
        n=n, t=84, slow_fraction=0.75, delta_ar=0.3, rdoc_active_dims=3,
        rdoc_beta_seed=31011, rdoc_min_abs=0.35, anchor_observation=obs,
        irt_n_items=9, irt_discrimination=1.5, s0_basis="linear", iters=4,
        transition_sd=0.75, prior_sd=5.0, daily_ridge=1.0, beta_ridge=10.0)


def test_dgp_items_and_missing():
    data = B.make_data(_args(), 1729, 0.25, "linear")
    anc = data.anchors
    obs = anc[anc["observed"]]
    assert obs["irt_items"].apply(lambda x: x is not None and len(x) == N_ITEMS).all(), "items missing"
    naturally_unobserved = ~anc["observed"]
    if "measurement_eval" in anc.columns:
        naturally_unobserved &= ~anc["measurement_eval"].astype(bool)
    if "measurement_embargo" in anc.columns:
        naturally_unobserved &= ~anc["measurement_embargo"].astype(bool)
    unobs = anc[naturally_unobserved]
    if len(unobs):
        assert unobs["irt_total"].isna().all(), "unobserved totals must be NaN"
        assert unobs["irt_items"].apply(lambda x: all(not np.isfinite(v) for v in x)).all(), "unobserved items must be NaN"
    tot = obs["irt_total"].to_numpy(float)
    sat = float(np.mean((tot <= 0.5) | (tot >= MAX_TOTAL - 0.5)))
    assert sat < 0.10, f"instrument too saturated: {sat:.3f}"
    print("  [ok] DGP: item responses present, unobserved masked, de-saturated")


def test_anchor_batching():
    data = B.make_data(_args(), 1729, 0.25, "linear")
    batch, _ = build_ssm_batch(data, "train", torch.device("cpu"), 80)
    assert float(batch.anchor_obs.sum()) > 0, "BALL received no IRT anchors"
    assert int((batch.anchor_items != 0).sum()) > 0, "item responses not in batch"
    assert bool(torch.isfinite(batch.enc_features).all()), "encoder features must be finite under IRT"
    print(f"  [ok] anchor batching: {float(batch.anchor_obs.sum()):.0f} anchors, encoder finite")


def test_fixed_parameters_and_symmetry():
    data = B.make_data(_args(), 1729, 0.25, "linear")
    batch, _ = build_ssm_batch(data, "train", torch.device("cpu"), 80)
    gen, _ = _make_models(data, batch, SSMConfig(rdoc_drift_head=True, use_alpha_slow=False,
                          ensemble_size=1, teacher_epochs=1, student_epochs=1), torch.device("cpu"), causal=False)
    assert not any("irt" in n.lower() for n, _ in gen.named_parameters()), "IRT params must be fixed buffers"
    bufs = {n for n, _ in gen.named_buffers()}
    assert {"irt_item_disc", "irt_item_thresholds", "irt_loc", "irt_scale"} <= bufs, "missing IRT buffers"
    # BALL and the comparator receive the IDENTICAL frozen calibration
    st = F.init_irt_state(data, _args())
    # float32 buffers vs float64 config: compare at float32 precision
    assert np.allclose(gen.irt_item_disc.numpy(), st["disc"], atol=1e-5), "discrimination mismatch BALL vs comparator"
    assert np.allclose(gen.irt_item_thresholds.numpy(), np.asarray(st["thresholds"]), atol=1e-5), "threshold mismatch"
    assert abs(float(gen.irt_loc) - st["loc"]) < 1e-5 and abs(float(gen.irt_scale) - st["scale"]) < 1e-5, "affine mismatch"
    print("  [ok] fixed buffers, no learnable IRT params, BALL==comparator calibration")


def test_exact_likelihood():
    data = B.make_data(_args(), 1729, 0.25, "linear")
    batch, _ = build_ssm_batch(data, "train", torch.device("cpu"), 80)
    gen, _ = _make_models(data, batch, SSMConfig(rdoc_drift_head=True, use_alpha_slow=False,
                          ensemble_size=1, teacher_epochs=1, student_epochs=1), torch.device("cpu"), causal=False)
    loc, scale = float(gen.irt_loc), float(gen.irt_scale)
    disc, thr = gen.irt_item_disc.numpy(), gen.irt_item_thresholds.numpy()
    for zv, items in [(2.0, [1, 2, 0, 3, 1, 2, 0, 1, 3]), (-1.5, [0, 0, 1, 1, 2, 3, 2, 1, 0])]:
        lp = float(gen._irt_channel_log_lik(torch.tensor([[zv]]), torch.tensor([[items]], dtype=torch.float32)))
        trait = (zv - loc) / scale
        brute = 0.0
        for j in range(N_ITEMS):
            pstar = 1.0 / (1.0 + np.exp(-disc[j] * (trait - thr[j])))
            pad = np.concatenate(([1.0], pstar, [0.0]))
            probs = pad[:-1] - pad[1:]
            brute += float(np.log(max(probs[items[j]], 1e-12)))
        assert abs(lp - brute) < 1e-4, f"GRM mismatch at z={zv}: {lp} vs {brute}"
    print("  [ok] exact per-item GRM matches brute-force category sum")


def test_teacher_student_execution():
    data = B.make_data(_args(n=48), 1729, 0.25, "linear")
    cfg = SSMConfig(rdoc_drift_head=True, use_alpha_slow=False, ensemble_size=1,
                    teacher_epochs=4, student_epochs=4, anchor_warmup_epochs=2,
                    kl_warmup_epochs=2, max_individuals=48)
    student, teacher = fit_ball_ssm(data, cfg, device=torch.device("cpu"), causal=True,
                                    prediction_split=None, return_teacher=True)
    lr = H.latent_rmse(student.predictions, data.components, data.individuals, H.EVAL_SPLIT)
    assert np.isfinite(lr), "student latent RMSE must be finite under IRT"
    beta = np.asarray(student.metadata.get("rdoc_drift_beta_hat", []), float)
    assert beta.size and np.all(np.isfinite(beta)), "student beta must be finite"
    item_nll, etot_rmse = F.irt_holdout_metrics(data, student.predictions, "validation")
    assert np.isfinite(item_nll) and np.isfinite(etot_rmse), "IRT held-out metrics must be finite"
    print(f"  [ok] teacher/student trains under IRT: latent_rmse={lr:.3f} item_nll={item_nll:.3f}")


if __name__ == "__main__":
    tests = [test_dgp_items_and_missing, test_anchor_batching, test_fixed_parameters_and_symmetry,
             test_exact_likelihood, test_teacher_student_execution]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} IRT tests passed")
