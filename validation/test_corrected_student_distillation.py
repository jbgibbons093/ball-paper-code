from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from validation.direct_rdoc_benchmark import _ssm as ssm


def test_inactive_alpha_does_not_change_anchored_state() -> None:
    config = ssm.SSMConfig(use_alpha_slow=False)
    mean = torch.zeros(2, 3, 5)
    log_variance = torch.zeros_like(mean)
    rdoc = torch.randn(2, 3, 3)
    reference_mean, reference_covariance = ssm._anchored_state_moments(
        mean, log_variance, rdoc, config
    )
    altered_mean = mean.clone()
    altered_log_variance = log_variance.clone()
    altered_mean[..., :3] = 1000.0
    altered_log_variance[..., :3] = 4.0
    altered_state_mean, altered_state_covariance = ssm._anchored_state_moments(
        altered_mean, altered_log_variance, rdoc, config
    )
    torch.testing.assert_close(altered_state_mean, reference_mean)
    torch.testing.assert_close(altered_state_covariance, reference_covariance)


def test_active_slow_state_induces_expected_two_channel_covariance() -> None:
    config = ssm.SSMConfig(use_alpha_slow=True)
    mean = torch.tensor([[[1.0, -2.0, 0.5, -0.25]]])
    log_variance = torch.log(torch.tensor([[[0.25, 1.0, 0.5, 0.75]]]))
    rdoc = torch.tensor([[[2.0, 0.5]]])
    state_mean, covariance = ssm._anchored_state_moments(
        mean, log_variance, rdoc, config
    )
    expected_slow_mean = 1.0
    expected_slow_variance = 1.25
    torch.testing.assert_close(
        state_mean,
        torch.tensor([[[expected_slow_mean + 0.5, expected_slow_mean - 0.25]]]),
    )
    torch.testing.assert_close(
        covariance,
        torch.tensor([[[[expected_slow_variance + 0.5, expected_slow_variance],
                         [expected_slow_variance, expected_slow_variance + 0.75]]]]),
    )


def test_teacher_to_student_kl_matches_torch_and_identical_is_zero() -> None:
    teacher_mean = torch.tensor([[[0.0, 0.5]]])
    teacher_covariance = torch.tensor([[[[1.0, 0.2], [0.2, 0.8]]]])
    student_mean = torch.tensor([[[1.0, -0.5]]])
    student_covariance = torch.tensor([[[[1.5, 0.1], [0.1, 0.6]]]])
    observed = ssm._anchored_gaussian_kl(
        teacher_mean,
        teacher_covariance,
        student_mean,
        student_covariance,
        jitter=1e-6,
        direction="teacher_to_student",
    )
    identity = torch.eye(2) * 1e-6
    teacher_distribution = torch.distributions.MultivariateNormal(
        teacher_mean[0, 0], teacher_covariance[0, 0] + identity
    )
    student_distribution = torch.distributions.MultivariateNormal(
        student_mean[0, 0], student_covariance[0, 0] + identity
    )
    expected = torch.distributions.kl_divergence(
        teacher_distribution, student_distribution
    )
    torch.testing.assert_close(observed[0, 0], expected)
    identical = ssm._anchored_gaussian_kl(
        teacher_mean,
        teacher_covariance,
        teacher_mean,
        teacher_covariance,
        jitter=1e-6,
        direction="teacher_to_student",
    )
    torch.testing.assert_close(identical, torch.zeros_like(identical), atol=1e-6, rtol=0.0)


def _minimal_batch() -> object:
    zeros_3 = torch.zeros(1, 2, 1)
    return ssm.SSMBatch(
        ids=[1],
        t_values=np.asarray([0, 1]),
        rdoc=zeros_3.clone(),
        ehr=zeros_3.clone(),
        x_raw=zeros_3.clone(),
        x_obs=torch.ones_like(zeros_3),
        enc_features=zeros_3.clone(),
        action=torch.zeros(1, 2, dtype=torch.long),
        encoder_action=torch.zeros(1, 2, dtype=torch.long),
        dt=torch.ones(1, 2),
        valid=torch.ones(1, 2, dtype=torch.bool),
        anchor_value=torch.tensor([[[0.0], [1.0]]]),
        anchor_obs=torch.tensor([[[0.0], [1.0]]]),
        anchor_total=zeros_3.clone(),
        anchor_items=torch.zeros(1, 2, 1, 1),
        anchor_channel=torch.zeros(1, 2, 1, dtype=torch.long),
        win_start=torch.tensor([[[0], [0]]]),
        win_end=torch.tensor([[[0], [1]]]),
        z_truth=torch.zeros(1, 2, 2),
    )


def test_gaussian_predictive_anchor_uses_recall_window_variance() -> None:
    config = ssm.SSMConfig(anchor_session_balance=False)
    model = ssm.GenerativeModel(
        q=1,
        ehr_dim=1,
        config=config,
        n_actions=1,
        obs_sd=(1.0,),
        loadings=(1.0,),
        intercepts=(0.0,),
        anchor_channels=(0,),
    )
    state_mean = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    state_covariance = torch.zeros(1, 2, 2, 2)
    state_covariance[..., 0, 0] = 1.0
    state_covariance[..., 1, 1] = 1.0
    nll_sum, event_count = model.anchor_predictive_nll(
        state_mean, state_covariance, _minimal_batch()
    )
    expected_variance = 1.0 + 0.25 + 0.25
    expected_nll = 0.5 * (math.log(2.0 * math.pi) + math.log(expected_variance))
    torch.testing.assert_close(nll_sum, torch.tensor(expected_nll))
    torch.testing.assert_close(event_count, torch.tensor(1.0))


def test_teacher_initialization_copies_encoder_and_preserves_causal_mask() -> None:
    config = ssm.SSMConfig(
        d_model=8,
        n_heads=2,
        n_layers=1,
        action_embed_dim=2,
        dropout=0.0,
    )
    teacher_embedding = nn.Embedding(3, 2)
    student_embedding = nn.Embedding(3, 2)
    teacher = ssm.InferenceTransformer(4, 2, 5, config, teacher_embedding, causal=False)
    student = ssm.InferenceTransformer(4, 2, 5, config, student_embedding, causal=True)
    summary = ssm._initialize_student_from_teacher(student, teacher, "teacher")
    assert summary["copied_keys"]
    assert not summary["skipped_keys"]
    assert student.causal is True
    for key, value in teacher.state_dict().items():
        assert torch.equal(value, student.state_dict()[key])


def test_causal_student_is_invariant_to_future_feature_perturbation() -> None:
    config = ssm.SSMConfig(
        d_model=8,
        n_heads=2,
        n_layers=1,
        action_embed_dim=2,
        dropout=0.0,
    )
    network = ssm.InferenceTransformer(
        3, 1, 4, config, nn.Embedding(2, 2), causal=True
    )
    network.eval()
    batch = _minimal_batch()
    batch.enc_features = torch.zeros(1, 2, 3)
    with torch.no_grad():
        reference_mean, reference_log_variance = network(batch)
    altered = ssm._subbatch(batch, torch.tensor([0]))
    altered.enc_features = altered.enc_features.clone()
    altered.enc_features[:, 1, :] = 1000.0
    with torch.no_grad():
        altered_mean, altered_log_variance = network(altered)
    torch.testing.assert_close(altered_mean[:, 0], reference_mean[:, 0])
    torch.testing.assert_close(altered_log_variance[:, 0], reference_log_variance[:, 0])
