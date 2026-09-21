"""Behavioral regression checks for the September code/text alignment fixes."""
from types import SimpleNamespace
import dataclasses
import numpy as np
import torch

from validation.direct_rdoc_benchmark import H, _base_config, _inject_benchmark_cell, _ssm, make_data
from validation.direct_rdoc_fair_comparator import causal_encoder_action_array
from validation.test_direct_rdoc_input_contract import _args


def test_interval_experiment_uses_the_primary_direct_transition_configuration():
    from simulations.paper.run_fig3_calibration import calibration_configs
    from validation.direct_rdoc_benchmark import ball_config
    args = _args()
    args.teacher_epochs = 300
    args.student_epochs = 300
    args.anchor_warmup = 120
    args.kl_warmup = 80
    args.n = 1000
    args.members = args.ensemble_size = 5
    args.d_model = 96
    args.n_layers = 3
    args.batch_size = 32
    args.alpha_lasso_xi = 2.0
    args.rdoc_drift_l1 = 0.0
    args.delta_phi = 0.3
    args.ball_no_ehr_drift = False
    args.progress_every_epochs = 10
    args.diagnostics_every_epochs = 5
    args.checkpoint_every_epochs = 10
    simulation, interval_model = calibration_configs(args, 2027)
    primary_simulation = _base_config(args, 2027, 0.1, 'linear')
    assert dataclasses.asdict(simulation) == dataclasses.asdict(primary_simulation)
    primary_model = ball_config(args, 2027)
    # Replicate-specific model seeds intentionally remain those of the interval study.
    interval_values = dataclasses.asdict(interval_model)
    primary_values = dataclasses.asdict(primary_model)
    for key in ('seed', 'progress_every_epochs', 'diagnostics_every_epochs', 'state_checkpoint_every_epochs'):
        interval_values.pop(key)
        primary_values.pop(key)
    assert interval_values == primary_values


def test_neural_and_classical_treatment_inputs_are_strictly_lagged():
    args = _args()
    args.n = 30
    args.t = 42
    data = make_data(args, 2027, 0.1, 'linear')
    batch, _ = _ssm.build_ssm_batch(data, 'train', torch.device('cpu'), args.n, causal_impute=True)
    for row, pid in enumerate(batch.ids):
        person = data.components.loc[data.components.id.eq(pid)].sort_values('t')
        expected = np.r_[-1, person.a.to_numpy()[:-1]]
        np.testing.assert_array_equal(person.encoder_a, expected)
        np.testing.assert_array_equal(batch.encoder_action[row].numpy(), expected + 1)
        np.testing.assert_array_equal(causal_encoder_action_array(person), expected)

    # The backward-compatible input path must implement the identical contract.
    legacy = dataclasses.replace(data, components=data.components.drop(columns='encoder_a'))
    older, _ = _ssm.build_ssm_batch(legacy, 'train', torch.device('cpu'), args.n, causal_impute=True)
    torch.testing.assert_close(batch.encoder_action, older.encoder_action)


def test_treatment_free_coefficient_fixture_has_no_lagged_treatment():
    from validation.audit_classical_coefficient_recovery import _fit_args, _one_active_fixture
    data = _one_active_fixture(_fit_args(15, 28), active_index=0, seed=2027)
    assert data.components.a.eq(-1).all()
    assert data.components.encoder_a.eq(-1).all()
    for _, patient in data.components.groupby('id'):
        assert (causal_encoder_action_array(patient.sort_values('t')) == -1).all()


def test_subtype_effect_survives_and_pooled_signal_share_is_preserved():
    args = _args()
    args.n = 60
    for share in (0.1, 0.25):
        base = H.generate_dataset(_base_config(args, 2027, share, 'heterogeneous'))
        varied = _inject_benchmark_cell(base, 'heterogeneous', share, args)
        uniform_args = SimpleNamespace(**vars(args))
        uniform_args.heterogeneity_strength = 0.0
        uniform = _inject_benchmark_cell(base, 'heterogeneous', share, uniform_args)
        assert not np.allclose(varied.components.rdoc_direct_signal, uniform.components.rdoc_direct_signal)
        by_patient = []
        for pid, person in varied.components.groupby('id'):
            original = base.components.loc[base.components.id.eq(pid)].sort_values('t')
            base_var = np.var(np.diff(original.L))
            signal_var = np.var(person.sort_values('t').rdoc_direct_signal)
            by_patient.append((int(person.subtype.iloc[0]), base_var, signal_var))
        values = np.asarray(by_patient)
        pooled_share = values[:, 2].mean() / (values[:, 1].mean() + values[:, 2].mean())
        np.testing.assert_allclose(pooled_share, share, atol=1e-12)
        relative = np.array([np.mean(values[values[:,0]==s,2] / values[values[:,0]==s,1]) for s in range(3)])
        np.testing.assert_allclose(relative / relative[1], [0.25, 1.0, 2.25], rtol=1e-12)
