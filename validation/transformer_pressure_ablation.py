#!/usr/bin/env python
"""Transformer decomposition-pressure ablation (validation step B).

Validation A established that on the empirical-calibrated DGP the rigid structural
posterior recovers the slow/RDoC decomposition under a reverting delta prior, while
the flexible transformer does not: its EHR-conditioned delta drift re-creates
persistence from the covariates and the optimizer zeroes alpha. This ablation tests
whether targeted decomposition pressure lets the transformer recover the
decomposition, across the empirically justified reverting-prior family delta_phi in
{0.3, 0.5, 0.7} (not only the easy end), against the matched structural reference.

For each delta_phi in the family the DGP is generated with that residual persistence
(delta_ar = delta_phi) and slow_fraction 0.75, so the model prior is matched to a
plausible true persistence from the empirical PHQ-9 bracket. Arms:

  current        the unmodified transformer (control; expected alpha-collapse).
  no_ehr_drift   drop the EHR input from g_theta so the delta drift cannot re-create
                 persistence from the covariates (attacks the verified failure mode).
  delta_persist  penalize low-frequency (persistent) power in delta.
  alpha_anchor   anchor the slow component alpha'RDoC to the low-pass composite.
  combined       no_ehr_drift + delta_persist.

The model knobs are the default-off SSMConfig flags added for step B, so the
canonical fit is unchanged when no arm enables them. Decomposition metrics
(rho_RDoC, slow recovery, leakage) are primary; composite RMSE is severity-fit
context. The structural reference is reported at the same delta_phi.

This is a targeted ablation, not a manuscript run: single ensemble member and
reduced epochs by default. Scale up only after an arm clears the gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_validation_harness as H  # noqa: E402  (loads BALL + virtual package)

_ssm = sys.modules["simulations.src.methods.ball_ssm"]
SSMConfig = _ssm.SSMConfig
fit_ball_ssm = _ssm.fit_ball_ssm

SLOW_FRACTION = 0.75
DELTA_PHI_FAMILY = (0.3, 0.5, 0.7)
SCOPING_DELTA_PHI = (0.5,)  # central empirical prior for the fast scoping pass
# Pressure-weight selection (step 2): pick the delta-persistence penalty by a
# non-truth, held-out criterion (validation-anchor RMSE), not by test-set rho.
PRESSURE_PENALTY_GRID = (0.0, 300.0, 600.0, 1000.0)
SELECT_XI = 0.05  # loosened alpha-Laplace for the pressure structure (near structural xi 0.08)

# Scoping v1 (current, no_ehr_drift, delta_persist, alpha_anchor, combined) showed
# alpha collapses to rho~0 for ALL arms, including EHR removal, so the alpha-sparsity
# Laplace (xi=2.0) is the dominant collapsing force, not the EHR drift. Loosening the
# Laplace removes the barrier; pressure supplies the incentive; only together can
# they revive alpha. v2 tests lasso-loosening alone and combined with pressure.
# (name, flags). None placeholders are filled from CLI weights at runtime.
ARMS = [
    ("current", {}),
    ("lasso_loosen", {"alpha_lasso_xi": None}),
    ("lasso_persist", {"alpha_lasso_xi": None, "delta_persistence_penalty": None}),
    ("lasso_anchor", {"alpha_lasso_xi": None, "alpha_proxy_anchor_weight": None}),
    ("lasso_noehr_persist", {"alpha_lasso_xi": None, "delta_drift_use_ehr": False,
                             "delta_persistence_penalty": None}),
]
ARM_ORDER = [name for name, _ in ARMS]


def resolve_flags(arm_flags: dict, penalty: float, anchor_weight: float, xi_low: float) -> dict:
    out = {}
    for k, v in arm_flags.items():
        if v is None and k == "delta_persistence_penalty":
            out[k] = penalty
        elif v is None and k == "alpha_proxy_anchor_weight":
            out[k] = anchor_weight
        elif v is None and k == "alpha_lasso_xi":
            out[k] = xi_low
        else:
            out[k] = v
    return out


def make_dgp(delta_phi: float, seed: int, n: int, t: int):
    return H.generate_dataset(
        dataclasses.replace(
            H.SimulationConfig(),
            seed=seed,
            n=n,
            t=t,
            slow_fraction=SLOW_FRACTION,
            delta_ar=tuple([float(delta_phi)] * H.N_SUBTYPES),
        )
    )


def build_ssm_config(delta_phi: float, flags: dict, seed: int, args) -> "SSMConfig":
    return SSMConfig(
        delta_phi=float(delta_phi),
        ensemble_size=1,
        anchor_warmup_epochs=args.anchor_warmup,
        kl_warmup_epochs=args.kl_warmup,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        seed=seed,
        **flags,
    )


def metrics_row(prefix: str, scored: dict) -> dict:
    keep = ["slow_recovery_corr", "rho_rdoc", "slow_to_fast_leakage", "fast_recovery_corr",
            "comp_rmse", "active_f1"]
    return {f"{prefix}_{k}": scored[k] for k in keep}


def run_one(delta_phi: float, arm_name: str, flags: dict, seed: int, n: int, t: int, args, data) -> dict:
    cfg = build_ssm_config(delta_phi, flags, seed, args)
    student, teacher = fit_ball_ssm(data, cfg, causal=True, return_teacher=True)
    row = {
        "delta_phi": delta_phi,
        "arm": arm_name,
        "seed": seed,
        "n": n,
        "delta_drift_use_ehr": flags.get("delta_drift_use_ehr", True),
        "delta_persistence_penalty": flags.get("delta_persistence_penalty", 0.0),
        "alpha_proxy_anchor_weight": flags.get("alpha_proxy_anchor_weight", 0.0),
        "alpha_lasso_xi": flags.get("alpha_lasso_xi", float(SSMConfig().alpha_lasso_xi)),
    }
    row.update(metrics_row("teacher", H.score(data, teacher)))
    row.update(metrics_row("student", H.score(data, student)))
    return row


def run_selection(args) -> None:
    """Step 2: select the delta-persistence pressure weight by a NON-TRUTH criterion.

    Held-out validation-anchor RMSE is the criterion. The transformer's validation
    patients are not in training (anchors enter only through the train ELBO), so this
    is genuinely held out, unlike the structural's in-sample anchor fit. Test-set rho
    is recorded as a DIAGNOSTIC only and never used for selection, which removes the
    test-tuning concern from the v3 result. The pressure structure (drop EHR from the
    delta drift, loosen the alpha-Laplace to SELECT_XI) is fixed; only the penalty
    weight is selected.
    """
    family = DELTA_PHI_FAMILY if args.full else SCOPING_DELTA_PHI
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (H.REPO_ROOT / "validation" / "outputs" / f"stepB_select_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"step B pressure selection by held-out validation-anchor RMSE: family={family}, "
          f"penalty_grid={PRESSURE_PENALTY_GRID}, xi={SELECT_XI}, seeds={args.seeds}, n={args.n}")
    print(f"output -> {out_dir}")

    rows = []
    for delta_phi in family:
        for seed in args.seeds:
            data = make_dgp(delta_phi, seed, args.n, args.t)
            for penalty in PRESSURE_PENALTY_GRID:
                flags = {"alpha_lasso_xi": SELECT_XI, "delta_drift_use_ehr": False,
                         "delta_persistence_penalty": penalty}
                cfg = build_ssm_config(delta_phi, flags, seed, args)
                _, teacher = fit_ball_ssm(data, cfg, causal=True, return_teacher=True,
                                          prediction_split="validation")
                preds = teacher.predictions
                val_anchor = H.val_anchor_rmse(data, preds)
                rho_val = H.rdoc_recovery_fraction(preds, data, "validation")
                slow_val = H.identifiability_diagnostics(preds, data, "validation")["slow_recovery_corr"]
                rows.append({"delta_phi": delta_phi, "seed": seed, "penalty": penalty,
                             "val_anchor_rmse": val_anchor, "rho_rdoc_val_diag": rho_val,
                             "slow_corr_val_diag": slow_val})
                print(f"  delta_phi={delta_phi} seed={seed} penalty={penalty:>6.0f}: "
                      f"val_anchor={val_anchor:.4f} | (diag) rho_val={rho_val:.3f} slow_val={slow_val:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pressure_selection.csv", index=False)
    by_pen = df.groupby("penalty").agg(
        val_anchor=("val_anchor_rmse", "mean"), rho_diag=("rho_rdoc_val_diag", "mean")
    ).reset_index()
    selected = float(by_pen.loc[by_pen["val_anchor"].idxmin(), "penalty"])
    print("\n=== selection by held-out validation-anchor RMSE (lower = better held-out fit) ===")
    for _, r in by_pen.iterrows():
        mark = "  <-- selected" if r["penalty"] == selected else ""
        print(f"  penalty={r['penalty']:>6.0f}: val_anchor={r['val_anchor']:.4f}  rho_diag={r['rho_diag']:.3f}{mark}")
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "ball_py_sha256": H.file_sha256(H.BALL_PATH),
        "criterion": "held-out validation-anchor RMSE (non-truth); test rho recorded as diagnostic only",
        "fixed_pressure_structure": {"delta_drift_use_ehr": False, "alpha_lasso_xi": SELECT_XI},
        "penalty_grid": list(PRESSURE_PENALTY_GRID),
        "family": list(family),
        "seeds": list(args.seeds),
        "selected_delta_persistence_penalty": selected,
        "epochs": {"anchor_warmup": args.anchor_warmup, "teacher": args.teacher_epochs,
                   "student": args.student_epochs},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nselected delta_persistence_penalty = {selected} (by held-out validation-anchor RMSE)")
    print(f"wrote:\n  {out_dir / 'pressure_selection.csv'}\n  {out_dir / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer decomposition-pressure ablation (step B).")
    parser.add_argument("--select", action="store_true",
                        help="step 2: select the pressure weight by held-out validation-anchor RMSE")
    parser.add_argument("--full", action="store_true", help="sweep delta_phi {0.3,0.5,0.7} (else central 0.5)")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--t", type=int, default=H.DEFAULT_T)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729])
    parser.add_argument("--arms", type=str, nargs="+", default=ARM_ORDER, choices=ARM_ORDER)
    parser.add_argument("--penalty", type=float, default=200.0, help="delta_persistence_penalty weight")
    parser.add_argument("--anchor-weight", type=float, default=200.0, help="alpha_proxy_anchor_weight")
    parser.add_argument("--xi-low", type=float, default=0.1, help="loosened alpha_lasso_xi for lasso arms")
    parser.add_argument("--anchor-warmup", type=int, default=60)
    parser.add_argument("--kl-warmup", type=int, default=40)
    parser.add_argument("--teacher-epochs", type=int, default=120)
    parser.add_argument("--student-epochs", type=int, default=80)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.select:
        run_selection(args)
        return

    family = DELTA_PHI_FAMILY if args.full else SCOPING_DELTA_PHI
    arms = [(name, flags) for name, flags in ARMS if name in set(args.arms)]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (H.REPO_ROOT / "validation" / "outputs" / f"stepB_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"step B: delta_phi family={family}, arms={[a for a, _ in arms]}, seeds={args.seeds}, n={args.n}")
    print(f"epochs: warmup={args.anchor_warmup} teacher={args.teacher_epochs} student={args.student_epochs} "
          f"| penalty={args.penalty} anchor_weight={args.anchor_weight}")
    print(f"output -> {out_dir}")

    rows = []
    struct_rows = []
    for delta_phi in family:
        for seed in args.seeds:
            data = make_dgp(delta_phi, seed, args.n, args.t)
            struct = H.score(data, H.fit_structural(data, delta_phi))
            struct_rows.append({"delta_phi": delta_phi, "seed": seed, **metrics_row("structural", struct)})
            print(
                f"  [delta_phi={delta_phi} seed={seed}] structural rho={struct['rho_rdoc']:.3f} "
                f"slow={struct['slow_recovery_corr']:.3f} leak={struct['slow_to_fast_leakage']:.3f}"
            )
            for arm_name, arm_flags in arms:
                flags = resolve_flags(arm_flags, args.penalty, args.anchor_weight, args.xi_low)
                row = run_one(delta_phi, arm_name, flags, seed, args.n, args.t, args, data)
                rows.append(row)
                print(
                    f"    {arm_name:<14} teacher rho={row['teacher_rho_rdoc']:.3f} slow={row['teacher_slow_recovery_corr']:.3f} "
                    f"leak={row['teacher_slow_to_fast_leakage']:.3f} comp={row['teacher_comp_rmse']:.3f} | "
                    f"student rho={row['student_rho_rdoc']:.3f} slow={row['student_slow_recovery_corr']:.3f}"
                )

    per = pd.DataFrame(rows)
    struct_df = pd.DataFrame(struct_rows)
    per.to_csv(out_dir / "stepB_per_run.csv", index=False)
    struct_df.to_csv(out_dir / "stepB_structural_reference.csv", index=False)
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "ball_py_sha256": H.file_sha256(H.BALL_PATH),
        "ball_py_mtime": datetime.fromtimestamp(H.BALL_PATH.stat().st_mtime).isoformat(timespec="seconds"),
        "delta_phi_family": list(family),
        "slow_fraction": SLOW_FRACTION,
        "seeds": list(args.seeds),
        "n": args.n,
        "arms": {name: resolve_flags(flags, args.penalty, args.anchor_weight, args.xi_low) for name, flags in arms},
        "ensemble_size": 1,
        "epochs": {"anchor_warmup": args.anchor_warmup, "kl_warmup": args.kl_warmup,
                   "teacher": args.teacher_epochs, "student": args.student_epochs},
        "penalty_weight": args.penalty,
        "alpha_anchor_weight": args.anchor_weight,
        "alpha_lasso_xi": float(SSMConfig().alpha_lasso_xi),
        "note": (
            "Targeted ablation: ensemble_size=1, reduced epochs, single/few seeds. "
            "DGP delta_ar matched to the model delta_phi at each family point. "
            "Scale to deep ensemble and multi-seed MCSE only after an arm clears the gate."
        ),
        "gate": "nonzero meaningful rho_RDoC, slow recovery approaching structural, low leakage, no catastrophic comp_rmse",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n=== summary: teacher rho_RDoC by arm and delta_phi (structural ref in []) ===")
    for delta_phi in family:
        ref = struct_df[struct_df["delta_phi"] == delta_phi]["structural_rho_rdoc"].mean()
        cells = []
        for arm_name, _ in arms:
            v = per[(per["delta_phi"] == delta_phi) & (per["arm"] == arm_name)]["teacher_rho_rdoc"].mean()
            cells.append(f"{arm_name}={v:.3f}")
        print(f"  delta_phi={delta_phi} [structural={ref:.3f}]: " + "  ".join(cells))
    print(f"\nwrote:\n  {out_dir / 'stepB_per_run.csv'}\n  {out_dir / 'stepB_structural_reference.csv'}\n  {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
