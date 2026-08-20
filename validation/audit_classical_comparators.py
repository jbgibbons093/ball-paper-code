'''Check convergence and persistence selection for the classical models.'''

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


VALIDATION_DIR = Path(__file__).resolve().parent
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import direct_rdoc_benchmark as benchmark  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit convergence and persistence boundaries for LGSSM and Markov comparators."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1729, 2027, 2028])
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--t", type=int, default=84)
    parser.add_argument("--share", type=float, default=0.10)
    parser.add_argument("--cell", default="linear")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    source = json.loads(cli.manifest.read_text(encoding="utf-8"))
    settings = dict(source["args"])
    settings.update({"n": cli.n, "t": cli.t, "s0_basis": "linear"})
    rows: list[dict[str, float | int | str]] = []
    candidate_rows: list[dict[str, float | int | str]] = []
    cli.out.mkdir(parents=True, exist_ok=True)
    result_path = cli.out / "classical_comparator_audit.csv"
    candidate_path = cli.out / "classical_persistence_candidates.csv"
    audit_manifest_path = cli.out / "audit_manifest.json"
    audit_manifest = {
        "status": "running",
        "started": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(cli.manifest.resolve()),
        "source_manifest_sha256": _sha256(cli.manifest),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "benchmark_script_sha256": _sha256(VALIDATION_DIR / "direct_rdoc_benchmark.py"),
        "classical_comparator_script_sha256": _sha256(
            VALIDATION_DIR / "direct_rdoc_fair_comparator.py"
        ),
        "seeds": cli.seeds,
        "n": cli.n,
        "t": cli.t,
        "share": cli.share,
        "cell": cli.cell,
    }
    audit_manifest_path.write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")

    for seed in cli.seeds:
        generation_args = argparse.Namespace(**settings)
        data = benchmark.make_data(generation_args, seed, cli.share, cli.cell)
        jobs = [
            (method, iterations, "base", [0.1, 0.3, 0.6, 0.9])
            for iterations in (10, 25, 50)
            for method in ("s0", "markov")
        ]
        jobs.extend(
            (method, 25, "expanded", [0.1, 0.3, 0.6, 0.9, 0.95, 0.98, 0.99])
            for method in ("s0", "markov")
        )

        for method, iterations, grid_name, persistence_grid in jobs:
            fit_args = argparse.Namespace(**settings)
            fit_args.iters = iterations
            fit_args.markov_iters = iterations
            fit_args.classical_min_iterations = min(3, iterations)
            fit_args.classical_convergence_tol = 1e-4
            fit_args.classical_innovation_floor = 0.05
            fit_args.classical_min_stratum_validation_patients = 3
            fit_args.classical_markov_selection_sweeps = 3
            fit_args.classical_persistence_grid = persistence_grid
            started = time.perf_counter()
            if method == "s0":
                predictions, _ = benchmark.fit_direct_map(data, fit_args)
            else:
                predictions, _, _ = benchmark.fit_markov_direct(
                    data, fit_args, basis="linear"
                )
            exact_persistence = predictions.attrs["stratum_persistence"]
            depression_values = list(exact_persistence["depression"].values())
            anxiety_values = list(exact_persistence["anxiety"].values())
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "iterations": iterations,
                    "persistence_grid": grid_name,
                    "latent_rmse": benchmark._channel_latent_rmse(
                        data, predictions, "L_hat", "L"
                    ),
                    "depression_latent_rmse": benchmark._channel_latent_rmse(
                        data, predictions, "z_d_hat", "z_d"
                    ),
                    "anxiety_latent_rmse": benchmark._channel_latent_rmse(
                        data, predictions, "z_p_hat", "z_p"
                    ),
                    "questionnaire_rmse": benchmark._measurement_anchor_rmse(
                        data, predictions
                    ),
                    "validation_selection_rmse": predictions.attrs[
                        "validation_selection_rmse"
                    ],
                    "depression_persistence_mean": sum(depression_values) / len(depression_values),
                    "depression_persistence_min": min(depression_values),
                    "depression_persistence_max": max(depression_values),
                    "anxiety_persistence_mean": sum(anxiety_values) / len(anxiety_values),
                    "anxiety_persistence_min": min(anxiety_values),
                    "anxiety_persistence_max": max(anxiety_values),
                    "stratum_persistence": json.dumps(exact_persistence, sort_keys=True),
                    "depression_innovation_sd": predictions.attrs[
                        "channel_innovation_sd"
                    ]["depression"],
                    "anxiety_innovation_sd": predictions.attrs[
                        "channel_innovation_sd"
                    ]["anxiety"],
                    "training_transition_mse": predictions.attrs[
                        "training_transition_mse"
                    ],
                    "converged": predictions.attrs["converged"],
                    "persistence_selection_converged": predictions.attrs[
                        "persistence_selection_converged"
                    ],
                    "iterations_completed": predictions.attrs["iterations_completed"],
                    "max_convergence_change": predictions.attrs[
                        "max_convergence_change"
                    ],
                    "innovation_floor_hit": bool(
                        min(
                            predictions.attrs["channel_innovation_sd"]["depression"],
                            predictions.attrs["channel_innovation_sd"]["anxiety"],
                        )
                        <= predictions.attrs["innovation_floor"] + 1e-12
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            for candidate in predictions.attrs["persistence_candidate_trace"]:
                candidate_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "maximum_iterations": iterations,
                        "persistence_grid": grid_name,
                        **candidate,
                    }
                )
            pd.DataFrame(rows).to_csv(result_path, index=False)
            pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)
            print(
                f"completed seed={seed} method={method} iterations={iterations} "
                f"grid={grid_name}",
                flush=True,
            )

    print(f"wrote {result_path}", flush=True)
    print(f"wrote {candidate_path}", flush=True)
    audit_manifest.update(
        {
            "status": "complete",
            "completed": datetime.now().isoformat(timespec="seconds"),
            "result_rows": len(rows),
            "candidate_rows": len(candidate_rows),
            "result_sha256": _sha256(result_path),
            "candidate_sha256": _sha256(candidate_path),
        }
    )
    audit_manifest_path.write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")
    print(f"wrote {audit_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
