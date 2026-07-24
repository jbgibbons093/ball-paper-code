"""Run the two publication sensitivity analyses sequentially and resumably."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
BENCHMARK = VALIDATION / "direct_rdoc_benchmark.py"
OUTPUT_ROOT = VALIDATION / "outputs"
LOG_ROOT = OUTPUT_ROOT / "publication_sensitivity_logs_20260618"
STATUS_PATH = LOG_ROOT / "status.json"

COMMON_ARGS = [
    str(BENCHMARK),
    "--cells",
    "linear",
    "interaction",
    "nonlinear",
    "heterogeneous",
    "missingness",
    "--shares",
    "0.10",
    "0.25",
    "--seeds",
    "1729",
    "2027",
    "2028",
    "2029",
    "4242",
    "9001",
    "13",
    "71",
    "137",
    "311",
    "--n",
    "150",
    "--t",
    "84",
    "--ensemble-size",
    "5",
    "--teacher-epochs",
    "300",
    "--student-epochs",
    "300",
    "--anchor-warmup",
    "120",
    "--kl-warmup",
    "80",
    "--batch-size",
    "32",
    "--device",
    "cuda",
    "--resume",
]

RUNS = [
    {
        "name": "irt",
        "output": OUTPUT_ROOT / "direct_rdoc_irt_10seed_300ep_k5",
        "extra": ["--anchor-observation", "irt"],
    },
    {
        "name": "adaptive_lasso",
        "output": OUTPUT_ROOT / "direct_rdoc_adaptive_lasso_10seed_300ep_k5",
        "extra": [
            "--rdoc-drift-l1",
            "0.1",
            "--rdoc-drift-adaptive",
            "--rdoc-drift-adaptive-gamma",
            "1.0",
            "--rdoc-drift-adaptive-eps",
            "0.001",
        ],
    },
]


def write_status(
    stage: str,
    state: str,
    output: Path,
    log_file: Path,
    exit_code: int | None = None,
) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "stage": stage,
        "state": state,
        "output_directory": str(output),
        "log_file": str(log_file),
        "exit_code": exit_code,
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
    )

    for run in RUNS:
        name = run["name"]
        output = run["output"]
        log_file = LOG_ROOT / f"{name}.log"
        command = [
            sys.executable,
            "-u",
            *COMMON_ARGS,
            *run["extra"],
            "--out",
            str(output),
        ]
        write_status(name, "running", output, log_file)
        with log_file.open("a", encoding="utf-8", buffering=1) as log:
            log.write(
                f"\n[{datetime.now().astimezone().isoformat()}] "
                f"starting: {subprocess.list2cmdline(command)}\n"
            )
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            write_status(name, "failed", output, log_file, completed.returncode)
            return completed.returncode
        write_status(name, "completed", output, log_file, 0)

    write_status("all", "completed", OUTPUT_ROOT, LOG_ROOT, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
