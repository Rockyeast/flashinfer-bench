"""Thin wrapper around the canonical FlashInfer-Bench dataset validator."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def run_dataset_validator(
    *,
    dataset_dir: Path,
    checks: str = "layout,definition,workload",
    outputs: str = "stdout",
    disable_gpu: bool = True,
    definitions: list[str] | None = None,
) -> dict[str, Any]:
    """Run the canonical dataset validator and return its machine report."""
    command = [
        sys.executable,
        "-c",
        "from flashinfer_bench.cli.main import cli; cli()",
        "validate",
        "--dataset",
        str(dataset_dir),
        "--checks",
        checks,
        "--outputs",
        outputs,
    ]
    if disable_gpu:
        command.append("--disable-gpu")
    if definitions:
        command.extend(["--definitions", *definitions])

    with tempfile.TemporaryDirectory(prefix="flashinfer-bench-dataset-validate-") as temporary:
        cache = Path(temporary) / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["XDG_CACHE_HOME"] = str(cache)
        environment["FLASHINFER_WORKSPACE_BASE"] = str(cache)
        result = subprocess.run(
            command, env=environment, text=True, capture_output=True, check=False
        )

    combined = f"{result.stdout}\n{result.stderr}".lower()
    output_has_errors = bool(re.search(r"\b[1-9]\d*\s+error\b", combined)) or any(
        marker in combined
        for marker in ("[error]", "parse error", "validation error", "cannot validate workloads")
    )
    report: dict[str, Any] = {
        "command": command,
        "dataset_dir": str(dataset_dir),
        "checks": checks,
        "outputs": outputs,
        "disable_gpu": disable_gpu,
        "definitions": definitions or [],
        "returncode": result.returncode,
        "ok": result.returncode == 0 and not output_has_errors,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if output_has_errors:
        report["error"] = "dataset validator output contains errors"
    return report
