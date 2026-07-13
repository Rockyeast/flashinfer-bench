"""Local Modal submission and stage-result materialization."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any


def run_modal_stage_entrypoint(
    *, plan_path: str, output_dir: str, remote_function: Any, resume_call_id: str = ""
) -> None:
    """Spawn or resume one Modal call and materialize its returned archive."""
    import modal

    plan_file = Path(plan_path).resolve()
    temporary_dir = Path(output_dir).resolve()
    temporary_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if resume_call_id:
        call = modal.FunctionCall.from_id(resume_call_id)
        function_call_id = resume_call_id
    else:
        call = remote_function.spawn(plan)
        function_call_id = str(call.object_id)

    metadata = {
        "stage": plan.get("stage"),
        "status": "submitted",
        "function_call_id": function_call_id,
        "plan_path": str(plan_file),
        "output_dir": str(temporary_dir),
        "resume_command": (
            "modal run -m flashinfer_bench.onboarding.runners.modal_app::stage "
            f"--plan-path {plan_file} --output-dir {temporary_dir} --resume-call-id {function_call_id}"
        ),
    }
    _write_json(temporary_dir / "stage_run.json", metadata)
    print(f"[flashinfer_bench.onboarding] Modal function_call_id: {function_call_id}", flush=True)
    print(f"[flashinfer_bench.onboarding] resume: {metadata['resume_command']}", flush=True)
    result = call.get()
    metadata["status"] = "completed"
    _write_json(temporary_dir / "stage_run.json", metadata)
    materialize_modal_stage_result(result, temporary_dir)


def run_modal_stage(
    *,
    stage_plan: dict[str, Any],
    output_dir: Path,
    timeout: int = 3600,
    resume_call_id: str | None = None,
) -> dict[str, Any]:
    """Launch one remote stage through the Modal CLI for streaming logs."""
    runtime = stage_plan.get("runtime") if isinstance(stage_plan.get("runtime"), dict) else {}
    image = str(runtime.get("image") or "lmsysorg/sglang:v0.5.12.post1")
    gpu = str(runtime.get("gpu") or "")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "modal_stage_plan.json"
    _write_json(plan_path, stage_plan)

    environment = os.environ.copy()
    environment["FLASHINFER_BENCH_MODAL_IMAGE"] = image
    environment["FLASHINFER_BENCH_MODAL_TIMEOUT"] = str(timeout)
    if gpu:
        environment["FLASHINFER_BENCH_MODAL_GPU"] = gpu
    command = [
        "modal",
        "run",
        "-m",
        "flashinfer_bench.onboarding.runners.modal_app::stage",
        "--plan-path",
        str(plan_path),
        "--output-dir",
        str(output_dir),
    ]
    if resume_call_id:
        command.extend(["--resume-call-id", resume_call_id])
    print(
        f"[flashinfer_bench.onboarding] stage={stage_plan.get('stage')} image={image} gpu={gpu}",
        flush=True,
    )
    subprocess.run(command, check=True, env=environment)

    result_path = output_dir / "modal_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Modal stage did not write result: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def materialize_modal_stage_result(result: dict[str, Any], temporary_dir: Path) -> None:
    """Extract the stage archive into the stable run directory."""
    stage = result.get("stage")
    run_dir = temporary_dir.parent
    if stage == "definitions":
        _extract_result_archive(
            root=run_dir / "definitions",
            archive_b64=result.get("definitions_archive_b64"),
            expected_root="definitions",
        )
    elif stage == "workloads":
        _extract_result_archive(
            root=run_dir / "output",
            archive_b64=result.get("output_archive_b64"),
            expected_root="output",
        )
    else:
        raise ValueError(f"unknown Modal stage result: {stage!r}")
    _write_json(temporary_dir / "modal_result.json", _redact_archives(result))


def _extract_result_archive(*, root: Path, archive_b64: Any, expected_root: str) -> None:
    if not isinstance(archive_b64, str) or not archive_b64:
        raise ValueError(f"{expected_root} stage returned no archive")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(root, ignore_errors=True)
    archive_path = parent / f".{expected_root}.tar.gz"
    archive_path.write_bytes(base64.b64decode(archive_b64.encode("ascii")))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            parent_resolved = parent.resolve()
            for member in archive.getmembers():
                if not (parent / member.name).resolve().is_relative_to(parent_resolved):
                    raise ValueError(f"unsafe archive member: {member.name}")
            try:
                archive.extractall(parent, filter="data")
            except TypeError:
                archive.extractall(parent)
    finally:
        archive_path.unlink(missing_ok=True)


def _redact_archives(result: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(result)
    for key in ("definitions_archive_b64", "output_archive_b64"):
        value = redacted.get(key)
        if isinstance(value, str):
            redacted[key] = {"redacted": True, "encoded_bytes": len(value)}
    return redacted


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
