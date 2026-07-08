"""Local Modal submission and result materialization."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from flashinfer_bench.onboarding.runners.remote_runner import DEFAULT_REMOTE_OUTPUT_DIR


def run_modal_probe_entrypoint(
    *,
    plan_path: str,
    output_dir: str,
    remote_function: Any,
    resume_call_id: str = "",
) -> None:
    """Run the local Modal CLI entrypoint and materialize its remote result."""
    import modal

    plan_file = Path(plan_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))

    print(f"[flashinfer_bench.onboarding] Modal CLI plan: {plan_file}", flush=True)
    print(f"[flashinfer_bench.onboarding] Modal CLI output: {out_dir}", flush=True)
    print(
        "[flashinfer_bench.onboarding] Modal CLI runtime: "
        f"image={os.environ.get('FLASHINFER_BENCH_MODAL_IMAGE', 'lmsysorg/sglang:v0.5.12.post1')} "
        f"gpu={os.environ.get('FLASHINFER_BENCH_MODAL_GPU') or 'none'} "
        f"timeout={os.environ.get('FLASHINFER_BENCH_MODAL_TIMEOUT', '3600')}s",
        flush=True,
    )
    if resume_call_id:
        print(f"[flashinfer_bench.onboarding] Reattaching Modal call: {resume_call_id}", flush=True)
        call = modal.FunctionCall.from_id(resume_call_id)
        function_call_id = resume_call_id
    else:
        call = remote_function.spawn(plan)
        function_call_id = str(call.object_id)
        print(f"[flashinfer_bench.onboarding] Modal function_call_id: {function_call_id}", flush=True)
        print(
            "[flashinfer_bench.onboarding] Resume command: "
            f"modal run -m flashinfer_bench.onboarding.runners.modal_app::probe "
            f"--plan-path {plan_file} --output-dir {out_dir} --resume-call-id {function_call_id}",
            flush=True,
        )

    metadata = {
        "status": "submitted",
        "function_call_id": function_call_id,
        "plan_path": str(plan_file),
        "output_dir": str(out_dir),
        "resume_command": (
            f"modal run -m flashinfer_bench.onboarding.runners.modal_app::probe "
            f"--plan-path {plan_file} --output-dir {out_dir} --resume-call-id {function_call_id}"
        ),
    }
    (out_dir / "probe_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = call.get()
    metadata["status"] = "completed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    materialize_modal_result(result, out_dir)

def run_modal_probe(
    *,
    modal_probe_plan: dict[str, Any],
    output_dir: Path,
    timeout: int = 3600,
    resume_call_id: str | None = None,
) -> dict[str, Any]:
    """Launch the remote Modal probe using ``modal run`` for streaming logs."""
    runtime = modal_probe_plan.get("runtime", {})
    image_name = str(runtime.get("image") or "lmsysorg/sglang:v0.5.12.post1")
    gpu = str(runtime.get("gpu") or "")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "modal_probe_plan.json"
    plan_path.write_text(json.dumps(modal_probe_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["FLASHINFER_BENCH_MODAL_IMAGE"] = image_name
    env["FLASHINFER_BENCH_MODAL_TIMEOUT"] = str(timeout)
    if gpu:
        env["FLASHINFER_BENCH_MODAL_GPU"] = gpu
    else:
        env.pop("FLASHINFER_BENCH_MODAL_GPU", None)

    cmd = [
        "modal",
        "run",
        "-m",
        "flashinfer_bench.onboarding.runners.modal_app::probe",
        "--plan-path",
        str(plan_path),
        "--output-dir",
        str(output_dir),
    ]
    if resume_call_id:
        cmd.extend(["--resume-call-id", resume_call_id])
    print(
        "[flashinfer_bench.onboarding] launching Modal CLI probe: "
        f"image={image_name} gpu={gpu or 'none'} timeout={timeout}s",
        flush=True,
    )
    print(f"[flashinfer_bench.onboarding] command: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)

    result_path = output_dir / "modal_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Modal probe did not write result: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Result materialization (local/result-IO side of the Modal probe)
# ---------------------------------------------------------------------------
#
# Turn the result dict returned by the remote entrypoint into local reports and
# remote-collect outputs on disk.


def materialize_modal_result(result: dict[str, Any], output_dir: Path) -> None:
    """Materialize returned Modal probe artifacts into the run directory.

    ``output_dir`` is a transient Modal CLI handoff directory. Standard run
    artifacts are written under the parent run's ``output`` and ``reports``
    directories; raw Modal handoff files are not part of the public run layout.
    """
    print("[flashinfer_bench.onboarding] remote result received; materializing local outputs", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir.parent
    shutil.rmtree(output_dir / "captures", ignore_errors=True)
    output_root = run_dir / "output"
    reports_dir = run_dir / "reports"
    definitions_dir = output_root / "definitions"
    shutil.rmtree(reports_dir / "generated_definition_hints", ignore_errors=True)
    (reports_dir / "generated_definition_hints.tar.gz").unlink(missing_ok=True)
    _materialize_definition_outputs(definitions_dir, result)
    _summarize_generated_definition_hints(result)
    _materialize_collect_outputs(
        result,
        collect_dir=output_dir / "collect",
        output_root=output_root,
        definitions_dir=definitions_dir,
    )
    result_path = output_dir / "modal_result.json"
    result_path.write_text(
        json.dumps(_redact_modal_result(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _redact_modal_result(result: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(result)
    for key in (
        "collect_archive_b64",
        "definitions_archive_b64",
    ):
        value = redacted.get(key)
        if isinstance(value, str):
            redacted[key] = {
                "redacted": True,
                "encoded_bytes": len(value),
            }
    return redacted


def _materialize_collect_outputs(
    result: dict[str, Any],
    *,
    collect_dir: Path,
    output_root: Path,
    definitions_dir: Path,
) -> None:
    archive_b64 = result.get("collect_archive_b64")
    if not isinstance(archive_b64, str) or not archive_b64:
        return

    _write_named_archive(root=collect_dir, archive_b64=archive_b64, expected_root="collect")

    # The public output tree is a snapshot of this run, not an incremental cache.
    # Clear stale workloads before moving the audited collect results into place.
    shutil.rmtree(output_root / "workloads", ignore_errors=True)
    shutil.rmtree(output_root / "blob", ignore_errors=True)

    src_workloads = collect_dir / "workloads"
    dst_workloads = output_root / "workloads"
    if src_workloads.exists():
        for src_op_dir in src_workloads.iterdir():
            if not src_op_dir.is_dir():
                continue
            dst_op_dir = dst_workloads / src_op_dir.name
            dst_op_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_op_dir.iterdir():
                shutil.move(str(src_file), str(dst_op_dir / src_file.name))

    src_blob = collect_dir / "blob"
    dst_blob = output_root / "blob"
    if src_blob.exists():
        for src_op_dir in src_blob.glob("workloads/*"):
            if not src_op_dir.is_dir():
                continue
            dst_op_dir = dst_blob / "workloads" / src_op_dir.name
            dst_op_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_op_dir.iterdir():
                shutil.move(str(src_file), str(dst_op_dir / src_file.name))

    rewrite_collect_output_paths(
        collect_dir,
        local_collect_dir=output_root,
        local_definitions_dir=definitions_dir,
    )
    plan_path = collect_dir / "collect_plan.json"
    manifest_path = collect_dir / "workload_manifest.json"
    if plan_path.exists():
        result["collect_plan"] = json.loads(plan_path.read_text(encoding="utf-8"))
    if manifest_path.exists():
        result["workload_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    shutil.rmtree(collect_dir, ignore_errors=True)


def _materialize_definition_outputs(root: Path, result: dict[str, Any]) -> None:
    archive_b64 = result.get("definitions_archive_b64")
    if isinstance(archive_b64, str) and archive_b64:
        _write_named_archive(root=root, archive_b64=archive_b64, expected_root="definitions")


def _summarize_generated_definition_hints(result: dict[str, Any]) -> None:
    report = result.get("definition_audit_report")
    if not isinstance(report, dict):
        return

    for section in ("passed", "repaired"):
        items = report.get(section)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.pop("hints_path", None):
                item["hints_generated"] = True


def rewrite_collect_output_paths(
    collect_dir: Path,
    *,
    local_collect_dir: Path,
    local_definitions_dir: Path | None = None,
) -> None:
    """Rewrite remote absolute paths in collect outputs to local extracted paths."""
    for path in (collect_dir / "workload_manifest.json", collect_dir / "collect_plan.json"):
        _rewrite_collect_json_paths(
            path,
            local_collect_dir=local_collect_dir,
            local_definitions_dir=local_definitions_dir,
        )


def _rewrite_collect_json_paths(
    path: Path,
    *,
    local_collect_dir: Path,
    local_definitions_dir: Path | None,
) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    remote_collect_prefix = f"{DEFAULT_REMOTE_OUTPUT_DIR}/collect/"
    remote_definitions_prefix = f"{DEFAULT_REMOTE_OUTPUT_DIR}/definitions/"
    remote_audited_definitions_prefix = f"{DEFAULT_REMOTE_OUTPUT_DIR}/audited_definitions/"

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith(remote_collect_prefix):
                return str(local_collect_dir / value.removeprefix(remote_collect_prefix))
            if local_definitions_dir is not None and value.startswith(remote_definitions_prefix):
                return str(local_definitions_dir / value.removeprefix(remote_definitions_prefix))
            if local_definitions_dir is not None and value.startswith(remote_audited_definitions_prefix):
                return str(local_definitions_dir / value.removeprefix(remote_audited_definitions_prefix))
            return value
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    rewritten = rewrite(payload)
    path.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_named_archive(*, root: Path, archive_b64: str, expected_root: str) -> None:
    root_parent = root.parent
    root_parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(root, ignore_errors=True)
    archive_path = root_parent / f"{expected_root}.tar.gz"
    archive_path.write_bytes(base64.b64decode(archive_b64.encode("ascii")))
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = root_parent / member.name
            if not member_path.resolve().is_relative_to(root_parent.resolve()):
                raise ValueError(f"unsafe archive member: {member.name}")
        try:
            archive.extractall(root_parent, filter="data")
        except TypeError:
            archive.extractall(root_parent)
    archive_path.unlink(missing_ok=True)
