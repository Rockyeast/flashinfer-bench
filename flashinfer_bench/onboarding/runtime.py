"""Shared run configuration and Modal execution helpers for onboarding stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from flashinfer_bench.onboarding.definition_review import render_definition_review
from flashinfer_bench.onboarding.planning import build_stage_plan
from flashinfer_bench.onboarding.runners.modal_client import run_modal_stage
from flashinfer_bench.tracing.flashinfer_logging import (
    infer_sglang_pass_settings,
    load_fi_definition_files,
)

DEFAULT_IMAGE = "lmsysorg/sglang:v0.5.12.post1"
RUN_CONFIG_KEYS = {
    "model_name",
    "image",
    "gpu",
    "tp_size",
    "timeout",
    "batch_sizes",
    "max_new_tokens",
    "supplemental_runs",
    "disable_cuda_graph",
    "enable_piecewise_cuda_graph",
    "force_flashinfer_backends",
    "mem_fraction_static",
    "cuda_graph_max_bs",
    "engine_kwargs",
    "max_new_workloads",
    "compare_sglang_logger",
    "sglang_logger_layers",
    "isl",
    "osl",
    "random_range_ratio",
    "seed",
}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower() or "run"


def run_dir(run: str) -> Path:
    path = Path(run)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(f"ERROR: invalid run name: {run}")
    return Path("runs").joinpath(*(_slug(part) for part in path.parts))


def load_config(run_path: Path) -> dict[str, Any]:
    path = run_path / "config" / "run_config.json"
    if not path.exists():
        return {}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid run config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"ERROR: run config must be a JSON object: {path}")
    unknown = sorted(set(config) - RUN_CONFIG_KEYS)
    if unknown:
        raise SystemExit(f"ERROR: unsupported run_config.json fields: {unknown}")
    return config


def _runtime_value(
    args: argparse.Namespace, config: dict[str, Any], key: str, default: Any = None
) -> Any:
    value = getattr(args, key, None)
    return value if value is not None else config.get(key, default)


def _required_runtime_value(
    args: argparse.Namespace, config: dict[str, Any], key: str
) -> Any:
    value = _runtime_value(args, config, key)
    if value is None or value == "":
        raise SystemExit(
            f"ERROR: --{key.replace('_', '-')} is required or must be set in config/run_config.json"
        )
    return value


def resolve_config(args: argparse.Namespace, run_path: Path) -> dict[str, Any]:
    config = load_config(run_path)
    resolved = dict(config)
    resolved.update(
        {
            "model_name": _required_runtime_value(args, config, "model_name"),
            "image": _runtime_value(args, config, "image", DEFAULT_IMAGE),
            "gpu": _required_runtime_value(args, config, "gpu"),
            "tp_size": int(_runtime_value(args, config, "tp_size", 1)),
            "timeout": int(_runtime_value(args, config, "timeout", 3600)),
            "batch_sizes": config.get("batch_sizes", [1, 2, 4, 8]),
            "max_new_tokens": int(config.get("max_new_tokens", 16)),
            "supplemental_runs": config.get("supplemental_runs", []),
            "max_new_workloads": int(config.get("max_new_workloads", 20)),
            "isl": int(_runtime_value(args, config, "isl", 1024)),
            "osl": int(_runtime_value(args, config, "osl", 8)),
            "random_range_ratio": float(
                _runtime_value(args, config, "random_range_ratio", 1.0)
            ),
            "seed": int(_runtime_value(args, config, "seed", 0)),
            "compare_sglang_logger": bool(
                _runtime_value(args, config, "compare_sglang_logger", True)
            ),
        }
    )
    logger_layers = _runtime_value(args, config, "sglang_logger_layers")
    if logger_layers is not None:
        resolved["sglang_logger_layers"] = logger_layers
    if resolved["tp_size"] < 1:
        raise SystemExit("ERROR: tp_size must be at least 1")
    path = run_path / "config" / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return resolved


def plan_for_stage(
    *,
    stage: str,
    run_path: Path,
    config: dict[str, Any],
    reviewed_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pass_modes = None
    page_sizes = None
    if reviewed_definitions is not None:
        files, _ = load_fi_definition_files(run_path / "definitions")
        pass_modes, page_sizes = infer_sglang_pass_settings(files)
    plan = build_stage_plan(
        stage=stage,
        config=config,
        reviewed_definitions=reviewed_definitions,
        pass_modes=pass_modes,
        page_sizes=page_sizes,
    )
    if reviewed_definitions is not None:
        plan["definitions_sha256"] = definitions_digest(run_path / "definitions")
    return plan


def load_definition_artifacts(definitions_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(definitions_dir.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        artifacts.append({"path": str(path.relative_to(definitions_dir)), "data": data})
    return artifacts


def definitions_digest(definitions_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(definitions_dir.rglob("*.json")):
        digest.update(str(path.relative_to(definitions_dir)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_modal(
    plan: dict[str, Any],
    run_path: Path,
    config: dict[str, Any],
    resume_call_id: str | None,
) -> dict[str, Any]:
    temporary_dir = run_path / ".modal_tmp"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    result = run_modal_stage(
        stage_plan=plan,
        output_dir=temporary_dir,
        timeout=int(config["timeout"]),
        resume_call_id=resume_call_id,
    )
    shutil.rmtree(temporary_dir, ignore_errors=True)
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_definition_review(run_path: Path, report: dict[str, Any]) -> None:
    reports_dir = run_path / "reports"
    logger_report_path = reports_dir / "evidence" / "sglang_logger.json"
    if logger_report_path.exists():
        logger_report = json.loads(logger_report_path.read_text(encoding="utf-8"))
        comparison = logger_report.get("comparison") if isinstance(logger_report, dict) else None
        if isinstance(comparison, dict):
            report["sglang_logger_comparison"] = comparison
    write_json(reports_dir / "definition_report.json", report)
    (reports_dir / "definition_review.md").write_text(
        render_definition_review(report), encoding="utf-8"
    )


def sglang_inventory(run_path: Path) -> set[str] | None:
    path = run_path / "reports" / "evidence" / "sglang_execution_inventory.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    modules = value.get("modules") if isinstance(value, dict) else None
    if not isinstance(modules, list):
        return None
    return {
        str(item["class_path"])
        for item in modules
        if isinstance(item, dict) and isinstance(item.get("class_path"), str)
    }
