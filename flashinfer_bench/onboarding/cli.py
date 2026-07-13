#!/usr/bin/env python3
"""Two-stage CLI for model definition and workload onboarding."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from flashinfer_bench.onboarding.agent_analysis import run_definition_analysis
from flashinfer_bench.onboarding.definition_review import (
    check_definition_directory,
    organize_definition_directory,
    render_definition_review,
    render_workload_review,
)
from flashinfer_bench.onboarding.planning import INFERENCEX_PROFILES, build_stage_plan
from flashinfer_bench.onboarding.runners.modal_client import run_modal_stage
from flashinfer_bench.onboarding.validation import run_dataset_validator
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
    "inferencex_profiles",
    "inferencex_range_ratio",
    "inferencex_seed",
}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower() or "run"


def _run_dir(run: str) -> Path:
    path = Path(run)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(f"ERROR: invalid run name: {run}")
    return Path("runs").joinpath(*(_slug(part) for part in path.parts))


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config" / "run_config.json"
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


def _required_runtime_value(args: argparse.Namespace, config: dict[str, Any], key: str) -> Any:
    value = _runtime_value(args, config, key)
    if value is None or value == "":
        raise SystemExit(
            f"ERROR: --{key.replace('_', '-')} is required or must be set in config/run_config.json"
        )
    return value


def _resolved_config(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    config = _load_config(run_dir)
    resolved = dict(config)
    resolved.update(
        {
            "model_name": _required_runtime_value(args, config, "model_name"),
            "image": _runtime_value(args, config, "image", DEFAULT_IMAGE),
            "gpu": _required_runtime_value(args, config, "gpu"),
            "tp_size": int(_runtime_value(args, config, "tp_size", 1)),
            "timeout": int(_runtime_value(args, config, "timeout", 3600)),
            "batch_sizes": config.get("batch_sizes", [1]),
            "max_new_tokens": int(config.get("max_new_tokens", 16)),
            "supplemental_runs": config.get("supplemental_runs", []),
            "max_new_workloads": int(config.get("max_new_workloads", 20)),
            "compare_sglang_logger": bool(
                _runtime_value(args, config, "compare_sglang_logger", True)
            ),
        }
    )
    inferencex_profiles = _runtime_value(args, config, "inferencex_profiles")
    if inferencex_profiles is not None:
        resolved["inferencex_profiles"] = inferencex_profiles
        resolved["inferencex_range_ratio"] = float(
            _runtime_value(args, config, "inferencex_range_ratio", 0.8)
        )
        resolved["inferencex_seed"] = int(_runtime_value(args, config, "inferencex_seed", 0))
    if resolved["tp_size"] < 1:
        raise SystemExit("ERROR: tp_size must be at least 1")
    path = run_dir / "config" / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def _plan_for_stage(
    *,
    stage: str,
    run_dir: Path,
    config: dict[str, Any],
    reviewed_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pass_modes = None
    page_sizes = None
    if reviewed_definitions is not None:
        files, _ = load_fi_definition_files(run_dir / "definitions")
        pass_modes, page_sizes = infer_sglang_pass_settings(files)
    plan = build_stage_plan(
        stage=stage,
        config=config,
        reviewed_definitions=reviewed_definitions,
        pass_modes=pass_modes,
        page_sizes=page_sizes,
    )
    if reviewed_definitions is not None:
        plan["definitions_sha256"] = _definitions_digest(run_dir / "definitions")
    return plan


def _load_definition_artifacts(definitions_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(definitions_dir.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        artifacts.append({"path": str(path.relative_to(definitions_dir)), "data": data})
    return artifacts


def _definitions_digest(definitions_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(definitions_dir.rglob("*.json")):
        digest.update(str(path.relative_to(definitions_dir)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_modal(
    plan: dict[str, Any], run_dir: Path, config: dict[str, Any], resume_call_id: str | None
) -> dict[str, Any]:
    temporary_dir = run_dir / ".modal_tmp"
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_definition_review(run_dir: Path, report: dict[str, Any]) -> None:
    reports_dir = run_dir / "reports"
    _write_json(reports_dir / "definition_report.json", report)
    (reports_dir / "definition_review.md").write_text(
        render_definition_review(report), encoding="utf-8"
    )


def _sglang_inventory(run_dir: Path) -> set[str] | None:
    path = run_dir / "reports" / "sglang_modules.json"
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


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run path below runs/.")
    parser.add_argument("--model-name")
    parser.add_argument("--image")
    parser.add_argument("--gpu")
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--resume-call-id")
    parser.add_argument(
        "--agent",
        choices=["codex"],
        help="Analyze runtime/source evidence and write reviewed definitions in place.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-stage model onboarding")
    subparsers = parser.add_subparsers(dest="command", required=True)

    definitions = subparsers.add_parser(
        "dump-definition",
        help="Run a short FlashInfer trace pass and write editable definition templates.",
    )
    _add_runtime_args(definitions)
    definitions.add_argument(
        "--overwrite", action="store_true", help="Replace an existing definitions/ directory."
    )
    definitions.add_argument(
        "--compare-sglang-logger",
        action="store_true",
        default=None,
        help="Enable SGLang's optional output-only tensor logger comparison.",
    )

    analysis = subparsers.add_parser(
        "analyze-definition",
        help="Re-run local Agent definition analysis from existing runtime evidence.",
    )
    analysis.add_argument("--run", required=True, help="Run path below runs/.")
    analysis.add_argument("--agent", choices=["codex"], default="codex")

    workloads = subparsers.add_parser(
        "dump-workload",
        help="Collect workloads from the reviewed definitions/ snapshot and validate the dataset.",
    )
    _add_runtime_args(workloads)
    workloads.add_argument("--max-new-workloads", type=int)
    workloads.add_argument(
        "--inferencex-profile",
        dest="inferencex_profiles",
        action="append",
        choices=sorted(INFERENCEX_PROFILES),
        help="Use an InferenceX fixed-sequence request profile; repeat to run both profiles.",
    )
    workloads.add_argument("--inferencex-range-ratio", type=float)
    workloads.add_argument("--inferencex-seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = _run_dir(args.run)

    if args.command == "analyze-definition":
        config = _load_config(run_dir)
        model_name = config.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise SystemExit("ERROR: config/run_config.json must contain model_name")
        definitions_dir = run_dir / "definitions"
        inventory = _sglang_inventory(run_dir)
        if inventory is None:
            raise SystemExit("ERROR: reports/sglang_modules.json is required for local analysis")
        report = check_definition_directory(
            definitions_dir, sglang_inventory=inventory, model_name=model_name
        )
        report_path = run_dir / "reports" / "definition_report.json"
        if report_path.exists():
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(existing_report, dict) and isinstance(
                existing_report.get("remote"), dict
            ):
                report["remote"] = existing_report["remote"]
        before = _definitions_digest(definitions_dir)
        returncode = run_definition_analysis(
            run_dir=run_dir, model_name=model_name, report=report, source="definition_reanalysis"
        )
        if returncode == 0:
            organize_definition_directory(definitions_dir)
            remote = report.get("remote")
            report = check_definition_directory(
                definitions_dir, sglang_inventory=inventory, model_name=model_name
            )
            if isinstance(remote, dict):
                report["remote"] = remote
        report["agent_analysis"] = {
            "returncode": returncode,
            "definitions_changed": _definitions_digest(definitions_dir) != before,
        }
        _write_definition_review(run_dir, report)
        print(f"definitions: {definitions_dir}")
        print(f"definition review: {run_dir / 'reports' / 'definition_review.md'}")
        print(f"schema ready: {report['summary']['ok']}")
        return 0 if returncode == 0 and report["summary"]["ok"] else 1

    if args.command == "dump-definition":
        definitions_dir = run_dir / "definitions"
        if definitions_dir.exists() and any(definitions_dir.rglob("*.json")) and not args.overwrite:
            raise SystemExit(
                f"ERROR: {definitions_dir} already contains reviewed files; pass --overwrite to replace them"
            )
        config = _resolved_config(args, run_dir)
        plan = _plan_for_stage(stage="definitions", run_dir=run_dir, config=config)
        result = _run_modal(plan, run_dir, config, args.resume_call_id)
        reports_dir = run_dir / "reports"
        module_inventory = result.get("module_inventory")
        if isinstance(module_inventory, dict):
            _write_json(reports_dir / "sglang_modules.json", module_inventory)
        sglang_logger = result.get("sglang_logger")
        if isinstance(sglang_logger, dict):
            _write_json(reports_dir / "sglang_logger_report.json", sglang_logger)
        organize_definition_directory(definitions_dir)
        inventory = _sglang_inventory(run_dir)
        report = check_definition_directory(
            definitions_dir,
            sglang_inventory=inventory,
            model_name=str(config["model_name"]),
        )
        report["remote"] = result.get("summary", {})
        if args.agent:
            before = _definitions_digest(definitions_dir)
            returncode = run_definition_analysis(
                run_dir=run_dir,
                model_name=str(config["model_name"]),
                report=report,
                source="definition_authoring",
            )
            if returncode == 0:
                organize_definition_directory(definitions_dir)
                report = check_definition_directory(
                    definitions_dir,
                    sglang_inventory=inventory,
                    model_name=str(config["model_name"]),
                )
                report["remote"] = result.get("summary", {})
            report["agent_analysis"] = {
                "returncode": returncode,
                "definitions_changed": _definitions_digest(definitions_dir) != before,
            }
        _write_definition_review(run_dir, report)
        print(f"definitions: {definitions_dir}")
        print(f"definition review: {run_dir / 'reports' / 'definition_review.md'}")
        print(f"schema ready: {report['summary']['ok']}")
        return 0

    if args.command == "dump-workload":
        config = _resolved_config(args, run_dir)
        if args.max_new_workloads is not None:
            if args.max_new_workloads < 1:
                raise SystemExit("ERROR: --max-new-workloads must be at least 1")
            config["max_new_workloads"] = args.max_new_workloads
            _write_json(run_dir / "config" / "run_config.json", config)

        definitions_dir = run_dir / "definitions"
        inventory = _sglang_inventory(run_dir)
        definition_report = check_definition_directory(
            definitions_dir,
            sglang_inventory=inventory,
            model_name=str(config["model_name"]),
        )
        if not definition_report["summary"]["ok"] and args.agent:
            returncode = run_definition_analysis(
                run_dir=run_dir,
                model_name=str(config["model_name"]),
                report=definition_report,
                source="pre_workload_definition_analysis",
            )
            if returncode == 0:
                organize_definition_directory(definitions_dir)
                definition_report = check_definition_directory(
                    definitions_dir,
                    sglang_inventory=inventory,
                    model_name=str(config["model_name"]),
                )
        _write_definition_review(run_dir, definition_report)
        if not definition_report["summary"]["ok"]:
            raise SystemExit(
                "ERROR: definitions failed review; fix reports/definition_review.md before dump-workload"
            )
        if definition_report["summary"]["collectable"] == 0:
            raise SystemExit("ERROR: no reviewed definition contains a supported capture tag")

        artifacts = _load_definition_artifacts(definitions_dir)
        plan = _plan_for_stage(
            stage="workloads", run_dir=run_dir, config=config, reviewed_definitions=artifacts
        )
        result = _run_modal(plan, run_dir, config, args.resume_call_id)
        workload_report = result.get("workload_report")
        if not isinstance(workload_report, dict):
            raise SystemExit("ERROR: remote workload stage returned no workload report")
        current_digest = _definitions_digest(definitions_dir)
        if (
            result.get("definitions_sha256") != plan["definitions_sha256"]
            or current_digest != plan["definitions_sha256"]
        ):
            raise SystemExit(
                "ERROR: definitions changed while dump-workload was running; rerun with the reviewed snapshot"
            )

        names = [
            str(item["name"])
            for item in workload_report.get("definitions", [])
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and int(item.get("workloads") or 0) > 0
        ]
        dataset_validation = run_dataset_validator(
            dataset_dir=run_dir / "output",
            checks="layout,definition,workload",
            outputs="stdout",
            disable_gpu=True,
            definitions=names,
        )
        accepted = bool(workload_report["summary"]["ok"] and dataset_validation.get("ok"))
        report = {
            "schema_version": 1,
            "model_name": config["model_name"],
            "definitions_sha256": current_digest,
            "workload": workload_report,
            "dataset_validation": dataset_validation,
            "accepted": accepted,
        }
        reports_dir = run_dir / "reports"
        analysis = None
        if not accepted and args.agent:
            returncode = run_definition_analysis(
                run_dir=run_dir,
                model_name=str(config["model_name"]),
                report=report,
                source="workload_result_analysis",
            )
            analyzed_digest = _definitions_digest(definitions_dir)
            refreshed = check_definition_directory(
                definitions_dir,
                sglang_inventory=inventory,
                model_name=str(config["model_name"]),
            )
            _write_definition_review(run_dir, refreshed)
            analysis = {
                "returncode": returncode,
                "definitions_changed": analyzed_digest != current_digest,
                "definitions_sha256": analyzed_digest,
                "requires_review_and_rerun": analyzed_digest != current_digest,
            }
            report["agent_analysis"] = analysis
            report["current_definitions_sha256"] = analyzed_digest
        else:
            report["current_definitions_sha256"] = current_digest
        _write_json(reports_dir / "run_report.json", report)
        (reports_dir / "review.md").write_text(
            render_workload_review(workload_report, dataset_validation, analysis), encoding="utf-8"
        )
        print(f"workloads: {workload_report['summary']['workloads']}")
        print(f"run report: {reports_dir / 'run_report.json'}")
        print(f"review: {reports_dir / 'review.md'}")
        print(f"run accepted: {accepted}")
        if analysis and analysis["requires_review_and_rerun"]:
            print("definitions changed: review definitions/ and rerun dump-workload")
        return 0 if accepted else 1

    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
