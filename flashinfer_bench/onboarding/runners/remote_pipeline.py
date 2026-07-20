"""Remote execution for the two-stage onboarding pipeline."""

from __future__ import annotations

import base64
import json
import os
import shutil
import tarfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

from flashinfer_bench.onboarding.runners.sglang_runner import run_sglang_model
from flashinfer_bench.onboarding.submission import publication_definition
from flashinfer_bench.tracing.flashinfer_logging import (
    flashinfer_definition_dump,
    flashinfer_workload_dump,
    load_fi_definition_files,
)
from flashinfer_bench.tracing.sanitize import sanitize_dumps
from flashinfer_bench.tracing.sglang_logging import (
    compare_sglang_logger_inventory,
    load_sglang_definition_files,
    merge_sglang_workload_shards,
    sglang_capture_environment,
    summarize_module_inventory,
    summarize_sglang_capture_status,
    summarize_sglang_tensor_logger,
)

DEFAULT_REMOTE_OUTPUT_DIR = "/tmp/flashinfer-bench-onboarding"
TENSOR_DUMP_LAYOUTS = {
    "gemma-4": ("language_model", "layers"),
}


def run_remote_stage(
    plan: dict[str, Any], remote_output_dir: str = DEFAULT_REMOTE_OUTPUT_DIR
) -> dict[str, Any]:
    """Run exactly one remote stage and return its archived artifacts."""
    stage = plan.get("stage")
    if stage not in {"definitions", "workloads"}:
        raise ValueError("plan.stage must be 'definitions' or 'workloads'")
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    output_dir = Path(remote_output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FLASHINFER_TRACE_OUTPUT_DIR"] = str(output_dir)
    print(f"[flashinfer_bench.onboarding] remote stage started: {stage}", flush=True)

    if stage == "definitions":
        return _dump_definitions(plan, output_dir)
    return _dump_workloads(plan, output_dir)


def _dump_definitions(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    definitions_dir = output_dir / "definitions"
    capture_root = output_dir / "sglang_capture"
    logger_root = output_dir / "sglang_logger"
    execution_plan = dict(plan)
    sglang_config = dict(execution_plan.get("sglang") or {})
    compare_tensor_logger = bool(sglang_config.pop("compare_tensor_logger", False))
    logger_layers: list[int] = []
    if compare_tensor_logger:
        logger_layers = _representative_logger_layers(execution_plan)
    execution_plan["sglang"] = sglang_config
    with (
        flashinfer_definition_dump(definitions_dir),
        sglang_capture_environment(capture_root, mode="inventory"),
        _tensor_dump_layout(plan) if compare_tensor_logger else _tensor_dump_layout({}),
    ):
        run_sglang_model(execution_plan)
        if compare_tensor_logger:
            logger_plan = _logger_comparison_plan(
                execution_plan,
                output_dir=logger_root,
                logger_layers=logger_layers,
            )
            run_sglang_model(logger_plan)

    files = sorted(definitions_dir.rglob("*.json"))
    module_inventory = summarize_module_inventory(capture_root)
    sglang_logger = summarize_sglang_tensor_logger(logger_root)
    sglang_logger["enabled"] = compare_tensor_logger
    sglang_logger["layers"] = logger_layers
    sglang_logger["comparison"] = compare_sglang_logger_inventory(
        sglang_logger, module_inventory
    )
    if not compare_tensor_logger:
        sglang_logger["note"] = (
            "SGLang's output-only tensor logger was disabled for this run. "
            "Remove --no-compare-sglang-logger or set compare_sglang_logger=true "
            "to enable it."
        )
    shutil.rmtree(capture_root, ignore_errors=True)
    shutil.rmtree(logger_root, ignore_errors=True)
    return {
        "stage": "definitions",
        "definitions_archive_b64": _archive_dir_b64(definitions_dir, arcname="definitions"),
        "module_inventory": module_inventory,
        "sglang_logger": sglang_logger,
        "summary": {
            "definitions": len(files),
            "model_passes": 2 if compare_tensor_logger else 1,
            "sglang_modules": module_inventory["summary"]["modules"],
            "sglang_logger_operators": sglang_logger["summary"]["operators"],
            "sglang_logger_exact_identity_matches": sglang_logger["comparison"][
                "summary"
            ]["exact_identity_matched_logger_operators"],
            "sglang_logger_signature_fallback_matches": sglang_logger["comparison"][
                "summary"
            ]["signature_fallback_matched_logger_operators"],
        },
    }


def _logger_comparison_plan(
    plan: dict[str, Any], *, output_dir: Path, logger_layers: list[int]
) -> dict[str, Any]:
    """Keep SGLang's output-only logger bounded to one short representative request."""
    scenarios = plan.get("request_scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("definition stage requires request_scenarios")
    scenario = dict(scenarios[0])
    scenario.update(
        {
            "name": "sglang_logger_short_probe",
            "input_len": min(int(scenario.get("input_len") or 128), 128),
            "output_len": 1,
            "batch_size": 1,
            "range_ratio": 1.0,
        }
    )
    scenario.pop("context_fraction", None)
    scenario.pop("shared_prefix_len", None)
    logger_plan = dict(plan)
    logger_plan["request_scenarios"] = [scenario]
    logger_plan["sampling"] = {"max_new_tokens": 1}
    logger_plan["supplemental_runs"] = []
    sglang_config = dict(plan.get("sglang") or {})
    sglang_config["debug_tensor_dump_output_folder"] = str(output_dir)
    sglang_config["debug_tensor_dump_layers"] = logger_layers
    logger_plan["sglang"] = sglang_config
    return logger_plan


def _representative_logger_layers(plan: dict[str, Any]) -> list[int]:
    """Choose one logger layer per distinct HF layer type."""
    sglang_config = plan.get("sglang")
    if not isinstance(sglang_config, dict):
        sglang_config = {}
    configured = sglang_config.get("logger_layers")
    if isinstance(configured, list) and configured and all(
        type(item) is int and item >= 0 for item in configured
    ):
        return list(dict.fromkeys(configured))

    model_config = _embedded_model_config(sglang_config)
    if model_config is None:
        try:
            from transformers import AutoConfig

            loaded = AutoConfig.from_pretrained(
                str(plan.get("model_name") or ""), trust_remote_code=True
            )
            model_config = loaded.to_dict()
        except Exception as exc:
            print(
                "[flashinfer_bench.onboarding] could not inspect HF layer types; "
                f"using SGLang logger layer 0: {exc}",
                flush=True,
            )
            return [0]

    layer_types = _find_layer_types(model_config)
    if not layer_types:
        return [0]
    first_by_type: dict[str, int] = {}
    for index, layer_type in enumerate(layer_types):
        if isinstance(layer_type, str) and layer_type:
            first_by_type.setdefault(layer_type, index)
    return list(first_by_type.values()) or [0]


def _embedded_model_config(sglang_config: dict[str, Any]) -> dict[str, Any] | None:
    engine_kwargs = sglang_config.get("engine_kwargs")
    if not isinstance(engine_kwargs, dict):
        return None
    raw = engine_kwargs.get("decrypted_config_json")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _find_layer_types(value: dict[str, Any]) -> list[Any] | None:
    layer_types = value.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        return layer_types
    for child in value.values():
        if isinstance(child, dict):
            found = _find_layer_types(child)
            if found:
                return found
    return None


@contextmanager
def _tensor_dump_layout(plan: dict[str, Any]) -> Iterator[None]:
    model_name = str(plan.get("model_name") or "").lower()
    layout = next(
        (value for marker, value in TENSOR_DUMP_LAYOUTS.items() if marker in model_name),
        None,
    )
    if layout is None:
        yield
        return
    updates = {
        "TENSOR_DUMP_TOP_LEVEL_MODULE_NAME": layout[0],
        "TENSOR_DUMP_LAYERS_MODULE_NAME": layout[1],
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _dump_workloads(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    definitions_dir = output_dir / "definitions"
    _write_reviewed_definitions(definitions_dir, plan.get("reviewed_definitions"))
    fi_definition_files, _ = load_fi_definition_files(definitions_dir)
    sglang_definition_files, _ = load_sglang_definition_files(definitions_dir)
    definition_files = list(dict.fromkeys([*fi_definition_files, *sglang_definition_files]))
    skipped = _uncaptured_definitions(definitions_dir, definition_files)
    if not definition_files:
        raise RuntimeError("no reviewed definition contains a supported capture tag")

    dump_dir = output_dir / "native_dumps"
    capture_root = output_dir / "sglang_capture"
    dataset_dir = output_dir / "output"
    capture_metadata = []
    for source in definition_files:
        destination = dataset_dir / "definitions" / source.relative_to(definitions_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_value = json.loads(source.read_text(encoding="utf-8"))
        published_value, metadata = publication_definition(source_value)
        destination.write_text(
            json.dumps(published_value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        metadata["path"] = str(source.relative_to(definitions_dir))
        capture_metadata.append(metadata)

    include_pattern = ""
    with ExitStack() as stack:
        if fi_definition_files:
            include_pattern = stack.enter_context(
                flashinfer_workload_dump(fi_definition_files, dump_dir)
            )
        if sglang_definition_files:
            stack.enter_context(
                sglang_capture_environment(
                    capture_root, mode="workloads", definitions_dir=definitions_dir
                )
            )
        run_sglang_model(plan)

    max_new_workloads = int(plan.get("max_new_workloads") or 20)
    results: dict[str, list[dict[str, Any]]] = {}
    if fi_definition_files:
        results = sanitize_dumps(
            dump_dir=dump_dir,
            definition_files=fi_definition_files,
            flashinfer_trace_dir=dataset_dir,
            replace=True,
            max_new_workloads=max_new_workloads,
        )
    sglang_counts: dict[str, int] = {}
    sglang_diagnostics: dict[str, Any] = {
        "summary": {"processes": 0, "capture_calls": 0, "errors": 0},
        "captures": {},
        "errors": [],
    }
    if sglang_definition_files:
        sglang_diagnostics = summarize_sglang_capture_status(capture_root)
        sglang_counts = merge_sglang_workload_shards(
            capture_root,
            dataset_dir=dataset_dir,
            definition_files=sglang_definition_files,
            max_new_workloads=max_new_workloads,
        )
    shutil.rmtree(dump_dir, ignore_errors=True)
    shutil.rmtree(capture_root, ignore_errors=True)

    fi_names = {path.stem for path in fi_definition_files}
    sglang_names = {path.stem for path in sglang_definition_files}
    collected = [
        {
            "name": path.stem,
            "path": str(path.relative_to(definitions_dir)),
            "backend": "flashinfer" if path.stem in fi_names else "sglang",
            "workloads": (
                len(results.get(path.stem, []))
                if path.stem in fi_names
                else sglang_counts.get(path.stem, 0)
            ),
        }
        for path in definition_files
    ]
    missing = [item for item in collected if item["workloads"] == 0]
    report = {
        "summary": {
            "definitions": len(definition_files),
            "definitions_with_workloads": len(collected) - len(missing),
            "workloads": sum(item["workloads"] for item in collected),
            "missing": len(missing),
            "skipped": len(skipped),
            "ok": not missing,
        },
        "include_pattern": include_pattern,
        "definitions": collected,
        "missing": missing,
        "skipped": skipped,
        "sglang_capture": sglang_diagnostics,
        "request_profiles": _request_profile_summary(plan),
    }
    return {
        "stage": "workloads",
        "definitions_sha256": plan.get("definitions_sha256"),
        "output_archive_b64": _archive_dir_b64(dataset_dir, arcname="output"),
        "workload_report": report,
        "capture_metadata": capture_metadata,
        "summary": report["summary"],
    }


def _write_reviewed_definitions(root: Path, artifacts: Any) -> None:
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("workload stage requires reviewed_definitions")
    shutil.rmtree(root, ignore_errors=True)
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"reviewed definition #{index} must be an object")
        raw_path = artifact.get("path")
        data = artifact.get("data")
        if not isinstance(raw_path, str) or not isinstance(data, dict):
            raise ValueError(f"reviewed definition #{index} must contain path and data")
        relative = Path(raw_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"unsafe reviewed definition path: {raw_path}")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _uncaptured_definitions(root: Path, captured: list[Path]) -> list[dict[str, str]]:
    captured_paths = {path.resolve() for path in captured}
    return [
        {
            "path": str(path.relative_to(root)),
            "reason": "missing fi_api, sglang_module, or sglang_callable tag",
        }
        for path in sorted(root.rglob("*.json"))
        if path.resolve() not in captured_paths
    ]


def _request_profile_summary(plan: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for scenario in plan.get("request_scenarios") or []:
        if not isinstance(scenario, dict):
            continue
        if scenario.get("source") == "synthetic":
            summaries.append(
                {
                    key: scenario[key]
                    for key in (
                        "name",
                        "source",
                        "input_len",
                        "output_len",
                        "batch_size",
                        "range_ratio",
                        "seed",
                        "context_fraction",
                        "shared_prefix_len",
                    )
                    if key in scenario
                }
            )
    return summaries


def _archive_dir_b64(root: Path, *, arcname: str) -> str:
    archive_path = root.parent / f"{arcname}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(root, arcname=arcname)
    encoded = base64.b64encode(archive_path.read_bytes()).decode("ascii")
    archive_path.unlink(missing_ok=True)
    return encoded
