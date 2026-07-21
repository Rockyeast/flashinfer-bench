"""Capture non-FlashInfer SGLang calls with the shared tracing runtime."""

from __future__ import annotations

import atexit
import functools
import importlib
import inspect
import json
import os
import re
import shutil
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from flashinfer_bench.data import TraceSet

from .config import TracingConfig, TracingConfigRegistry
from .runtime import TracingRuntime

_MODE_ENV = "FIB_SGLANG_CAPTURE_MODE"
_ROOT_ENV = "FIB_SGLANG_CAPTURE_ROOT"
_DEFINITIONS_ENV = "FIB_SGLANG_DEFINITIONS_DIR"
_INVENTORY_PREFIXES = ("sglang.", "transformers_modules.")


@dataclass(frozen=True)
class SGLangCaptureSpec:
    """One reviewed definition and its exact SGLang capture point."""

    name: str
    modules: tuple[str, ...]
    callables: tuple[str, ...]
    input_sources: dict[str, tuple[str, str]]


@dataclass
class _ProcessState:
    pid: int
    mode: str
    root: Path
    definitions_dir: Path | None
    specs_by_module: dict[str, list[SGLangCaptureSpec]] = field(default_factory=dict)
    specs_by_callable: dict[str, list[SGLangCaptureSpec]] = field(default_factory=dict)
    runtime: TracingRuntime | None = None
    inventory_seen: set[tuple[str, str]] = field(default_factory=set)
    module_binding_seen: set[tuple[int, str, str]] = field(default_factory=set)
    module_parents: dict[int, list[tuple[torch.nn.Module, str]]] = field(
        default_factory=dict
    )
    captures: dict[str, int] = field(default_factory=dict)
    bindings: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    callable_patches: list[tuple[Any, str, Any]] = field(default_factory=list)
    calls_since_flush: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


_STATES: dict[int, _ProcessState] = {}
_HOOK_HANDLE: Any = None
_HOOK_PID: int | None = None
_REGISTRATION_HOOK_HANDLE: Any = None
_REGISTRATION_HOOK_PID: int | None = None
_ATEXIT_REGISTERED = False
_IN_HOOK = threading.local()


def load_sglang_definition_files(definitions_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Return definitions with an exact SGLang module or callable capture tag."""
    files: list[Path] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(definitions_dir.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"definition must be a JSON object: {path}")
        spec = _capture_spec(value)
        if spec.modules or spec.callables:
            files.append(path)
        else:
            skipped.append({"path": str(path), "reason": "missing_sglang_capture_tag"})
    return files, skipped


def build_sglang_dumper_workload_filter(
    definition_files: list[Path], module_inventory: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Build an exact SGLang dumper filter for reviewed module definitions."""
    module_classes: set[str] = set()
    for path in definition_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        module_classes.update(_capture_spec(value).modules)
    if not module_classes:
        return "", {"selected_modules": {}, "missing_modules": []}

    inventory_modules = module_inventory.get("modules")
    if not isinstance(inventory_modules, list):
        raise ValueError("SGLang execution inventory has no modules list")
    paths_by_class: dict[str, list[str]] = {}
    for item in inventory_modules:
        if not isinstance(item, dict) or not isinstance(item.get("class_path"), str):
            continue
        module_paths = item.get("module_paths")
        if not isinstance(module_paths, list):
            continue
        paths_by_class[str(item["class_path"])] = sorted(
            {path for path in module_paths if isinstance(path, str) and path}
        )

    selected: dict[str, str] = {}
    missing: list[str] = []
    for class_path in sorted(module_classes):
        paths = paths_by_class.get(class_path, [])
        if not paths:
            missing.append(class_path)
            continue
        # SGLang's dumper walks ``named_modules()`` with duplicate removal, so a
        # shared module is registered under its first numeric layer path.  A
        # lexical sort would incorrectly place ``layers.14`` before ``layers.4``.
        selected[class_path] = min(paths, key=_module_path_order)
    if missing:
        raise ValueError(
            "reviewed SGLang modules are absent from execution inventory: "
            + ", ".join(missing)
        )
    if not selected:
        return "", {"selected_modules": {}, "missing_modules": []}

    alternatives = "|".join(re.escape(path) for path in selected.values())
    pattern = (
        rf"^non_intrusive__(?:{alternatives})\."
        r"(?:inputs(?:\.|$)|output(?:\.|$))"
    )
    return (
        f"search({pattern!r}, name) is not None",
        {"selected_modules": selected, "missing_modules": []},
    )


def adapt_sglang_dumper_workloads(
    dump_root: Path,
    *,
    capture_root: Path,
    definition_files: list[Path],
) -> dict[str, Any]:
    """Map SGLang module input dumps to Definitions and reuse TracingRuntime.collect."""
    definitions: dict[str, tuple[Path, SGLangCaptureSpec, list[str]]] = {}
    for path in definition_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        spec = _capture_spec(value)
        if spec.modules:
            inputs = value.get("inputs")
            definitions[spec.name] = (
                path,
                spec,
                list(inputs) if isinstance(inputs, dict) else [],
            )
    if not definitions:
        return {
            "summary": {
                "dump_files": 0,
                "module_calls": 0,
                "collect_attempts": 0,
                "collect_accepted": 0,
                "collect_rejected": 0,
            },
            "collect_results": {},
            "errors": [],
        }

    bindings = _load_module_bindings(capture_root)
    shard = capture_root / "shards" / "sglang_dumper"
    runtime = _create_shard_runtime(
        shard,
        definitions_dir=None,
        specs=[spec for _, spec, _ in definitions.values()],
        definition_files=[path for path, _, _ in definitions.values()],
    )

    records: list[tuple[tuple[str, int, int, int, str], str, str, str, Any]] = []
    errors: list[str] = []
    files = sorted(dump_root.rglob("*.pt")) if dump_root.exists() else []
    for path in files:
        try:
            item = _load_torch_payload(path)
        except Exception as exc:  # noqa: BLE001 - bad evidence must not hide all captures
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(item, dict) or not isinstance(item.get("meta"), dict):
            continue
        meta = item["meta"]
        parsed = _parse_sglang_dumper_name(meta.get("name"))
        if parsed is None:
            continue
        module_path, value_kind, value_name = parsed
        relative = path.relative_to(dump_root)
        experiment = relative.parts[0] if len(relative.parts) > 1 else ""
        rank = meta.get("rank", meta.get("world_rank"))
        step = meta.get("step")
        dump_index = meta.get("dump_index")
        order = (
            experiment,
            int(rank) if type(rank) is int else -1,
            int(step) if type(step) is int else -1,
            int(dump_index) if type(dump_index) is int else -1,
            str(path),
        )
        records.append((order, module_path, value_kind, value_name, item.get("value")))

    pending: dict[tuple[str, int, str], dict[str, Any]] = {}
    module_calls = 0
    collect_attempts = 0
    collect_accepted = 0
    collect_rejected = 0
    collect_results: dict[str, dict[str, Any]] = {}
    missing_bindings: set[str] = set()
    for order, module_path, value_kind, value_name, value in sorted(records):
        key = (order[0], order[1], module_path)
        if value_kind == "input":
            pending.setdefault(key, {})[value_name] = value
            continue
        call_inputs = pending.pop(key, {})
        if not call_inputs:
            continue
        module_calls += 1
        module_bindings = bindings.get(module_path, [])
        if not module_bindings:
            missing_bindings.add(module_path)
            continue
        for binding in module_bindings:
            definition_name = binding.get("definition")
            definition_entry = definitions.get(str(definition_name))
            if definition_entry is None:
                continue
            _, spec, input_names = definition_entry
            try:
                values = _definition_values_from_dumper(
                    spec, binding, call_inputs, input_names=input_names
                )
            except (KeyError, TypeError, ValueError) as exc:
                if len(errors) < 50:
                    errors.append(f"{spec.name}@{module_path}: {exc}")
                continue
            result = runtime.collect(spec.name, args=(), kwargs=values)
            collect_attempts += 1
            definition_result = collect_results.setdefault(
                spec.name, {"attempts": 0, "accepted": 0, "rejected": 0, "reasons": {}}
            )
            definition_result["attempts"] += 1
            if result.accepted:
                collect_accepted += 1
                definition_result["accepted"] += 1
            else:
                collect_rejected += 1
                definition_result["rejected"] += 1
                reason = result.reason or "unknown"
                reasons = definition_result["reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
    runtime.flush()
    return {
        "summary": {
            "dump_files": len(files),
            "module_calls": module_calls,
            "collect_attempts": collect_attempts,
            "collect_accepted": collect_accepted,
            "collect_rejected": collect_rejected,
            "missing_bindings": len(missing_bindings),
            "errors": len(errors),
        },
        "collect_results": collect_results,
        "missing_binding_paths": sorted(missing_bindings),
        "errors": errors[:50],
    }


@contextmanager
def sglang_capture_environment(
    capture_root: Path, *, mode: str, definitions_dir: Path | None = None
) -> Iterator[None]:
    """Install the same SGLang hook in the Modal parent and spawned workers."""
    if mode not in {"inventory", "workloads"}:
        raise ValueError("SGLang capture mode must be inventory or workloads")
    if mode == "workloads" and definitions_dir is None:
        raise ValueError("workload capture requires definitions_dir")

    shutil.rmtree(capture_root, ignore_errors=True)
    capture_root.mkdir(parents=True, exist_ok=True)
    bootstrap_dir = capture_root / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    (bootstrap_dir / "sitecustomize.py").write_text(
        "from flashinfer_bench.tracing.sglang_logging import install_from_env\n"
        "install_from_env()\n",
        encoding="utf-8",
    )

    updates = {
        _MODE_ENV: mode,
        _ROOT_ENV: str(capture_root),
        _DEFINITIONS_ENV: str(definitions_dir or ""),
        "PYTHONPATH": os.pathsep.join(
            part for part in (str(bootstrap_dir), os.environ.get("PYTHONPATH", "")) if part
        ),
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    install_from_env()
    try:
        yield
    finally:
        uninstall_current_process()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def install_from_env() -> None:
    """Worker bootstrap entrypoint called by the temporary ``sitecustomize``."""
    mode = os.environ.get(_MODE_ENV)
    root = os.environ.get(_ROOT_ENV)
    if mode not in {"inventory", "workloads"} or not root:
        return
    state = _state_for_current_process()
    _install_module_registration_hook()
    _install_global_module_hook()
    if state.mode == "workloads" and not state.callable_patches:
        _install_callable_wrappers(state)
    _register_atexit()


def uninstall_current_process() -> None:
    """Flush and remove hooks installed in the current process."""
    global _HOOK_HANDLE, _HOOK_PID, _REGISTRATION_HOOK_HANDLE, _REGISTRATION_HOOK_PID
    state = _STATES.pop(os.getpid(), None)
    if state is not None:
        _flush_state(state)
        for owner, attribute, original in reversed(state.callable_patches):
            setattr(owner, attribute, original)
    if _HOOK_HANDLE is not None and _HOOK_PID == os.getpid():
        _HOOK_HANDLE.remove()
        _HOOK_HANDLE = None
        _HOOK_PID = None
    if _REGISTRATION_HOOK_HANDLE is not None and _REGISTRATION_HOOK_PID == os.getpid():
        _REGISTRATION_HOOK_HANDLE.remove()
        _REGISTRATION_HOOK_HANDLE = None
        _REGISTRATION_HOOK_PID = None


def summarize_module_inventory(capture_root: Path) -> dict[str, Any]:
    """Combine worker module observations into one small source-evidence report."""
    modules: dict[str, dict[str, Any]] = {}
    files = sorted((capture_root / "inventory").glob("*.jsonl"))
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            class_path = item.get("class_path")
            if not isinstance(class_path, str):
                continue
            existing = modules.setdefault(class_path, dict(item))
            processes = set(existing.pop("observed_processes", []))
            processes.add(int(item.get("pid") or 0))
            existing["observed_processes"] = sorted(process for process in processes if process)
            module_paths = set(existing.pop("module_paths", []))
            raw_paths = item.get("module_paths")
            if isinstance(raw_paths, list):
                module_paths.update(path for path in raw_paths if isinstance(path, str) and path)
            existing["module_paths"] = sorted(module_paths)
            layer_indices = set(existing.pop("layer_indices", []))
            raw_indices = item.get("layer_indices")
            if isinstance(raw_indices, list):
                layer_indices.update(index for index in raw_indices if type(index) is int)
            existing["layer_indices"] = sorted(layer_indices)
            observations: set[tuple[int, str, int | None]] = set()
            for observation in existing.pop("module_observations", []):
                if not isinstance(observation, dict):
                    continue
                pid = observation.get("pid")
                module_path = observation.get("module_path")
                layer_index = observation.get("layer_index")
                if type(pid) is int and isinstance(module_path, str):
                    observations.add(
                        (pid, module_path, layer_index if type(layer_index) is int else None)
                    )
            raw_observations = item.get("module_observations")
            if isinstance(raw_observations, list):
                for observation in raw_observations:
                    if not isinstance(observation, dict):
                        continue
                    pid = observation.get("pid")
                    module_path = observation.get("module_path")
                    layer_index = observation.get("layer_index")
                    if type(pid) is int and isinstance(module_path, str):
                        observations.add(
                            (
                                pid,
                                module_path,
                                layer_index if type(layer_index) is int else None,
                            )
                        )
            elif type(item.get("pid")) is int:
                for module_path in raw_paths if isinstance(raw_paths, list) else []:
                    if isinstance(module_path, str) and module_path:
                        observations.add(
                            (
                                item["pid"],
                                module_path,
                                _layer_index_from_path(module_path),
                            )
                        )
            existing["module_observations"] = [
                {
                    "pid": pid,
                    "module_path": module_path,
                    "layer_index": layer_index,
                }
                for pid, module_path, layer_index in sorted(
                    observations,
                    key=lambda observation: (observation[0], observation[1]),
                )
            ]
    return {
        "summary": {"modules": len(modules), "worker_files": len(files)},
        "modules": [modules[name] for name in sorted(modules)],
    }


def _parse_sglang_dumper_name(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, str) or not value.startswith("non_intrusive__"):
        return None
    value = value.removeprefix("non_intrusive__")
    match = re.fullmatch(r"(.+)\.(inputs|output)(?:\.(.+))?", value)
    if match is None:
        return None
    module_path, kind, value_name = match.groups()
    if kind == "inputs" and not value_name:
        return None
    return module_path, "input" if kind == "inputs" else "output", value_name or "output"


def summarize_sglang_capture_status(capture_root: Path) -> dict[str, Any]:
    """Combine process-local capture counters and bounded error samples."""
    captures: dict[str, int] = {}
    bindings: dict[str, int] = {}
    errors: list[str] = []
    files = sorted((capture_root / "status").glob("*.json"))
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        for name, count in item.get("captures", {}).items():
            captures[str(name)] = captures.get(str(name), 0) + int(count)
        for name, count in item.get("bindings", {}).items():
            bindings[str(name)] = bindings.get(str(name), 0) + int(count)
        errors.extend(str(error) for error in item.get("errors", []))
    return {
        "summary": {
            "processes": len(files),
            "capture_calls": sum(captures.values()),
            "module_bindings": sum(bindings.values()),
            "errors": len(errors),
        },
        "captures": captures,
        "bindings": bindings,
        "errors": errors[:50],
    }


def merge_sglang_workload_shards(
    capture_root: Path, *, dataset_dir: Path, definition_files: list[Path], max_new_workloads: int
) -> dict[str, int]:
    """Merge process-local workload shards and deduplicate them by definition axes."""
    definitions: dict[str, dict[str, Any]] = {}
    for path in definition_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        definitions[str(value["name"])] = value

    for name, definition in definitions.items():
        op_type = str(definition["op_type"])
        (dataset_dir / "workloads" / op_type / f"{name}.jsonl").unlink(missing_ok=True)
        shutil.rmtree(dataset_dir / "blob" / "workloads" / op_type / name, ignore_errors=True)

    selected: dict[str, list[tuple[dict[str, Any], Path]]] = {name: [] for name in definitions}
    seen_axes: dict[str, set[str]] = {name: set() for name in definitions}
    for shard in sorted((capture_root / "shards").glob("*")):
        for path in sorted((shard / "workloads").rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                name = item.get("definition")
                if name not in definitions or len(selected[name]) >= max_new_workloads:
                    continue
                axes = item.get("workload", {}).get("axes", {})
                axes_key = json.dumps(axes, sort_keys=True, separators=(",", ":"))
                if axes_key in seen_axes[name]:
                    continue
                _restore_scalar_inputs(item, definitions[name], blob_root=shard)
                seen_axes[name].add(axes_key)
                selected[name].append((item, shard))

    counts: dict[str, int] = {}
    for name, entries in selected.items():
        definition = definitions[name]
        op_type = str(definition["op_type"])
        output_path = dataset_dir / "workloads" / op_type / f"{name}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if entries:
            with output_path.open("w", encoding="utf-8") as stream:
                for item, shard in entries:
                    _copy_trace_blobs(item, shard=shard, dataset_dir=dataset_dir)
                    stream.write(json.dumps(item, separators=(",", ":")) + "\n")
        counts[name] = len(entries)
    return counts


def restore_dataset_scalar_inputs(
    dataset_dir: Path, *, definition_files: list[Path]
) -> int:
    """Repair scalar SGLang inputs in an already materialized dataset."""
    restored = 0
    for definition_path in definition_files:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        workload_path = (
            dataset_dir
            / "workloads"
            / str(definition["op_type"])
            / f"{definition['name']}.jsonl"
        )
        if not workload_path.exists():
            continue
        items = [
            json.loads(line)
            for line in workload_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        changed = 0
        for item in items:
            changed += _restore_scalar_inputs(item, definition, blob_root=dataset_dir)
        if changed:
            workload_path.write_text(
                "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in items),
                encoding="utf-8",
            )
            restored += changed
        referenced = {
            dataset_dir / str(input_spec["path"])
            for item in items
            for input_spec in item.get("workload", {}).get("inputs", {}).values()
            if isinstance(input_spec, dict)
            and input_spec.get("type") == "safetensors"
            and isinstance(input_spec.get("path"), str)
        }
        blob_dir = (
            dataset_dir
            / "blob"
            / "workloads"
            / str(definition["op_type"])
            / str(definition["name"])
        )
        for blob_path in blob_dir.glob("*.safetensors"):
            if blob_path not in referenced:
                blob_path.unlink()
    return restored


def _restore_scalar_inputs(
    item: dict[str, Any], definition: dict[str, Any], *, blob_root: Path
) -> int:
    """Convert zero-dimensional captured tensors back to workload scalars."""
    workload_inputs = item.get("workload", {}).get("inputs", {})
    definition_inputs = definition.get("inputs", {})
    if not isinstance(workload_inputs, dict) or not isinstance(definition_inputs, dict):
        return 0

    restored = 0
    loaded: dict[Path, dict[str, torch.Tensor]] = {}
    for name, definition_input in definition_inputs.items():
        workload_input = workload_inputs.get(name)
        if (
            not isinstance(definition_input, dict)
            or definition_input.get("shape") is not None
            or not isinstance(workload_input, dict)
            or workload_input.get("type") != "safetensors"
        ):
            continue
        raw_path = workload_input.get("path")
        tensor_key = workload_input.get("tensor_key")
        if not isinstance(raw_path, str) or not isinstance(tensor_key, str):
            continue
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe SGLang workload blob path: {raw_path}")
        source = blob_root / relative
        tensors = loaded.get(source)
        if tensors is None:
            from safetensors.torch import load_file

            tensors = load_file(str(source), device="cpu")
            loaded[source] = tensors
        tensor = tensors.get(tensor_key)
        if tensor is None or tensor.numel() != 1:
            continue
        workload_inputs[name] = {"type": "scalar", "value": tensor.item()}
        restored += 1
    return restored


def _state_for_current_process() -> _ProcessState:
    pid = os.getpid()
    existing = _STATES.get(pid)
    if existing is not None:
        return existing
    mode = os.environ[_MODE_ENV]
    root = Path(os.environ[_ROOT_ENV])
    raw_definitions = os.environ.get(_DEFINITIONS_ENV, "")
    definitions_dir = Path(raw_definitions) if raw_definitions else None
    state = _ProcessState(pid=pid, mode=mode, root=root, definitions_dir=definitions_dir)
    if mode == "workloads":
        if definitions_dir is None:
            raise RuntimeError("SGLang workload capture has no definitions directory")
        state = _prepare_workload_state(state)
    _STATES[pid] = state
    return state


def _prepare_workload_state(state: _ProcessState) -> _ProcessState:
    assert state.definitions_dir is not None
    specs: list[SGLangCaptureSpec] = []
    for path in sorted(state.definitions_dir.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        spec = _capture_spec(value)
        if spec.modules or spec.callables:
            specs.append(spec)
            for module_path in spec.modules:
                state.specs_by_module.setdefault(module_path, []).append(spec)
            for callable_path in spec.callables:
                state.specs_by_callable.setdefault(callable_path, []).append(spec)

    callable_specs = [spec for spec in specs if spec.callables]
    if callable_specs:
        state.runtime = _create_shard_runtime(
            state.root / "shards" / str(state.pid),
            definitions_dir=state.definitions_dir,
            specs=callable_specs,
        )
    return state


def _create_shard_runtime(
    shard: Path,
    *,
    definitions_dir: Path | None,
    specs: list[SGLangCaptureSpec],
    definition_files: list[Path] | None = None,
) -> TracingRuntime:
    definitions_output = shard / "definitions"
    names = {spec.name for spec in specs}
    sources = (
        sorted(definition_files)
        if definition_files is not None
        else sorted(definitions_dir.rglob("*.json")) if definitions_dir is not None else []
    )
    for source in sources:
        value = json.loads(source.read_text(encoding="utf-8"))
        name = value.get("name") if isinstance(value, dict) else None
        if not isinstance(name, str) or name not in names:
            continue
        if definitions_dir is not None:
            relative = source.relative_to(definitions_dir)
        else:
            relative = Path(str(value["op_type"])) / source.name
        destination = definitions_output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    trace_set = TraceSet.from_path(str(shard))
    config = TracingConfig(
        input_dump_policy="dump_structural",
        filter_policy="keep_first_k_by_axes",
        filter_policy_kwargs={"k": 1},
    )
    registry = TracingConfigRegistry(per_definition={spec.name: config for spec in specs})
    return TracingRuntime(trace_set, registry)


def _capture_spec(definition: dict[str, Any]) -> SGLangCaptureSpec:
    tags = definition.get("tags") if isinstance(definition.get("tags"), list) else []
    modules = tuple(
        tag.removeprefix("sglang_module:")
        for tag in tags
        if isinstance(tag, str) and tag.startswith("sglang_module:")
    )
    callables = tuple(
        tag.removeprefix("sglang_callable:")
        for tag in tags
        if isinstance(tag, str) and tag.startswith("sglang_callable:")
    )
    mappings: dict[str, tuple[str, str]] = {}
    for tag in tags:
        if not isinstance(tag, str) or not tag.startswith("sglang_input:"):
            continue
        payload = tag.removeprefix("sglang_input:")
        definition_input, separator, source = payload.partition("=")
        kind, kind_separator, source_name = source.partition(":")
        if separator and kind_separator and kind in {"arg", "attr"}:
            mappings[definition_input] = (kind, source_name)
    return SGLangCaptureSpec(
        name=str(definition.get("name") or ""),
        modules=modules,
        callables=callables,
        input_sources=mappings,
    )


def _install_global_module_hook() -> None:
    global _HOOK_HANDLE, _HOOK_PID
    if _HOOK_HANDLE is not None:
        _HOOK_PID = os.getpid()
        return
    register = torch.nn.modules.module.register_module_forward_hook
    try:
        _HOOK_HANDLE = register(_module_forward_hook, with_kwargs=True)
    except TypeError:
        _HOOK_HANDLE = register(_module_forward_hook_without_kwargs)
    _HOOK_PID = os.getpid()


def _install_module_registration_hook() -> None:
    global _REGISTRATION_HOOK_HANDLE, _REGISTRATION_HOOK_PID
    if _REGISTRATION_HOOK_HANDLE is not None:
        _REGISTRATION_HOOK_PID = os.getpid()
        return
    register = torch.nn.modules.module.register_module_module_registration_hook
    _REGISTRATION_HOOK_HANDLE = register(_module_registration_hook)
    _REGISTRATION_HOOK_PID = os.getpid()


def _module_registration_hook(
    parent: torch.nn.Module, name: str, child: torch.nn.Module | None
) -> None:
    if child is None:
        return
    state = _state_for_current_process()
    parents = state.module_parents.setdefault(id(child), [])
    if not any(existing is parent and existing_name == name for existing, existing_name in parents):
        parents.append((parent, name))


def _module_forward_hook(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
) -> None:
    _handle_module_call(module, args, kwargs, output)


def _module_forward_hook_without_kwargs(
    module: torch.nn.Module, args: tuple[Any, ...], output: Any
) -> None:
    _handle_module_call(module, args, {}, output)


def _handle_module_call(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
) -> None:
    if getattr(_IN_HOOK, "active", False):
        return
    _IN_HOOK.active = True
    try:
        state = _state_for_current_process()
        class_path = f"{type(module).__module__}.{type(module).__qualname__}"
        if state.mode == "inventory":
            _record_inventory(state, module, class_path, args, kwargs, output)
            return
        specs = state.specs_by_module.get(class_path, [])
        for spec in specs:
            _record_module_binding(state, spec, module)
    finally:
        _IN_HOOK.active = False


def _install_callable_wrappers(state: _ProcessState) -> None:
    for path, specs in state.specs_by_callable.items():
        owner, attribute, original = _resolve_attribute(path)

        @functools.wraps(original)
        def wrapped(*args: Any, __original=original, __specs=tuple(specs), **kwargs: Any) -> Any:
            current = _state_for_current_process()
            if not getattr(_IN_HOOK, "active", False):
                _IN_HOOK.active = True
                try:
                    for spec in __specs:
                        _collect_call(current, spec, __original, args, kwargs, module=None)
                    if current.runtime is not None:
                        _flush_incrementally(current)
                finally:
                    _IN_HOOK.active = False
            return __original(*args, **kwargs)

        setattr(owner, attribute, wrapped)
        state.callable_patches.append((owner, attribute, original))


def _collect_call(
    state: _ProcessState,
    spec: SGLangCaptureSpec,
    callable_value: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    module: torch.nn.Module | None,
) -> None:
    if state.runtime is None:
        return
    try:
        signature = inspect.signature(callable_value)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        definition = state.runtime._trace_set.definitions[spec.name]
        values: dict[str, Any] = {}
        for input_name in definition.inputs:
            kind, source_name = spec.input_sources.get(input_name, ("arg", input_name))
            if kind == "arg":
                if source_name.isdecimal():
                    position = int(source_name)
                    if position >= len(args):
                        raise KeyError(f"forward positional argument {position} is missing")
                    values[input_name] = args[position]
                else:
                    if source_name not in bound.arguments:
                        raise KeyError(f"forward argument {source_name!r} is missing")
                    values[input_name] = bound.arguments[source_name]
            else:
                if module is None:
                    raise KeyError(f"attribute source {source_name!r} requires sglang_module")
                values[input_name] = _nested_attribute(module, source_name)
        state.runtime.collect(spec.name, args=(), kwargs=values)
        state.captures[spec.name] = state.captures.get(spec.name, 0) + 1
        state.calls_since_flush += 1
    except Exception as exc:  # noqa: BLE001 - capture must never break model inference
        if len(state.errors) < 20:
            state.errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")


def _record_module_binding(
    state: _ProcessState, spec: SGLangCaptureSpec, module: torch.nn.Module
) -> None:
    """Persist only Definition/path/attribute metadata that SGLang's dumper lacks."""
    module_paths = _module_instance_paths(state, module)
    if not module_paths:
        if len(state.errors) < 20:
            state.errors.append(f"{spec.name}: module instance path is unavailable")
        return
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        signature = None
    argument_positions: dict[str, int] = {}
    if signature is not None:
        position = 0
        for parameter in signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }:
                argument_positions[parameter.name] = position
                position += 1

    attributes: dict[str, Any] = {}
    try:
        for input_name, (kind, source_name) in spec.input_sources.items():
            if kind == "attr":
                attributes[input_name] = _portable_binding_value(
                    _nested_attribute(module, source_name)
                )
    except Exception as exc:  # noqa: BLE001 - capture metadata cannot break inference
        if len(state.errors) < 20:
            state.errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")
        return

    for module_path in module_paths:
        identity = (id(module), module_path, spec.name)
        if identity in state.module_binding_seen:
            continue
        state.module_binding_seen.add(identity)
        destination = (
            state.root
            / "module_bindings"
            / str(state.pid)
            / f"{len(state.module_binding_seen):05d}.pt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "definition": spec.name,
                "class_path": f"{type(module).__module__}.{type(module).__qualname__}",
                "module_path": module_path,
                "argument_positions": argument_positions,
                "attributes": attributes,
            },
            destination,
        )
        state.bindings[spec.name] = state.bindings.get(spec.name, 0) + 1


def _portable_binding_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_portable_binding_value(item) for item in value)
    if isinstance(value, list):
        return [_portable_binding_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _portable_binding_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported module attribute type: {type(value).__name__}")


def _load_module_bindings(capture_root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((capture_root / "module_bindings").rglob("*.pt")):
        value = _load_torch_payload(path)
        if not isinstance(value, dict) or not isinstance(value.get("module_path"), str):
            continue
        result.setdefault(str(value["module_path"]), []).append(value)
    return result


def _definition_values_from_dumper(
    spec: SGLangCaptureSpec,
    binding: dict[str, Any],
    call_inputs: dict[str, Any],
    *,
    input_names: list[str],
) -> dict[str, Any]:
    argument_positions = binding.get("argument_positions")
    attributes = binding.get("attributes")
    if not isinstance(argument_positions, dict) or not isinstance(attributes, dict):
        raise TypeError("invalid module binding sidecar")
    values: dict[str, Any] = {}
    for input_name in input_names:
        source = spec.input_sources.get(input_name, ("arg", input_name))
        kind, source_name = source
        if kind == "attr":
            if input_name not in attributes:
                raise KeyError(f"attribute input {input_name!r} is missing")
            values[input_name] = attributes[input_name]
            continue
        dumper_name = source_name
        if not source_name.isdecimal() and source_name in argument_positions:
            dumper_name = str(argument_positions[source_name])
        values[input_name] = _dumper_argument(call_inputs, dumper_name)
    return values


def _dumper_argument(call_inputs: dict[str, Any], name: str) -> Any:
    if name in call_inputs:
        return call_inputs[name]
    prefix = f"{name}."
    indexed = [
        (int(key.removeprefix(prefix)), value)
        for key, value in call_inputs.items()
        if key.startswith(prefix) and key.removeprefix(prefix).isdigit()
    ]
    if indexed:
        return tuple(value for _, value in sorted(indexed))
    raise KeyError(f"SGLang dumper input {name!r} is missing")


def _load_torch_payload(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def _record_inventory(
    state: _ProcessState,
    module: torch.nn.Module,
    class_path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
) -> None:
    if not class_path.startswith(_INVENTORY_PREFIXES):
        return
    module_paths = _module_instance_paths(state, module)
    identities = {(class_path, path) for path in module_paths} or {(class_path, "")}
    if identities <= state.inventory_seen:
        return
    state.inventory_seen.update(identities)
    forward = module.forward
    try:
        signature = inspect.signature(forward)
        bound = signature.bind_partial(*args, **kwargs)
        inputs: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            parameter = signature.parameters.get(name)
            if parameter is not None and parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                inputs.update(
                    {f"arg_{index}": _describe_value(item) for index, item in enumerate(value)}
                )
            elif parameter is not None and parameter.kind is inspect.Parameter.VAR_KEYWORD:
                inputs.update({key: _describe_value(item) for key, item in value.items()})
            else:
                inputs[name] = _describe_value(value)
    except (TypeError, ValueError):
        signature = "unavailable"
        inputs = {f"arg_{index}": _describe_value(value) for index, value in enumerate(args)}
    try:
        source = inspect.getsource(type(module))[:12000]
    except (OSError, TypeError):
        source = ""
    try:
        source_file = inspect.getsourcefile(type(module)) or ""
    except TypeError:
        source_file = ""
    item = {
        "class_path": class_path,
        "module_paths": module_paths,
        "module_observations": [
            {
                "pid": state.pid,
                "module_path": path,
                "layer_index": _layer_index_from_path(path),
            }
            for path in module_paths
        ],
        "layer_indices": sorted(
            {
                layer_index
                for path in module_paths
                if (layer_index := _layer_index_from_path(path)) is not None
            }
        ),
        "forward_signature": str(signature),
        "sample_inputs": inputs,
        "sample_output": _describe_value(output),
        "source_file": source_file,
        "source": source,
        "pid": state.pid,
    }
    path = state.root / "inventory" / f"{state.pid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def _module_instance_paths(state: _ProcessState, module: torch.nn.Module) -> list[str]:
    """Recover paths relative to the root model from module registration events."""

    def visit(current: torch.nn.Module, seen: set[int]) -> list[str]:
        current_id = id(current)
        if current_id in seen:
            return []
        parents = state.module_parents.get(current_id, [])
        if not parents:
            return [""]
        paths: list[str] = []
        for parent, child_name in parents:
            for parent_path in visit(parent, {*seen, current_id}):
                paths.append(".".join(part for part in (parent_path, child_name) if part))
        return paths

    return sorted(set(path for path in visit(module, set()) if path))


def _layer_index_from_path(path: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", path)
    return int(match.group(1)) if match else None


def _module_path_order(path: str) -> tuple[int, str]:
    layer_index = _layer_index_from_path(path)
    return (layer_index if layer_index is not None else sys.maxsize, path)


def _logger_process_metadata(path: Path) -> dict[str, int | None]:
    match = re.search(r"Rank(?P<rank>\d+)_pid(?P<pid>\d+)", str(path.parent))
    if match is None:
        return {"rank": None, "pid": None}
    return {"rank": int(match.group("rank")), "pid": int(match.group("pid"))}


def _describe_value(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return {
            "type": type(value).__name__,
            "items": [_describe_value(item) for item in value[:8]],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": {str(key): _describe_value(item) for key, item in list(value.items())[:8]},
        }
    return {"type": type(value).__name__, "value": repr(value)[:200]}


def _resolve_attribute(path: str) -> tuple[Any, str, Any]:
    parts = path.split(".")
    for index in range(len(parts) - 1, 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:index]))
        except ImportError:
            continue
        for part in parts[index:-1]:
            value = getattr(value, part)
        attribute = parts[-1]
        original = getattr(value, attribute)
        if not callable(original):
            raise TypeError(f"SGLang capture target is not callable: {path}")
        return value, attribute, original
    raise ImportError(f"cannot import SGLang capture target: {path}")


def _nested_attribute(value: Any, path: str) -> Any:
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _copy_trace_blobs(item: dict[str, Any], *, shard: Path, dataset_dir: Path) -> None:
    inputs = item.get("workload", {}).get("inputs", {})
    if not isinstance(inputs, dict):
        return
    for spec in inputs.values():
        if not isinstance(spec, dict) or spec.get("type") != "safetensors":
            continue
        raw_path = spec.get("path")
        if not isinstance(raw_path, str):
            continue
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe SGLang workload blob path: {raw_path}")
        source = shard / relative
        destination = dataset_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_state_summary(state: _ProcessState) -> None:
    path = state.root / "status" / f"{state.pid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "pid": state.pid,
                "captures": state.captures,
                "bindings": state.bindings,
                "errors": state.errors,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _flush_state(state: _ProcessState) -> None:
    if state.runtime is not None:
        state.runtime.flush()
        state.calls_since_flush = 0
    _write_state_summary(state)


def _flush_incrementally(state: _ProcessState) -> None:
    """Bound buffered captures without flushing after every module call."""
    if state.calls_since_flush >= 256:
        _flush_state(state)


def _flush_current_process() -> None:
    state = _STATES.get(os.getpid())
    if state is not None:
        _flush_state(state)


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_flush_current_process)
        _ATEXIT_REGISTERED = True
