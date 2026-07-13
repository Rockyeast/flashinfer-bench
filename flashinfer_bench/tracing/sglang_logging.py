"""Capture non-FlashInfer SGLang calls with the shared tracing runtime."""

from __future__ import annotations

import atexit
import functools
import importlib
import inspect
import json
import os
import shutil
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
    inventory_seen: set[str] = field(default_factory=set)
    captures: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    callable_patches: list[tuple[Any, str, Any]] = field(default_factory=list)
    calls_since_flush: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


_STATES: dict[int, _ProcessState] = {}
_HOOK_HANDLE: Any = None
_HOOK_PID: int | None = None
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
    _install_global_module_hook()
    if state.mode == "workloads" and not state.callable_patches:
        _install_callable_wrappers(state)
    _register_atexit()


def uninstall_current_process() -> None:
    """Flush and remove hooks installed in the current process."""
    global _HOOK_HANDLE, _HOOK_PID
    state = _STATES.pop(os.getpid(), None)
    if state is not None:
        _flush_state(state)
        for owner, attribute, original in reversed(state.callable_patches):
            setattr(owner, attribute, original)
    if _HOOK_HANDLE is not None and _HOOK_PID == os.getpid():
        _HOOK_HANDLE.remove()
        _HOOK_HANDLE = None
        _HOOK_PID = None


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
    return {
        "summary": {"modules": len(modules), "worker_files": len(files)},
        "modules": [modules[name] for name in sorted(modules)],
    }


def summarize_sglang_tensor_logger(dump_root: Path) -> dict[str, Any]:
    """Summarize SGLang's output-only tensor logger without retaining tensor dumps."""
    operators: dict[str, dict[str, Any]] = {}
    files = sorted(dump_root.rglob("*.pt")) if dump_root.exists() else []
    total_bytes = sum(path.stat().st_size for path in files)
    for path in files:
        try:
            values = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            values = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if isinstance(name, str) and name not in operators:
                operators[name] = _describe_value(value)
    return {
        "summary": {
            "files": len(files),
            "bytes": total_bytes,
            "operators": len(operators),
            "used_for_workloads": False,
        },
        "operators": [{"name": name, **operators[name]} for name in sorted(operators)],
        "note": (
            "SGLang's logger records module outputs. It is comparison evidence only; "
            "definition-driven input capture uses the shared TracingRuntime."
        ),
    }


def summarize_sglang_capture_status(capture_root: Path) -> dict[str, Any]:
    """Combine process-local capture counters and bounded error samples."""
    captures: dict[str, int] = {}
    errors: list[str] = []
    files = sorted((capture_root / "status").glob("*.json"))
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        for name, count in item.get("captures", {}).items():
            captures[str(name)] = captures.get(str(name), 0) + int(count)
        errors.extend(str(error) for error in item.get("errors", []))
    return {
        "summary": {
            "processes": len(files),
            "capture_calls": sum(captures.values()),
            "errors": len(errors),
        },
        "captures": captures,
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

    shard = state.root / "shards" / str(state.pid)
    definitions_output = shard / "definitions"
    for source in sorted(state.definitions_dir.rglob("*.json")):
        value = json.loads(source.read_text(encoding="utf-8"))
        name = value.get("name") if isinstance(value, dict) else None
        if not isinstance(name, str) or not any(spec.name == name for spec in specs):
            continue
        destination = definitions_output / source.relative_to(state.definitions_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    trace_set = TraceSet.from_path(str(shard))
    config = TracingConfig(
        input_dump_policy="dump_structural",
        filter_policy="keep_first_k_by_axes",
        filter_policy_kwargs={"k": 1},
    )
    registry = TracingConfigRegistry(per_definition={spec.name: config for spec in specs})
    state.runtime = TracingRuntime(trace_set, registry)
    return state


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


def _module_forward_hook(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], _output: Any
) -> None:
    _handle_module_call(module, args, kwargs)


def _module_forward_hook_without_kwargs(
    module: torch.nn.Module, args: tuple[Any, ...], _output: Any
) -> None:
    _handle_module_call(module, args, {})


def _handle_module_call(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    if getattr(_IN_HOOK, "active", False):
        return
    _IN_HOOK.active = True
    try:
        state = _state_for_current_process()
        class_path = f"{type(module).__module__}.{type(module).__qualname__}"
        if state.mode == "inventory":
            _record_inventory(state, module, class_path, args, kwargs)
            return
        specs = state.specs_by_module.get(class_path, [])
        for spec in specs:
            _collect_call(state, spec, module.forward, args, kwargs, module=module)
        if specs and state.runtime is not None:
            _flush_incrementally(state)
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


def _record_inventory(
    state: _ProcessState,
    module: torch.nn.Module,
    class_path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    if class_path in state.inventory_seen or not class_path.startswith(_INVENTORY_PREFIXES):
        return
    state.inventory_seen.add(class_path)
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
        "forward_signature": str(signature),
        "sample_inputs": inputs,
        "source_file": source_file,
        "source": source,
        "pid": state.pid,
    }
    path = state.root / "inventory" / f"{state.pid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")


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
            {"pid": state.pid, "captures": state.captures, "errors": state.errors},
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
