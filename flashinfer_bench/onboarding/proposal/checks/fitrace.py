"""FlashInfer fitrace proposal checks."""

from __future__ import annotations

from ..common import *  # noqa: F403

def _resolve_dotted_object(target: str) -> tuple[Any | None, str | None]:
    """Resolve a dotted Python object without importing guessed parent modules."""
    parts = target.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        return None, "target is not a valid dotted path"

    last_error: Exception | None = None
    for split_at in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_at])
        attr_parts = parts[split_at:]
        try:
            obj = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exact import errors vary by install
            last_error = exc
            continue
        try:
            for attr in attr_parts:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            return None, f"attribute not found: {'.'.join(attr_parts)} ({exc})"
        return obj, None

    detail = f": {type(last_error).__name__}: {last_error}" if last_error else ""
    return None, f"module import failed{detail}"


def _decorator_uses_flashinfer_api(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        candidate = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(candidate, ast.Name) and candidate.id == "flashinfer_api":
            return True
        if isinstance(candidate, ast.Attribute) and candidate.attr == "flashinfer_api":
            return True
    return False


def _trace_template_metadata(flashinfer_root: Path | None, trace_name: str | None) -> dict[str, str | None]:
    if flashinfer_root is None or not trace_name:
        return {"trace_name": trace_name, "trace_op_type": None, "trace_name_prefix": None}
    for path in sorted(flashinfer_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == trace_name for target in node.targets):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (
                isinstance(func, ast.Name) and func.id == "TraceTemplate"
                or isinstance(func, ast.Attribute) and func.attr == "TraceTemplate"
            ):
                continue
            metadata: dict[str, str | None] = {
                "trace_name": trace_name,
                "trace_op_type": None,
                "trace_name_prefix": None,
            }
            for keyword in call.keywords:
                if keyword.arg in {"op_type", "name_prefix"} and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    if isinstance(value, str):
                        metadata[f"trace_{keyword.arg}"] = value
            return metadata
    return {"trace_name": trace_name, "trace_op_type": None, "trace_name_prefix": None}


def _trace_name_from_decorator(node: ast.FunctionDef) -> str | None:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        candidate = decorator.func
        if not (
            isinstance(candidate, ast.Name) and candidate.id == "flashinfer_api"
            or isinstance(candidate, ast.Attribute) and candidate.attr == "flashinfer_api"
        ):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "trace" and isinstance(keyword.value, ast.Name):
                return keyword.value.id
    return None


def _function_fitrace_metadata(
    module_path: Path,
    attr_parts: list[str],
    flashinfer_root: Path | None,
) -> dict[str, Any]:
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"source parse failed: {type(exc).__name__}: {exc}"}

    scope: list[ast.stmt] = list(tree.body)
    node: ast.AST | None = None
    for index, part in enumerate(attr_parts):
        node = None
        matches = [
            item
            for item in scope
            if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == part
        ]
        if matches and index == len(attr_parts) - 1:
            function_matches = [item for item in matches if isinstance(item, ast.FunctionDef)]
            if function_matches:
                for item in function_matches:
                    if _decorator_uses_flashinfer_api(item):
                        return {
                            "ok": True,
                            "error": None,
                            **_trace_template_metadata(flashinfer_root, _trace_name_from_decorator(item)),
                        }
                return {"ok": False, "error": "source function exists but has no @flashinfer_api decorator"}
        if matches:
            node = matches[0]
        if node is None:
            return {"ok": False, "error": f"source attribute not found: {'.'.join(attr_parts)}"}
        if isinstance(node, ast.ClassDef):
            scope = list(node.body)
        elif isinstance(node, ast.FunctionDef):
            scope = []
        else:
            return {"ok": False, "error": f"unsupported source node for {part}"}

    if isinstance(node, ast.FunctionDef) and _decorator_uses_flashinfer_api(node):
        return {
            "ok": True,
            "error": None,
            **_trace_template_metadata(flashinfer_root, _trace_name_from_decorator(node)),
        }
    if isinstance(node, ast.FunctionDef):
        return {"ok": False, "error": "source function exists but has no @flashinfer_api decorator"}
    return {"ok": False, "error": "source target is not a function"}


def _suggest_fitrace_attr_parts(module_path: Path, attr_parts: list[str]) -> list[str] | None:
    """Suggest a sibling decorated function when a target points at a wrapper alias."""
    if len(attr_parts) < 2:
        return None
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    scope: list[ast.stmt] = list(tree.body)
    for part in attr_parts[:-1]:
        matches = [
            item
            for item in scope
            if isinstance(item, ast.ClassDef) and item.name == part
        ]
        if not matches:
            return None
        scope = list(matches[0].body)

    decorated_methods = [
        item.name
        for item in scope
        if isinstance(item, ast.FunctionDef) and _decorator_uses_flashinfer_api(item)
    ]
    if "run" in decorated_methods and attr_parts[-1] != "run":
        return [*attr_parts[:-1], "run"]
    if decorated_methods:
        return [*attr_parts[:-1], decorated_methods[0]]
    return None


def _find_flashinfer_source_target(
    target: str,
    flashinfer_root: Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_checked": False,
        "source_path": None,
        "source_fitrace_ok": False,
        "source_error": None,
        "suggested_target": None,
    }
    if not target.startswith("flashinfer."):
        result["source_error"] = "not a flashinfer target"
        return result

    roots = [flashinfer_root] if flashinfer_root is not None else list(DEFAULT_FLASHINFER_ROOTS)
    parts = target.split(".")
    for root in roots:
        if root is None:
            continue
        root = root.resolve() if root.exists() else root
        for split_at in range(len(parts) - 1, 1, -1):
            module_parts = parts[1:split_at]
            attr_parts = parts[split_at:]
            candidates = [
                root.joinpath(*module_parts).with_suffix(".py"),
                root.joinpath(*module_parts, "__init__.py"),
            ]
            for module_path in candidates:
                if not module_path.exists():
                    continue
                metadata = _function_fitrace_metadata(module_path, attr_parts, root)
                ok = bool(metadata["ok"])
                error = metadata["error"]
                suggested_attr_parts = None if ok else _suggest_fitrace_attr_parts(module_path, attr_parts)
                suggested_target = None
                if suggested_attr_parts is not None:
                    suggested_target = ".".join([*parts[:split_at], *suggested_attr_parts])
                result.update({
                    "source_checked": True,
                    "source_path": str(module_path),
                    "source_fitrace_ok": ok,
                    "source_error": error,
                    "suggested_target": suggested_target,
                    "trace_name": metadata.get("trace_name"),
                    "trace_op_type": metadata.get("trace_op_type"),
                    "trace_name_prefix": metadata.get("trace_name_prefix"),
                })
                return result

    result["source_error"] = "source file not found"
    return result


def _expected_fitrace_preview(hf_config: dict[str, Any]) -> dict[str, Any]:
    """Return non-authoritative model facts useful for reviewing fitrace candidates."""
    heads = hf_config.get("num_attention_heads")
    kv_heads = hf_config.get("num_key_value_heads", heads)
    hidden_size = hf_config.get("hidden_size")
    head_dim = hf_config.get("head_dim")
    if not isinstance(head_dim, int) and isinstance(hidden_size, int) and isinstance(heads, int) and heads > 0:
        head_dim = hidden_size // heads
    return {
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": hf_config.get("vocab_size"),
        "note": "diagnostic only; final definition name/schema must come from fitrace dump",
    }


def _required_sglang_engine_kwargs(hf_config: dict[str, Any]) -> dict[str, str]:
    """Return reviewed Engine kwargs needed before SGLang can load this config."""
    rope_scaling = hf_config.get("rope_scaling")
    if (
        hf_config.get("model_type") == "phi3"
        and isinstance(rope_scaling, dict)
        and rope_scaling.get("type") == "longrope"
        and "rope_theta" in hf_config
    ):
        cleaned = dict(hf_config)
        cleaned.pop("rope_theta", None)
        cleaned_rope = {
            key: value
            for key, value in rope_scaling.items()
            if key in {"type", "short_factor", "long_factor"}
        }
        if set(cleaned_rope) == {"type", "short_factor", "long_factor"}:
            cleaned["rope_scaling"] = cleaned_rope
            return {
                "decrypted_config_json": json.dumps(cleaned, separators=(",", ":")),
            }
    return {}


def _check_run_config(
    *,
    proposal_dir: Path,
    hf_config: dict[str, Any],
) -> dict[str, Any]:
    """Check proposal-time run_config requirements that prevent known startup failures."""
    findings: list[dict[str, str]] = []
    path = proposal_dir.parent / "config" / "run_config.json"
    required_engine_kwargs = _required_sglang_engine_kwargs(hf_config)
    if not path.exists():
        if required_engine_kwargs:
            findings.append({
                "severity": "error",
                "name": "run_config",
                "reason": (
                    "missing proposed runtime config required for model startup; "
                    f"run_config.engine_kwargs must include {sorted(required_engine_kwargs)}: {path}"
                ),
            })
    else:
        data = _load_json(path)
        if not isinstance(data, dict):
            findings.append({
                "severity": "error",
                "name": "run_config",
                "reason": f"run_config must be a JSON object: {path}",
            })
        else:
            engine_kwargs = data.get("engine_kwargs")
            if engine_kwargs is None:
                engine_kwargs = {}
            if not isinstance(engine_kwargs, dict):
                findings.append({
                    "severity": "error",
                    "name": "run_config",
                    "reason": "run_config.engine_kwargs must be an object",
                })
            else:
                for key, expected in required_engine_kwargs.items():
                    actual = engine_kwargs.get(key)
                    if actual != expected:
                        findings.append({
                            "severity": "error",
                            "name": "run_config",
                            "reason": (
                                f"run_config.engine_kwargs.{key} must contain the sanitized "
                                "HF config override required before SGLang loads this model"
                            ),
                        })
    errors = sum(1 for item in findings if item["severity"] == "error")
    warnings = sum(1 for item in findings if item["severity"] == "warning")
    return {
        "summary": {
            "ok": errors == 0,
            "errors": errors,
            "warnings": warnings,
            "required_engine_kwargs": sorted(required_engine_kwargs),
        },
        "run_config_path": str(path),
        "findings": findings,
    }


def _evaluate_fitrace_targets(
    *,
    candidates_path: Path,
    hf_config_path: Path,
    flashinfer_root: Path | None = None,
) -> dict[str, Any]:
    """Check collect candidates against fitrace capability, not old definitions."""
    candidates = _load_json(candidates_path)
    hf_config = _load_json(hf_config_path)
    if not isinstance(candidates, list):
        raise ValueError(f"candidate targets must be a list: {candidates_path}")
    if not isinstance(hf_config, dict):
        raise ValueError(f"HF config must be a JSON object: {hf_config_path}")

    collect_candidates = [
        item
        for item in candidates
        if isinstance(item, dict) and item.get("collect") is True and item.get("backend") == "flashinfer"
    ]

    target_results: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    for index, item in enumerate(collect_candidates):
        name = str(item.get("name") or item.get("definition_name") or f"#{index}")
        target = item.get("target")
        result = {
            "name": name,
            "definition_name": item.get("definition_name"),
            "target": target,
            "backend": item.get("backend"),
            "op_type": item.get("op_type"),
            "variant": item.get("variant"),
            "definition_source": item.get("definition_source"),
            "import_ok": False,
            "fitrace_ok": False,
            "source_checked": False,
            "source_fitrace_ok": False,
            "suggested_target": None,
            "trace_name": None,
            "trace_op_type": None,
            "trace_name_prefix": None,
            "resolved_type": None,
        }
        if not isinstance(target, str) or not target:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "collect candidate has no target",
            })
            target_results.append(result)
            continue

        obj, error = _resolve_dotted_object(target)
        source_result = _find_flashinfer_source_target(target, flashinfer_root)
        result.update(source_result)
        has_fi_trace = False
        if error is None:
            result["import_ok"] = True
            result["resolved_type"] = type(obj).__name__
            has_fi_trace = callable(getattr(obj, "fi_trace", None))
        else:
            result["import_error"] = error

        result["fitrace_ok"] = has_fi_trace or bool(source_result["source_fitrace_ok"])
        if not result["fitrace_ok"]:
            reason = (
                "local target is importable but has no callable .fi_trace"
                if error is None
                else f"local target cannot be imported and source check failed: {error}; {source_result['source_error']}"
            )
            if source_result.get("suggested_target"):
                reason = f"{reason}; suggested trace target: {source_result['suggested_target']}"
            findings.append({
                "severity": "warning",
                "name": name,
                "reason": f"{reason}; verify remote tracing output after run",
            })

        if item.get("backend") != "flashinfer":
            findings.append({
                "severity": "warning",
                "name": name,
                "reason": "collect candidate should declare backend=flashinfer for fitrace evaluation",
            })
        if item.get("definition_source") != "fitrace":
            findings.append({
                "severity": "warning",
                "name": name,
                "reason": "collect candidate should declare definition_source=fitrace when the target is fitrace-backed",
            })
        trace_op_type = source_result.get("trace_op_type")
        if isinstance(trace_op_type, str) and item.get("op_type") != trace_op_type:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": (
                    f"candidate op_type {item.get('op_type')!r} does not match "
                    f"FlashInfer trace template op_type {trace_op_type!r}"
                ),
            })

        definition_name = item.get("definition_name")
        if isinstance(definition_name, str) and definition_name:
            findings.append({
                "severity": "warning",
                "name": name,
                "reason": "definition_name is only a preview for fitrace-backed targets; final name/schema must come from fitrace dump",
            })
        target_results.append(result)

    errors = sum(1 for item in findings if item["severity"] == "error")
    warnings = sum(1 for item in findings if item["severity"] == "warning")
    return {
        "summary": {
            "collect_candidates": len(collect_candidates),
            "importable_targets": sum(1 for item in target_results if item["import_ok"]),
            "fitrace_targets": sum(1 for item in target_results if item["fitrace_ok"]),
            "errors": errors,
            "warnings": warnings,
            "ok": errors == 0,
        },
        "candidates_path": str(candidates_path),
        "hf_config_path": str(hf_config_path),
        "flashinfer_root": str(flashinfer_root) if flashinfer_root is not None else None,
        "fitrace_preview": _expected_fitrace_preview(hf_config),
        "targets": target_results,
        "findings": findings,
    }
