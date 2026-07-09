"""Non-FI definition and hint draft checks."""

from __future__ import annotations

from flashinfer_bench.data.definition import Definition

from ..common import *  # noqa: F403

def _definition_draft_path(proposal_dir: Path, op_type: str, definition_name: str) -> Path:
    return proposal_dir / "definitions" / op_type / f"{definition_name}.json"


def _definition_hint_path(proposal_dir: Path, op_type: str, definition_name: str) -> Path:
    return proposal_dir / "definition_hints" / op_type / f"{definition_name}.json"


def _check_definition_object(
    *,
    path: Path,
    name: str,
    op_type: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    try:
        data = _load_json(path)
    except Exception as exc:  # noqa: BLE001 - report agent-facing diagnostics
        return [{"severity": "error", "name": name, "reason": f"definition draft is not readable JSON: {exc}"}]
    if not isinstance(data, dict):
        return [{"severity": "error", "name": name, "reason": "definition draft must be a JSON object"}]
    missing = sorted(field for field in DEFINITION_REQUIRED_FIELDS if field not in data)
    if missing:
        findings.append({"severity": "error", "name": name, "reason": f"definition draft missing fields: {missing}"})
    if data.get("name") != name:
        findings.append({"severity": "error", "name": name, "reason": f"definition draft name mismatch: {data.get('name')!r}"})
    if data.get("op_type") != op_type:
        findings.append({"severity": "error", "name": name, "reason": f"definition draft op_type mismatch: {data.get('op_type')!r}"})
    if not isinstance(data.get("axes"), dict):
        findings.append({"severity": "error", "name": name, "reason": "definition draft axes must be an object"})
    inputs = data.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        findings.append({"severity": "error", "name": name, "reason": "definition draft inputs must be a non-empty object"})
    else:
        for input_name, input_spec in inputs.items():
            if not isinstance(input_name, str) or not input_name:
                findings.append({"severity": "error", "name": name, "reason": "definition draft input name must be a non-empty string"})
                continue
            if not isinstance(input_spec, dict):
                findings.append({"severity": "error", "name": name, "reason": f"definition draft input {input_name} must be an object"})
                continue
            if "shape" not in input_spec or "dtype" not in input_spec:
                findings.append({"severity": "error", "name": name, "reason": f"definition draft input {input_name} must declare shape and dtype"})
            elif input_spec.get("shape") == []:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": f"definition draft input {input_name} uses shape []; use null for scalar inputs",
                })
    if not isinstance(data.get("outputs"), dict):
        findings.append({"severity": "error", "name": name, "reason": "definition draft outputs must be an object"})
    reference = data.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        findings.append({"severity": "error", "name": name, "reason": "definition draft reference must be a non-empty string"})
    elif not _reference_has_top_level_run(reference):
        findings.append({"severity": "error", "name": name, "reason": "definition draft reference must define top-level run(...)"})
    try:
        Definition.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - surface pydantic schema errors to the agent
        findings.append({
            "severity": "error",
            "name": name,
            "reason": (
                "definition draft does not match formal Definition schema: "
                f"{_format_definition_schema_error(exc)}. "
                "Use axes {\"type\":\"var\"} or {\"type\":\"const\",\"value\":N}; "
                "shape lists must reference axis names, not raw numbers; scalar shape must be null."
            ),
        })
    return findings


def _format_definition_schema_error(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            items = errors()
        except Exception:  # noqa: BLE001 - fall back to string form
            items = None
        if isinstance(items, list) and items:
            parts: list[str] = []
            for item in items[:3]:
                if not isinstance(item, dict):
                    continue
                loc = ".".join(str(part) for part in item.get("loc", ()))
                msg = str(item.get("msg") or "invalid")
                parts.append(f"{loc}: {msg}" if loc else msg)
            if parts:
                suffix = f"; +{len(items) - len(parts)} more" if len(items) > len(parts) else ""
                return "; ".join(parts) + suffix
    return str(exc).splitlines()[0]


def _reference_has_top_level_run(reference: str) -> bool:
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == "run" for node in tree.body)


def _check_hint_object(
    *,
    path: Path,
    name: str,
    op_type: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    try:
        data = _load_json(path)
    except Exception as exc:  # noqa: BLE001 - report agent-facing diagnostics
        return [{"severity": "error", "name": name, "reason": f"definition hints draft is not readable JSON: {exc}"}]
    if not isinstance(data, dict):
        return [{"severity": "error", "name": name, "reason": "definition hints draft must be a JSON object"}]
    missing = sorted(field for field in HINT_REQUIRED_FIELDS if field not in data)
    if missing:
        findings.append({"severity": "error", "name": name, "reason": f"definition hints draft missing fields: {missing}"})
    if data.get("definition_name") != name:
        findings.append({"severity": "error", "name": name, "reason": f"definition hints name mismatch: {data.get('definition_name')!r}"})
    if data.get("op_type") != op_type:
        findings.append({"severity": "error", "name": name, "reason": f"definition hints op_type mismatch: {data.get('op_type')!r}"})
    if type(data.get("schema_version")) is not int:
        findings.append({"severity": "error", "name": name, "reason": "definition hints schema_version must be an integer"})
    inputs = data.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        findings.append({"severity": "error", "name": name, "reason": "definition hints inputs must be a non-empty object"})
    return findings


def _model_prefixes(model_slug: str | None) -> set[str]:
    if not model_slug:
        return set()
    normalized = re.sub(r"[^a-z0-9]+", "_", model_slug.lower()).strip("_")
    parts = [part for part in normalized.split("_") if part]
    prefixes = {normalized}
    if parts and len(parts[0]) >= 3:
        prefixes.add(parts[0])
    for size in range(2, len(parts) + 1):
        prefixes.add("_".join(parts[:size]))
    return {prefix for prefix in prefixes if prefix}


def _check_definition_name_style(*, name: str, op_type: str, model_slug: str | None) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name):
        findings.append({
            "severity": "error",
            "name": name,
            "reason": "definition_name must be lower snake_case with no punctuation",
        })
    for prefix in _model_prefixes(model_slug):
        if name == prefix or name.startswith(f"{prefix}_"):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": f"definition_name must not include model prefix {prefix!r}; use a stable op/shape name instead",
            })
            break
    if op_type == "rmsnorm":
        if not re.fullmatch(r"(?:fused_add_)?rmsnorm_h[0-9]+", name):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "rmsnorm definition_name must be rmsnorm_h<N> or fused_add_rmsnorm_h<N>",
            })
    elif op_type == "silu_and_mul":
        if not re.fullmatch(r"silu_and_mul_i[0-9]+", name):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "silu_and_mul definition_name must be silu_and_mul_i<N>",
            })
    elif not (name == op_type or name.startswith(f"{op_type}_")):
        findings.append({
            "severity": "warning",
            "name": name,
            "reason": f"definition_name should normally start with op_type {op_type!r}",
        })
    return findings


def _check_non_fitrace_definition_drafts(*, proposal_dir: Path, candidates_path: Path) -> dict[str, Any]:
    """Check review-only definition/hints drafts for non-fitrace agent targets."""
    return _check_non_fitrace_definition_drafts_with_model(
        proposal_dir=proposal_dir,
        candidates_path=candidates_path,
        model_slug=None,
    )


def _check_non_fitrace_definition_drafts_with_model(
    *,
    proposal_dir: Path,
    candidates_path: Path,
    model_slug: str | None,
) -> dict[str, Any]:
    """Check review-only definition/hints drafts for non-fitrace agent targets."""
    candidates = _load_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError(f"candidate targets must be a list: {candidates_path}")

    checked: list[dict[str, str]] = []
    findings: list[dict[str, str]] = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        if item.get("role", "target") != "target":
            continue
        if item.get("backend") == "flashinfer" or item.get("definition_source") != "agent":
            continue
        candidate_name = str(item.get("name") or f"#{index}")
        definition_name = item.get("definition_name")
        op_type = item.get("op_type")
        if not isinstance(definition_name, str) or not definition_name:
            findings.append({
                "severity": "error",
                "name": candidate_name,
                "reason": "non-fitrace agent target must declare definition_name for draft checking",
            })
            continue
        if not isinstance(op_type, str) or not op_type:
            findings.append({
                "severity": "error",
                "name": candidate_name,
                "reason": "non-fitrace agent target must declare op_type for draft checking",
            })
            continue
        findings.extend(
            {"severity": item["severity"], "name": candidate_name, "reason": item["reason"]}
            for item in _check_definition_name_style(name=definition_name, op_type=op_type, model_slug=model_slug)
        )
        definition_path = _definition_draft_path(proposal_dir, op_type, definition_name)
        hints_path = _definition_hint_path(proposal_dir, op_type, definition_name)
        checked.append({
            "name": candidate_name,
            "definition_name": definition_name,
            "op_type": op_type,
            "definition_path": str(definition_path),
            "hints_path": str(hints_path),
        })
        if not definition_path.exists():
            findings.append({
                "severity": "error",
                "name": candidate_name,
                "reason": f"missing review-only definition draft: {definition_path}",
            })
        else:
            findings.extend(
                {"severity": item["severity"], "name": candidate_name, "reason": item["reason"]}
                for item in _check_definition_object(path=definition_path, name=definition_name, op_type=op_type)
            )
        if not hints_path.exists():
            findings.append({
                "severity": "error",
                "name": candidate_name,
                "reason": f"missing review-only definition hints draft: {hints_path}",
            })
        else:
            findings.extend(
                {"severity": item["severity"], "name": candidate_name, "reason": item["reason"]}
                for item in _check_hint_object(path=hints_path, name=definition_name, op_type=op_type)
            )

    errors = sum(1 for item in findings if item["severity"] == "error")
    warnings = sum(1 for item in findings if item["severity"] == "warning")
    return {
        "summary": {
            "draft_targets": len(checked),
            "errors": errors,
            "warnings": warnings,
            "ok": errors == 0,
        },
        "proposal_dir": str(proposal_dir),
        "checked": checked,
        "findings": findings,
    }
