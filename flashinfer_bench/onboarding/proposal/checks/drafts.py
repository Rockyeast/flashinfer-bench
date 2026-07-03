"""Non-FI definition and hint draft checks."""

from __future__ import annotations

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
    return findings


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


def _check_non_fitrace_definition_drafts(*, proposal_dir: Path, candidates_path: Path) -> dict[str, Any]:
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
