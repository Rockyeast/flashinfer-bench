"""Candidate target proposal checks."""

from __future__ import annotations

from ..common import *  # noqa: F403

def _check_candidate_fields(path: Path) -> dict[str, Any]:
    """Return field findings for candidate/approved target JSON.

    This is a static field check for proposal hygiene. It does not import targets or
    decide approval automatically.
    """
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"candidate targets must be a list: {path}")

    findings: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            findings.append({
                "severity": "error",
                "name": f"#{index}",
                "reason": "entry is not a JSON object",
            })
            continue
        name = str(item.get("name") or f"#{index}")
        status = item.get("status")
        role = item.get("role", "target")
        backend = item.get("backend", "unknown")
        target = item.get("target")
        module = item.get("module")
        attr = item.get("attr")
        collect = item.get("collect", False)
        definition_source = item.get("definition_source", "unknown")
        definition_name = item.get("definition_name")
        op_type = item.get("op_type")
        raw_companions = item.get("companion_attrs", [])
        companion_attrs = raw_companions if isinstance(raw_companions, list) else []
        capture = item.get("capture")
        dispatch = item.get("dispatch")
        dispatch_value = item.get("dispatch_value")

        if not isinstance(collect, bool):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "collect must be a boolean",
            })

        if role not in {"target", "warmup"}:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "role must be target or warmup",
            })
            continue
        if role == "warmup":
            if not isinstance(module, str) or not module or not isinstance(attr, str) or not attr:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "warmup entry requires explicit module/attr",
                })
            if collect is True:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "warmup entry cannot use collect=true",
                })
            if isinstance(target, str) and target:
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "warmup entry ignores target; use module/attr as the reviewed hook spec",
                })
            if capture is not None:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "warmup entry must not declare capture",
                })
            if dispatch is not None:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "warmup entry must not declare dispatch",
                })
            continue

        if not isinstance(capture, dict):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "target entry must declare capture",
            })
        else:
            unexpected_capture = sorted(set(capture) - CAPTURE_SPEC_FIELDS)
            if unexpected_capture:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": f"capture has unexpected fields: {unexpected_capture}",
                })
        if item.get("page_size") is not None and not isinstance(dispatch, dict):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "page_size target must declare reviewed dispatch",
            })
        if isinstance(dispatch, dict) and item.get("page_size") is None and dispatch_value is None:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "non-page_size dispatch target must declare dispatch_value",
            })
        if dispatch_value is not None and type(dispatch_value) is not int:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "dispatch_value must be an integer",
            })

        if status == "approved" and not isinstance(target, str):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "approved entry has no hook target",
            })
        if (
            backend != "flashinfer"
            and isinstance(op_type, str)
            and op_type in KNOWN_NON_FITRACE_COLLECTABLE_OPS
        ):
            if collect is not True:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": f"known non-FlashInfer op {op_type} must be proposed with collect=true",
                })
            if definition_source != "agent":
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": f"known non-FlashInfer op {op_type} must use definition_source=agent",
                })
        if backend != "flashinfer" and collect is True:
            if not isinstance(definition_name, str) or not definition_name:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "non-FlashInfer collect target must declare reviewed definition_name",
                })
            if definition_source not in {"agent", "manual"}:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "non-FlashInfer collect target must use definition_source=agent or manual",
                })
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "non-FlashInfer collect target should include source evidence for reviewed definition/hints",
                })
        if backend == "flashinfer" and collect is True:
            if not isinstance(module, str) or not module or not isinstance(attr, str) or not attr:
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "collectable FlashInfer target should declare explicit module/attr hook spec",
                })
            if _requires_companion_attrs(target, attr) and not companion_attrs:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": (
                        "FlashInfer attention wrapper collect target must declare reviewed "
                        "companion_attrs so scalars like sm_scale are captured from sibling calls"
                    ),
                })
            if definition_source == "agent":
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "collectable FlashInfer target must not use agent-guessed final definition",
                })
            elif (
                status != "approved"
                and definition_source == "fitrace"
                and isinstance(definition_name, str)
                and definition_name
            ):
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "fitrace target has preview definition_name; final name must be verified from fitrace dump",
                })
            elif definition_source == "unknown":
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "collectable FlashInfer target should declare definition_source=fitrace or manual",
                })
        if definition_source == "fitrace" and not isinstance(target, str):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "fitrace definition source needs a FlashInfer API target",
            })

    summary = {
        "entries": len(raw),
        "errors": sum(1 for item in findings if item["severity"] == "error"),
        "warnings": sum(1 for item in findings if item["severity"] == "warning"),
    }
    summary["ok"] = summary["errors"] == 0
    return {"summary": summary, "path": str(path), "findings": findings}


def _requires_companion_attrs(target: Any, attr: Any) -> bool:
    """Return whether this proposal target needs reviewed companion captures."""
    candidates = [value for value in (target, attr) if isinstance(value, str)]
    return any(
        candidate.startswith(FLASHINFER_ATTENTION_TARGET_PREFIXES)
        and any(candidate.endswith(wrapper) for wrapper in COMPANION_REQUIRED_WRAPPER_SUFFIXES)
        for candidate in candidates
    )
