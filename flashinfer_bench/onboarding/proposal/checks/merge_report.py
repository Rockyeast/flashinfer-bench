"""Merged proposal conflict checks."""

from __future__ import annotations

from ..common import *  # noqa: F403

def _check_merge_report(proposal_dir: Path) -> dict[str, Any]:
    path = proposal_dir / "merge_report.json"
    if not path.exists():
        return {"summary": {"conflicts": 0, "errors": 0, "ok": True}, "findings": []}
    report = _load_json(path)
    if not isinstance(report, dict):
        return {
            "summary": {"conflicts": 1, "errors": 1, "ok": False},
            "findings": [{
                "severity": "error",
                "name": str(path),
                "reason": "merge_report.json must be a JSON object",
            }],
        }
    conflicts = report.get("conflicts")
    findings: list[dict[str, str]] = []
    if isinstance(conflicts, list):
        for index, item in enumerate(conflicts):
            if not isinstance(item, dict):
                findings.append({
                    "severity": "error",
                    "name": f"merge_conflict_{index}",
                    "reason": "unresolved merge conflict",
                })
                continue
            name = str(item.get("path") or item.get("key") or f"merge_conflict_{index}")
            reason = str(item.get("reason") or "unresolved merge conflict")
            fields = item.get("fields")
            if isinstance(fields, list) and fields:
                reason = f"{reason}; fields={fields}"
            findings.append({"severity": "error", "name": name, "reason": reason})
    errors = len(findings)
    return {
        "summary": {
            "conflicts": len(conflicts) if isinstance(conflicts, list) else 0,
            "errors": errors,
            "ok": errors == 0,
        },
        "path": str(path),
        "findings": findings,
    }
