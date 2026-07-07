"""Convert completed run reports into proposal checklist status."""

from __future__ import annotations

from ..common import *  # noqa: F403

def _append_findings_from_items(
    findings: list[dict[str, Any]],
    *,
    source: str,
    severity: str,
    items: Any,
) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        reason = item.get("reason") or item.get("error") or item
        if item.get("reason") == "sanitize_failed" and isinstance(item.get("sanitize_reject_reasons"), dict):
            reason = f"sanitize_failed: {item['sanitize_reject_reasons']}"
            if isinstance(item.get("sanitize_reject_examples"), dict):
                reason = f"{reason}; examples: {item['sanitize_reject_examples']}"
        findings.append({
            "source": source,
            "severity": str(item.get("severity") or severity),
            "name": str(item.get("name") or item.get("definition_name") or item.get("source_name") or "unknown"),
            "reason": _brief(reason),
        })


def _ignored_definition_reasons(proposal_dir: Path) -> dict[str, str]:
    path = proposal_dir / "ignored_definitions.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"ignored definitions must be a JSON list: {path}")
    ignored: dict[str, str] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"ignored definition #{index} must be an object")
        name = item.get("name")
        reason = item.get("reason")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"ignored definition #{index} has no name")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"ignored definition {name} has no reason")
        ignored[name.strip()] = reason.strip()
    return ignored


def _proposed_definition_candidates(proposal_dir: Path) -> dict[str, str]:
    path = proposal_dir / "candidate_targets.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, list):
        return {}
    proposed: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict) or item.get("collect") is not True:
            continue
        if item.get("status") == "rejected":
            continue
        definition_name = item.get("definition_name")
        if not isinstance(definition_name, str) or not definition_name.strip():
            continue
        candidate_name = item.get("name")
        proposed[definition_name.strip()] = (
            candidate_name.strip()
            if isinstance(candidate_name, str) and candidate_name.strip()
            else definition_name.strip()
        )
    return proposed


def _uncollected_definition_findings(
    *,
    items: Any,
    ignored: dict[str, str],
    proposed: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    findings: list[dict[str, Any]] = []
    ignored_items: list[dict[str, str]] = []
    proposed_items: list[dict[str, str]] = []
    if not isinstance(items, list):
        return findings, ignored_items, proposed_items
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("definition_name") or item.get("source_name") or "unknown")
        reason = _brief(item.get("reason") or item.get("details") or item.get("error") or "uncollected definition")
        candidate_name = proposed.get(name)
        if candidate_name:
            proposed_items.append({
                "name": name,
                "candidate": candidate_name,
            })
            continue
        ignore_reason = ignored.get(name)
        if ignore_reason:
            ignored_items.append({
                "name": name,
                "reason": _brief(ignore_reason),
            })
            continue
        findings.append({
            "source": "definition_audit.uncollected",
            "severity": "action_required",
            "name": name,
            "reason": reason,
        })
    return findings, ignored_items, proposed_items


def diagnose_run(*, run: Path) -> dict[str, Any]:
    """Convert a completed run report into the review checklist tool status."""
    run_dir = _resolve_run_dir(run)
    proposal_dir = run_dir / "proposal"
    report_path = run_dir / "reports" / "run_report.json"
    proposal_dir.mkdir(parents=True, exist_ok=True)

    findings: list[dict[str, Any]] = []
    if not report_path.exists():
        findings.append({
            "source": "run_report",
            "severity": "error",
            "name": str(run_dir),
            "reason": f"missing run report: {report_path}",
        })
        report: dict[str, Any] = {}
    else:
        report = _load_json(report_path)
        if not isinstance(report, dict):
            findings.append({
                "source": "run_report",
                "severity": "error",
                "name": str(report_path),
                "reason": "run report must be a JSON object",
            })
            report = {}

    remote = report.get("remote") if isinstance(report.get("remote"), dict) else {}
    remote_summary = remote.get("summary") if isinstance(remote.get("summary"), dict) else {}
    early_stopped = bool(remote_summary.get("early_stopped"))
    early_stop_reason = remote_summary.get("early_stop_reason") or ""

    parse_report = report.get("parse_report") if isinstance(report.get("parse_report"), dict) else {}
    for name in parse_report.get("missing_targets", []) if isinstance(parse_report.get("missing_targets"), list) else []:
        if early_stopped:
            reason = (
                f"approved target had zero non-warmup events because the run was early-stopped "
                f"(reason: {early_stop_reason}). This target was NOT reached before the stop — "
                f"do NOT set collect:false. Fix the early-stop cause first, then re-run."
            )
            severity = "warning"
        else:
            reason = "approved target had zero non-warmup events; check hook target, runtime route, probe_mode, or run_config coverage"
            severity = "error"
        findings.append({
            "source": "parse_report",
            "severity": severity,
            "name": str(name),
            "reason": reason,
        })

    definition_audit = report.get("definition_audit") if isinstance(report.get("definition_audit"), dict) else {}
    _append_findings_from_items(
        findings,
        source="definition_audit.rejected",
        severity="error",
        items=definition_audit.get("rejected"),
    )
    diagnostics_section = report.get("diagnostics") if isinstance(report.get("diagnostics"), dict) else {}
    ignored_definitions = _ignored_definition_reasons(proposal_dir)
    proposed_definitions = _proposed_definition_candidates(proposal_dir)
    uncollected_findings, ignored_uncollected, proposed_uncollected = _uncollected_definition_findings(
        items=diagnostics_section.get("uncollected_definitions"),
        ignored=ignored_definitions,
        proposed=proposed_definitions,
    )
    findings.extend(uncollected_findings)

    collect = report.get("collect") if isinstance(report.get("collect"), dict) else {}
    manifest = collect.get("manifest") if isinstance(collect.get("manifest"), dict) else {}
    _append_findings_from_items(
        findings,
        source="collect.skipped",
        severity="error",
        items=[
            item
            for item in manifest.get("skipped", [])
            if isinstance(item, dict) and item.get("reason") != "collect is false"
        ] if isinstance(manifest.get("skipped"), list) else [],
    )
    workloads = manifest.get("workloads")
    if isinstance(workloads, list):
        for workload in workloads:
            if not isinstance(workload, dict):
                continue
            reject_reasons = workload.get("reject_reasons")
            sanitized_count = workload.get("sanitized_count")
            if isinstance(reject_reasons, dict) and reject_reasons:
                findings.append({
                    "source": "workload_sanitize",
                    "severity": "error",
                    "name": str(workload.get("name") or workload.get("definition_name") or "unknown"),
                    "reason": f"sanitize rejects: {reject_reasons}",
                })
            if sanitized_count == 0:
                findings.append({
                    "source": "workload_sanitize",
                    "severity": "error",
                    "name": str(workload.get("name") or workload.get("definition_name") or "unknown"),
                    "reason": "no captures were converted into workload entries",
                })

    internal = report.get("internal_validation") if isinstance(report.get("internal_validation"), dict) else {}
    _early_stop_suppressed_reasons = {
        "expected collect target has no workload",
        "approved target not observed in non-warmup events",
        "parse report marked target missing",
    }
    internal_findings = internal.get("findings")
    if early_stopped and isinstance(internal_findings, list):
        internal_findings = [
            {**item, "severity": "warning", "reason": f"[suppressed: early-stop] {item.get('reason', '')}"}
            if item.get("severity") == "error" and item.get("reason") in _early_stop_suppressed_reasons
            else item
            for item in internal_findings
            if isinstance(item, dict)
        ]
    _append_findings_from_items(
        findings,
        source="internal_validation",
        severity="error",
        items=internal_findings,
    )

    dataset_validation = report.get("dataset_validation") if isinstance(report.get("dataset_validation"), dict) else {}
    if dataset_validation and not dataset_validation.get("ok", False):
        stdout = dataset_validation.get("stdout")
        stderr = dataset_validation.get("stderr")
        reason = _brief(stdout or stderr or f"returncode={dataset_validation.get('returncode')}", limit=1200)
        findings.append({
            "source": "dataset_validation",
            "severity": "error",
            "name": "dataset_validator",
            "reason": reason,
        })

    errors = sum(1 for item in findings if item.get("severity") == "error")
    warnings = sum(1 for item in findings if item.get("severity") == "warning")
    action_required = sum(1 for item in findings if item.get("severity") == "action_required")
    diagnostics = {
        "summary": {
            "ok": errors == 0 and action_required == 0,
            "errors": errors,
            "warnings": warnings,
            "action_required": action_required,
            "findings": len(findings),
        },
        "run_dir": str(run_dir),
        "run_report": str(report_path),
        "findings": findings,
    }
    if ignored_uncollected:
        diagnostics["ignored_uncollected_definitions"] = ignored_uncollected
    if proposed_uncollected:
        diagnostics["proposed_uncollected_definitions"] = proposed_uncollected
    review_path = _write_review_tool_status(proposal_dir, _run_diagnostics_tool_status_markdown(diagnostics))
    return {
        "summary": diagnostics["summary"],
        "run_dir": str(run_dir),
        "diagnostics": diagnostics,
        "outputs": {
            "review_checklist": str(review_path),
        },
    }


def _run_diagnostics_tool_status_markdown(diagnostics: dict[str, Any]) -> str:
    summary = diagnostics["summary"]
    status = "PASS" if summary["ok"] else "FIX_REQUIRED"
    lines = [
        "## Tool Status",
        "",
        f"- status: {status}",
        f"- source: run diagnostics",
        f"- run: {diagnostics['run_dir']}",
        f"- errors: {summary['errors']}",
        f"- warnings: {summary['warnings']}",
        f"- action_required: {summary.get('action_required', 0)}",
        "",
    ]
    if summary["ok"]:
        lines.extend([
            "## Next Action",
            "",
            "Run diagnostics found no proposal-level failures. Stop revising and hand the result to human review.",
            "",
        ])
        proposed = diagnostics.get("proposed_uncollected_definitions")
        if isinstance(proposed, list) and proposed:
            lines.extend([
                "## Proposed Uncollected Definitions",
                "",
                "These definitions were uncollected in the previous run, but the current proposal now contains collect candidates. After human approval, rerun the pipeline.",
                "",
            ])
            for item in proposed:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('name', 'unknown')}: proposed by {item.get('candidate', 'unknown')}")
            lines.append("")
        ignored = diagnostics.get("ignored_uncollected_definitions")
        if isinstance(ignored, list) and ignored:
            lines.extend(["## Ignored Uncollected Definitions", ""])
            for item in ignored:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('name', 'unknown')}: {item.get('reason', '')}")
            lines.append("")
        return "\n".join(lines)

    lines.extend([
        "## Next Agent Action",
        "",
        "Revise only the review-only proposal bundle. Prefer fixing proposal/definitions and proposal/definition_hints for non-FI issues, or candidate_targets/run_config when the failure is a missing target or wrong route. Do not approve targets and do not run Modal.",
        "",
        "## Findings",
        "",
    ])
    for item in diagnostics.get("findings", []):
        lines.append(
            f"- {item.get('severity', 'unknown')} [{item.get('source', 'unknown')}] "
            f"{item.get('name', 'unknown')}: {item.get('reason', 'unknown')}"
        )
    lines.append("")
    return "\n".join(lines)
