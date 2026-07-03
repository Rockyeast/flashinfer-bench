"""Proposal deterministic gate and agent feedback loop."""

from __future__ import annotations

from .common import *  # noqa: F403
from .checks.candidate import _check_candidate_fields
from .checks.drafts import _check_non_fitrace_definition_drafts
from .checks.fitrace import _check_run_config, _evaluate_fitrace_targets
from .checks.merge_report import _check_merge_report

def check_proposal(
    *,
    candidates_path: Path,
    hf_config_path: Path,
    proposal_dir: Path | None = None,
    flashinfer_root: Path | None = None,
) -> dict[str, Any]:
    """Run all proposal checks in one review gate."""
    if proposal_dir is None:
        proposal_dir = candidates_path.parent
    candidate_fields = _check_candidate_fields(candidates_path)
    hf_config = _load_json(hf_config_path)
    if not isinstance(hf_config, dict):
        raise ValueError(f"HF config must be a JSON object: {hf_config_path}")
    run_config = _check_run_config(
        proposal_dir=proposal_dir,
        hf_config=hf_config,
    )
    fitrace_eval = _evaluate_fitrace_targets(
        candidates_path=candidates_path,
        hf_config_path=hf_config_path,
        flashinfer_root=flashinfer_root,
    )
    definition_drafts = _check_non_fitrace_definition_drafts(
        proposal_dir=proposal_dir,
        candidates_path=candidates_path,
    )
    merge_report = _check_merge_report(proposal_dir)
    findings = [
        {"check": "candidate_fields", **item}
        for item in candidate_fields["findings"]
    ]
    findings.extend({"check": "run_config", **item} for item in run_config["findings"])
    findings.extend({"check": "fitrace_target", **item} for item in fitrace_eval["findings"])
    findings.extend({"check": "definition_draft", **item} for item in definition_drafts["findings"])
    findings.extend({"check": "merge_report", **item} for item in merge_report["findings"])
    errors = sum(1 for item in findings if item["severity"] == "error")
    warnings = sum(1 for item in findings if item["severity"] == "warning")
    return {
        "summary": {
            "entries": candidate_fields["summary"]["entries"],
            "collect_candidates": fitrace_eval["summary"]["collect_candidates"],
            "importable_targets": fitrace_eval["summary"]["importable_targets"],
            "fitrace_targets": fitrace_eval["summary"]["fitrace_targets"],
            "definition_draft_targets": definition_drafts["summary"]["draft_targets"],
            "merge_conflicts": merge_report["summary"]["conflicts"],
            "errors": errors,
            "warnings": warnings,
            "ok": errors == 0,
        },
        "candidates_path": str(candidates_path),
        "hf_config_path": str(hf_config_path),
        "flashinfer_root": str(flashinfer_root) if flashinfer_root is not None else None,
        "candidate_fields": candidate_fields,
        "run_config": run_config,
        "fitrace_eval": fitrace_eval,
        "definition_drafts": definition_drafts,
        "merge_report": merge_report,
        "findings": findings,
    }


def _proposal_feedback_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    findings = report.get("findings", [])
    status = "PASS" if summary["ok"] else "FIX_REQUIRED"
    lines = [
        "# Agent Proposal Feedback",
        "",
        f"- status: {status}",
        f"- entries: {summary['entries']}",
        f"- collect candidates: {summary['collect_candidates']}",
        f"- fitrace targets: {summary['fitrace_targets']}",
        f"- non-fitrace definition drafts: {summary['definition_draft_targets']}",
        f"- merge conflicts: {summary.get('merge_conflicts', 0)}",
        f"- errors: {summary['errors']}",
        f"- warnings: {summary['warnings']}",
        "",
    ]
    if summary["ok"]:
        lines.extend([
            "## Next Action",
            "",
            "Proposal checks passed. Stop revising and hand the proposal to human review.",
            "",
        ])
        return "\n".join(lines)

    lines.extend([
        "## Next Agent Action",
        "",
        "Revise only the proposal bundle. Do not edit config/approved_targets.json, do not run Modal, and do not approve candidates automatically.",
        "",
        "## Findings",
        "",
    ])
    for item in findings:
        severity = item.get("severity", "unknown")
        check = item.get("check", "unknown")
        name = item.get("name", "unknown")
        reason = item.get("reason", "unknown")
        lines.append(f"- {severity} [{check}] {name}: {reason}")

    suggestions = [
        item
        for item in report.get("fitrace_eval", {}).get("targets", [])
        if isinstance(item, dict) and item.get("suggested_target")
    ]
    if suggestions:
        lines.extend(["", "## Suggested Fitrace Targets", ""])
        for item in suggestions:
            lines.append(f"- {item.get('name')}: {item['suggested_target']}")
    lines.append("")
    return "\n".join(lines)


def run_agent_loop(
    *,
    proposal_dir: Path,
    hf_config_path: Path,
    candidates_path: Path | None = None,
    flashinfer_root: Path | None = None,
) -> dict[str, Any]:
    """Run one deterministic proposal-check loop and write agent feedback.

    The loop intentionally does not call an LLM. The external agent reads the
    generated feedback, edits the proposal bundle, and invokes this command
    again until the check passes.
    """
    proposal_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidates_path or proposal_dir / "candidate_targets.json"
    check_report = check_proposal(
        proposal_dir=proposal_dir,
        candidates_path=candidates,
        hf_config_path=hf_config_path,
        flashinfer_root=flashinfer_root,
    )
    check_path = proposal_dir / "proposal_check.json"
    feedback_path = proposal_dir / "agent_feedback.md"
    loop_path = proposal_dir / "agent_loop.json"
    _write_json(check_path, check_report)
    feedback_path.write_text(_proposal_feedback_markdown(check_report), encoding="utf-8")
    result = {
        "summary": {
            "ok": check_report["summary"]["ok"],
            "ready_for_human_review": check_report["summary"]["ok"],
            "errors": check_report["summary"]["errors"],
            "warnings": check_report["summary"]["warnings"],
        },
        "proposal_dir": str(proposal_dir),
        "candidates_path": str(candidates),
        "hf_config_path": str(hf_config_path),
        "flashinfer_root": str(flashinfer_root) if flashinfer_root is not None else None,
        "outputs": {
            "check_report": str(check_path),
            "feedback": str(feedback_path),
            "loop_report": str(loop_path),
        },
    }
    _write_json(loop_path, result)
    return result
