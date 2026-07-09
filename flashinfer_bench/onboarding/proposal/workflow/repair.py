"""Proposal repair loop helpers."""

from __future__ import annotations

from ..common import *  # noqa: F403
from ..gate import run_proposal_gate
from .diagnose import diagnose_run


def _findings_markdown(items: list[dict[str, Any]], *, source_field: str) -> str:
    if not items:
        return ""
    lines = ["## Findings To Fix", ""]
    for item in items:
        lines.append(
            f"- {item.get('severity', 'unknown')} [{item.get(source_field, 'unknown')}] "
            f"{item.get('name', 'unknown')}: {item.get('reason', 'unknown')}"
        )
    return "\n".join(lines) + "\n\n"


def _check_repair_prompt_markdown(
    *,
    proposal_dir: Path,
    hf_config_path: Path,
    flashinfer_root: Path | None,
    check_result: dict[str, Any],
) -> str:
    summary = check_result["summary"]
    flashinfer_arg = f" --flashinfer-root {flashinfer_root}" if flashinfer_root is not None else ""
    check_cmd = (
        "python3 -B -m flashinfer_bench.onboarding.proposal_tools check-proposal "
        f"--proposal-dir {proposal_dir} "
        f"--hf-config {hf_config_path}"
        f"{flashinfer_arg}"
    )
    findings = check_result.get("check_report", {}).get("findings", [])
    return _render_prompt_template(
        "check_repair.md",
        proposal_dir=proposal_dir,
        proposal_check=check_result["outputs"]["proposal_check"],
        review_checklist=check_result["outputs"]["review_checklist"],
        hf_config=hf_config_path,
        flashinfer_root=flashinfer_root if flashinfer_root is not None else "not provided",
        non_fi_definition_rules=_non_fitrace_definition_rules_markdown(),
        ready_for_human_review=summary["ready_for_human_review"],
        errors=summary["errors"],
        warnings=summary["warnings"],
        findings_block=_findings_markdown(findings, source_field="check"),
        check_cmd=check_cmd,
    )


def check_repair_loop(
    *,
    proposal_dir: Path,
    hf_config_path: Path,
    flashinfer_root: Path | None = None,
    agent_command: list[str] | None = None,
    agent_env: dict[str, str] | None = None,
    max_rounds: int = 1,
) -> dict[str, Any]:
    """Repair static proposal check failures before any runtime run."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    proposal_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = proposal_dir / "agent_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = artifacts_dir / "check_repair_prompt.md"
    agent_rounds: list[dict[str, Any]] = []
    check_result: dict[str, Any] | None = None

    for round_index in range(1, max_rounds + 1):
        print(f"[check-repair-loop] round {round_index}/{max_rounds}: running proposal check ...", flush=True)
        check_result = run_proposal_gate(
            proposal_dir=proposal_dir,
            hf_config_path=hf_config_path,
            flashinfer_root=flashinfer_root,
        )
        prompt_text = _check_repair_prompt_markdown(
            proposal_dir=proposal_dir,
            hf_config_path=hf_config_path,
            flashinfer_root=flashinfer_root,
            check_result=check_result,
        )
        prompt_path.write_text(prompt_text, encoding="utf-8")

        ready = bool(check_result["summary"]["ready_for_human_review"])
        if ready or not agent_command or round_index >= max_rounds:
            break
        print(f"[check-repair-loop] round {round_index}/{max_rounds}: invoking agent ...", flush=True)
        completed = subprocess.run(
            agent_command,
            check=False,
            text=True,
            input=prompt_text,
            env=agent_env,
        )
        print(f"[check-repair-loop] round {round_index}/{max_rounds}: agent exited (returncode={completed.returncode})", flush=True)
        agent_rounds.append({
            "round": round_index,
            "command": agent_command,
            "returncode": completed.returncode,
        })
        if completed.returncode != 0:
            break

    if check_result is None:
        raise RuntimeError("check repair loop did not run")
    result = {
        "summary": {
            "ready_for_human_review": check_result["summary"]["ready_for_human_review"],
            "errors": check_result["summary"]["errors"],
            "warnings": check_result["summary"]["warnings"],
            "agent_ran": bool(agent_rounds),
            "rounds": round_index,
            "max_rounds": max_rounds,
        },
        "proposal_dir": str(proposal_dir),
        "outputs": {
            "proposal_check": check_result["outputs"]["proposal_check"],
            "review_checklist": check_result["outputs"]["review_checklist"],
            "check_repair_prompt": str(prompt_path),
        },
        "agent": agent_rounds[-1] if agent_rounds else None,
        "agent_rounds": agent_rounds,
    }
    print(
        f"[check-repair-loop] done: ready={result['summary']['ready_for_human_review']}, "
        f"errors={result['summary']['errors']}, warnings={result['summary']['warnings']}",
        flush=True,
    )
    return result


def _run_repair_prompt_markdown(
    *,
    run_dir: Path,
    proposal_dir: Path,
    hf_config_path: Path,
    flashinfer_root: Path | None,
    diagnostics: dict[str, Any],
    check_result: dict[str, Any] | None = None,
) -> str:
    summary = diagnostics["summary"]
    flashinfer_arg = f" --flashinfer-root {flashinfer_root}" if flashinfer_root is not None else ""
    check_cmd = (
        "python3 -B -m flashinfer_bench.onboarding.proposal_tools check-proposal "
        f"--proposal-dir {proposal_dir} "
        f"--hf-config {hf_config_path}"
        f"{flashinfer_arg}"
    )
    diagnostic_findings_block = _findings_markdown(diagnostics.get("findings", []), source_field="source")
    proposal_check_block = ""
    if check_result is not None:
        check_summary = check_result["summary"]
        proposal_check_block = "\n".join([
            "## Current Proposal Check Summary",
            "",
            f"- ready_for_human_review: {check_summary['ready_for_human_review']}",
            f"- errors: {check_summary['errors']}",
            f"- warnings: {check_summary['warnings']}",
            f"- review_checklist: {check_result['outputs']['review_checklist']}",
            "",
        ])
    return _render_prompt_template(
        "run_repair.md",
        run_dir=run_dir,
        proposal_dir=proposal_dir,
        review_checklist=proposal_dir / "review_checklist.md",
        hf_config=hf_config_path,
        flashinfer_root=flashinfer_root if flashinfer_root is not None else "not provided",
        non_fi_definition_rules=_non_fitrace_definition_rules_markdown(),
        diagnostics_status="PASS" if summary["ok"] else "FIX_REQUIRED",
        diagnostics_errors=summary["errors"],
        diagnostics_warnings=summary["warnings"],
        diagnostics_action_required=summary.get("action_required", 0),
        diagnostic_findings_block=diagnostic_findings_block,
        proposal_check_block=proposal_check_block,
        check_cmd=check_cmd,
    )


def run_repair_loop(
    *,
    run: Path,
    hf_config_path: Path,
    flashinfer_root: Path | None = None,
    agent_command: list[str] | None = None,
    agent_env: dict[str, str] | None = None,
    max_rounds: int = 1,
) -> dict[str, Any]:
    """Generate repair prompts and optionally drive an external agent.

    This remains outside the runtime core. The optional agent command is a
    caller-provided executable that receives the repair prompt on stdin; this
    tool never approves config or runs Modal.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    run_dir = _resolve_run_dir(run)
    proposal_dir = run_dir / "proposal"
    artifacts_dir = proposal_dir / "agent_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = artifacts_dir / "run_repair_prompt.md"
    diagnostics_result: dict[str, Any] | None = None
    check_result: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    agent_rounds: list[dict[str, Any]] = []

    for round_index in range(1, max_rounds + 1):
        print(f"[run-repair-loop] round {round_index}/{max_rounds}: running diagnostics ...", flush=True)
        diagnostics_result = diagnose_run(run=run_dir)
        diagnostics = diagnostics_result["diagnostics"]

        print(f"[run-repair-loop] round {round_index}/{max_rounds}: running proposal check ...", flush=True)
        check_result = run_proposal_gate(
            proposal_dir=proposal_dir,
            hf_config_path=hf_config_path,
            flashinfer_root=flashinfer_root,
        )

        prompt_text = _run_repair_prompt_markdown(
            run_dir=run_dir,
            proposal_dir=proposal_dir,
            hf_config_path=hf_config_path,
            flashinfer_root=flashinfer_root,
            diagnostics=diagnostics,
            check_result=check_result,
        )
        prompt_path.write_text(prompt_text, encoding="utf-8")

        diagnostics_ok = bool(diagnostics_result["summary"]["ok"])
        diagnostics_action_required = int(diagnostics_result["summary"].get("action_required", 0) or 0)
        proposal_ready = bool(check_result["summary"]["ready_for_human_review"])
        ready = bool(diagnostics_ok and proposal_ready)
        needs_rerun = bool(
            proposal_ready
            and not diagnostics_ok
            and (
                diagnostics_action_required == 0
                or bool(agent_rounds)
            )
        )
        if ready or needs_rerun or not agent_command or round_index >= max_rounds:
            break
        print(f"[run-repair-loop] round {round_index}/{max_rounds}: invoking agent ...", flush=True)
        completed = subprocess.run(
            agent_command,
            check=False,
            text=True,
            input=prompt_text,
            env=agent_env,
        )
        print(f"[run-repair-loop] round {round_index}/{max_rounds}: agent exited (returncode={completed.returncode})", flush=True)
        agent_round = {
            "round": round_index,
            "command": agent_command,
            "returncode": completed.returncode,
        }
        agent_rounds.append(agent_round)
        if completed.returncode != 0:
            break

    if diagnostics_result is None:
        raise RuntimeError("repair loop did not run")
    if check_result is None:
        raise RuntimeError("repair loop did not run proposal check")
    check_summary = check_result["summary"]
    diagnostics_ok = bool(diagnostics_result["summary"]["ok"])
    diagnostics_action_required = int(diagnostics_result["summary"].get("action_required", 0) or 0)
    proposal_ready = bool(check_result["summary"]["ready_for_human_review"])
    ready_for_human_review = bool(
        diagnostics_ok
        and proposal_ready
    )
    needs_rerun = bool(
        proposal_ready
        and not diagnostics_ok
        and (
            diagnostics_action_required == 0
            or bool(agent_rounds)
        )
    )
    result = {
        "summary": {
            "diagnostics_ok": diagnostics_ok,
            "proposal_ready": proposal_ready,
            "needs_rerun": needs_rerun,
            "diagnostics_action_required": diagnostics_action_required,
            "ready_for_human_review": ready_for_human_review,
            "errors": check_summary["errors"],
            "warnings": check_summary["warnings"],
            "agent_ran": bool(agent_rounds),
            "rounds": round_index,
            "max_rounds": max_rounds,
        },
        "run_dir": str(run_dir),
        "outputs": {
            "review_checklist": diagnostics_result["outputs"]["review_checklist"],
            "proposal_check": check_result["outputs"]["proposal_check"],
            "run_repair_prompt": str(prompt_path),
        },
        "agent": agent_rounds[-1] if agent_rounds else None,
        "agent_rounds": agent_rounds,
    }
    print(f"[run-repair-loop] done: ready={ready_for_human_review}, errors={check_summary['errors']}, warnings={check_summary['warnings']}", flush=True)
    return result
