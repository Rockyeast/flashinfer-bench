"""Repair-pass proposal loop helpers."""

from __future__ import annotations

from ..common import *  # noqa: F403
from ..gate import run_agent_loop
from .diagnose import diagnose_run

def _repair_prompt_markdown(
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
    agent_loop_cmd = (
        "python3 -B -m flashinfer_bench.onboarding.proposal_tools agent-loop "
        f"--proposal-dir {proposal_dir} "
        f"--hf-config {hf_config_path}"
        f"{flashinfer_arg}"
    )
    lines = [
        "# Repair Prompt",
        "",
        "Use the review-onboarding-proposal skill in repair-pass mode to repair this review-only proposal.",
        "Do not restart first-pass onboarding for this run.",
        "",
        "## Scope",
        "",
        f"- run_dir: {run_dir}",
        f"- proposal_dir: {proposal_dir}",
        f"- diagnostics: {proposal_dir / 'agent_feedback.md'}",
        f"- hf_config: {hf_config_path}",
        f"- flashinfer_root: {flashinfer_root if flashinfer_root is not None else 'not provided'}",
        "",
        "## Hard Rules",
        "",
        "- This is repair-pass, not first-pass.",
        "- Edit only proposal artifacts: proposal/candidate_targets.json, proposal/architecture.md, proposal/review_checklist.md, proposal/definitions, and proposal/definition_hints.",
        "- Do not edit config/approved_targets.json, config/run_config.json, output/, reports/, or committed source code.",
        "- Do not approve candidates automatically.",
        "- Do not run Modal or any GPU job.",
        "- Fix the proposal so the deterministic checker passes, then stop for human review.",
        "",
        "## Current Diagnostics Summary",
        "",
        f"- status: {'PASS' if summary['ok'] else 'FIX_REQUIRED'}",
        f"- errors: {summary['errors']}",
        f"- warnings: {summary['warnings']}",
        f"- action_required: {summary.get('action_required', 0)}",
        "",
    ]
    findings = diagnostics.get("findings", [])
    if findings:
        lines.extend(["## Findings To Fix", ""])
        for item in findings:
            lines.append(
                f"- {item.get('severity', 'unknown')} [{item.get('source', 'unknown')}] "
                f"{item.get('name', 'unknown')}: {item.get('reason', 'unknown')}"
            )
        lines.append("")
    if check_result is not None:
        check_summary = check_result["summary"]
        lines.extend([
            "## Current Proposal Check Summary",
            "",
            f"- ready_for_human_review: {check_summary['ready_for_human_review']}",
            f"- errors: {check_summary['errors']}",
            f"- warnings: {check_summary['warnings']}",
            f"- feedback: {check_result['outputs']['feedback']}",
            f"- report: {check_result['outputs']['loop_report']}",
            "",
        ])
    lines.extend([
        "## Required Check",
        "",
        "After editing the proposal, run:",
        "",
        "```bash",
        agent_loop_cmd,
        "```",
        "",
        "Repeat proposal edits only until `ready for human review: True`, then stop.",
        "",
    ])
    return "\n".join(lines)


def repair_loop(
    *,
    run: Path,
    hf_config_path: Path,
    flashinfer_root: Path | None = None,
    agent_command: list[str] | None = None,
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
    prompt_path = proposal_dir / "repair_prompt.md"
    diagnostics_result: dict[str, Any] | None = None
    check_result: dict[str, Any] | None = None
    diagnostics_path: Path | None = None
    diagnostics: dict[str, Any] | None = None
    agent_rounds: list[dict[str, Any]] = []

    for round_index in range(1, max_rounds + 1):
        print(f"[repair-loop] round {round_index}/{max_rounds}: running diagnostics ...", flush=True)
        diagnostics_result = diagnose_run(run=run_dir)
        diagnostics_path = Path(diagnostics_result["outputs"]["diagnostics"])
        loaded = _load_json(diagnostics_path)
        if not isinstance(loaded, dict):
            raise ValueError(f"diagnostics must be a JSON object: {diagnostics_path}")
        diagnostics = loaded

        check_result = None
        if diagnostics_result["summary"]["ok"]:
            print(f"[repair-loop] round {round_index}/{max_rounds}: diagnostics ok, running proposal check ...", flush=True)
            check_result = run_agent_loop(
                proposal_dir=proposal_dir,
                hf_config_path=hf_config_path,
                flashinfer_root=flashinfer_root,
            )

        prompt_text = _repair_prompt_markdown(
            run_dir=run_dir,
            proposal_dir=proposal_dir,
            hf_config_path=hf_config_path,
            flashinfer_root=flashinfer_root,
            diagnostics=diagnostics,
            check_result=check_result,
        )
        prompt_path.write_text(prompt_text, encoding="utf-8")

        ready = bool(
            diagnostics_result["summary"]["ok"]
            and check_result
            and check_result["summary"]["ready_for_human_review"]
        )
        if ready or not agent_command or round_index >= max_rounds:
            break
        print(f"[repair-loop] round {round_index}/{max_rounds}: invoking agent ...", flush=True)
        completed = subprocess.run(
            agent_command,
            check=False,
            text=True,
            input=prompt_text,
        )
        print(f"[repair-loop] round {round_index}/{max_rounds}: agent exited (returncode={completed.returncode})", flush=True)
        agent_round = {
            "round": round_index,
            "command": agent_command,
            "returncode": completed.returncode,
        }
        agent_rounds.append(agent_round)
        if completed.returncode != 0:
            break

    if diagnostics_result is None or diagnostics_path is None:
        raise RuntimeError("repair loop did not run")
    check_summary = check_result["summary"] if check_result else diagnostics_result["summary"]
    ready_for_human_review = bool(
        diagnostics_result["summary"]["ok"]
        and check_result
        and check_result["summary"]["ready_for_human_review"]
    )
    result = {
        "summary": {
            "diagnostics_ok": diagnostics_result["summary"]["ok"],
            "diagnostics_action_required": diagnostics_result["summary"].get("action_required", 0),
            "ready_for_human_review": ready_for_human_review,
            "errors": check_summary["errors"],
            "warnings": check_summary["warnings"],
            "agent_ran": bool(agent_rounds),
            "rounds": round_index,
            "max_rounds": max_rounds,
        },
        "run_dir": str(run_dir),
        "outputs": {
            "diagnostics": str(diagnostics_path),
            "feedback": diagnostics_result["outputs"]["feedback"],
            "repair_prompt": str(prompt_path),
            "agent_loop": check_result["outputs"]["loop_report"] if check_result else None,
        },
        "agent": agent_rounds[-1] if agent_rounds else None,
        "agent_rounds": agent_rounds,
    }
    _write_json(proposal_dir / "repair_loop.json", result)
    print(f"[repair-loop] done: ready={ready_for_human_review}, errors={check_summary['errors']}, warnings={check_summary['warnings']}", flush=True)
    print(f"[repair-loop] repair_loop.json -> {proposal_dir / 'repair_loop.json'}", flush=True)
    return result
