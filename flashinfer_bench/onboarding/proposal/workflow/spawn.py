"""Generate first-pass prompts and optionally run external agents."""

from __future__ import annotations

from ..common import *  # noqa: F403
from ..checks.fitrace import _sglang_config_compat_engine_kwargs
from .merge import merge_proposals

def _agent_suffix(index: int) -> str:
    if index < len(string.ascii_lowercase):
        return string.ascii_lowercase[index]
    return str(index + 1)


def _agent_run_dir(run_prefix: Path, *, count: int, index: int) -> Path:
    if count == 1:
        return run_prefix
    return run_prefix.with_name(f"{run_prefix.name}_agent_{_agent_suffix(index)}")


def first_pass_prompt_markdown(
    *,
    model_name: str,
    run_dir: Path,
    hf_config_path: Path,
    sglang_root: Path,
    flashinfer_root: Path,
    cookbook_root: Path,
    sglang_model_hints: list[str],
) -> str:
    """Return the fixed first-pass prompt for one review-only agent run."""
    proposal_dir = run_dir / "proposal"
    hints = "\n".join(f"  - {item}" for item in sglang_model_hints) if sglang_model_hints else "  - none provided"
    runtime_guidance: list[str] = []
    hf_config = _load_json(hf_config_path) if hf_config_path.exists() else {}
    if isinstance(hf_config, dict):
        compat_kwargs = _sglang_config_compat_engine_kwargs(hf_config)
        if compat_kwargs:
            runtime_guidance.extend([
                "Runtime config compatibility requirement:",
                "",
                "```json",
                json.dumps({"engine_kwargs": compat_kwargs}, indent=2, ensure_ascii=False),
                "```",
                "",
                "Include these `engine_kwargs` in `config/run_config.json`. They are required before SGLang can load this HF config.",
                "",
            ])
    return "\n".join([
        f"# {model_name} First-Pass Proposal",
        "",
        "Use the `review-onboarding-proposal` skill to generate a review-only proposal for:",
        "",
        "```text",
        model_name,
        "```",
        "",
        "Working directory:",
        "",
        "```text",
        ".",
        "```",
        "",
        "Inputs:",
        "",
        "```text",
        f"HF config: {_display_path(hf_config_path)}",
        f"SGLang source root: {_display_path(sglang_root)}",
        "SGLang model implementation hints:",
        hints,
        f"FlashInfer source root: {_display_path(flashinfer_root)}",
        f"sgl-cookbook root: {_display_path(cookbook_root)}",
        "diagnostics: omit; first-pass",
        f"run dir: {_display_path(run_dir)}",
        "```",
        "",
        "Outputs:",
        "",
        "```text",
        f"{_display_path(proposal_dir / 'architecture.md')}",
        f"{_display_path(proposal_dir / 'candidate_targets.json')}",
        f"{_display_path(proposal_dir / 'review_checklist.md')}",
        f"{_display_path(proposal_dir / 'definitions/...')} only for non-FI definition_source=agent drafts",
        f"{_display_path(proposal_dir / 'definition_hints/...')} only for non-FI definition_source=agent drafts",
        f"{_display_path(run_dir / 'config' / 'run_config.json')}",
        "```",
        "",
        *runtime_guidance,
        "Strictly follow `.claude/skills/review-onboarding-proposal/SKILL.md`.",
        "",
        "Do not apply or approve anything. Do not write `config/approved_targets.json`.",
        "Do not write official `output/definitions`, `output/workloads`, or `output/blob`.",
        "Do not run Modal, collect, validate, or commit.",
        "",
        "After writing the proposal, run:",
        "",
        "```bash",
        "python3 -B -m flashinfer_bench.onboarding.proposal_tools agent-loop \\",
        f"  --proposal-dir {_display_path(proposal_dir)} \\",
        f"  --hf-config {_display_path(hf_config_path)} \\",
        f"  --flashinfer-root {_display_path(flashinfer_root)}",
        "```",
        "",
        "If `agent_feedback.md` reports `FIX_REQUIRED`, revise only the review-only",
        "proposal bundle and repeat the same deterministic check until it prints:",
        "",
        "```text",
        "ready for human review: True",
        "```",
        "",
    ])


def spawn_agents(
    *,
    model_name: str,
    run_prefix: Path,
    hf_config_path: Path,
    sglang_root: Path,
    flashinfer_root: Path,
    cookbook_root: Path,
    sglang_model_hints: list[str],
    count: int = 1,
    agent_command: list[str] | None = None,
    agent_env: dict[str, str] | None = None,
    merge_output_dir: Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Generate first-pass prompts and optionally run N external agents."""
    if count < 1:
        raise ValueError("count must be >= 1")
    run_base = _resolve_run_dir(run_prefix)
    prompts: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []

    for index in range(count):
        run_dir = _agent_run_dir(run_base, count=count, index=index)
        proposal_dir = run_dir / "proposal"
        config_dir = run_dir / "config"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        prompt_text = first_pass_prompt_markdown(
            model_name=model_name,
            run_dir=run_dir,
            hf_config_path=hf_config_path,
            sglang_root=sglang_root,
            flashinfer_root=flashinfer_root,
            cookbook_root=cookbook_root,
            sglang_model_hints=sglang_model_hints,
        )
        prompt_path = proposal_dir / "first_pass_prompt.md"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        item = {
            "index": index + 1,
            "run_dir": str(run_dir),
            "proposal_dir": str(proposal_dir),
            "prompt": str(prompt_path),
        }
        prompts.append(item)
        if progress:
            print(f"[spawn-agents] prompt {index + 1}/{count}: {prompt_path}", flush=True)

        if agent_command:
            stdout_path = proposal_dir / "agent_stdout.log"
            stderr_path = proposal_dir / "agent_stderr.log"
            stdout_file = stdout_path.open("w", encoding="utf-8")
            stderr_file = stderr_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                agent_command,
                text=True,
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                env=agent_env,
            )
            if progress:
                print(
                    f"[spawn-agents] started agent {index + 1}/{count}: "
                    f"pid={proc.pid} stdout={stdout_path} stderr={stderr_path}",
                    flush=True,
                )
            processes.append({
                "process": proc,
                "stdin": prompt_text,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "prompt": item,
            })

    for proc_info in processes:
        proc_info["process"].stdin.write(proc_info["stdin"])
        proc_info["process"].stdin.close()

    agent_results: list[dict[str, Any]] = []
    pending = list(processes)
    last_progress = 0.0
    while pending:
        remaining: list[dict[str, Any]] = []
        for proc_info in pending:
            proc = proc_info["process"]
            returncode = proc.poll()
            if returncode is None:
                remaining.append(proc_info)
                continue
            proc_info["stdout_file"].close()
            proc_info["stderr_file"].close()
            prompt = proc_info["prompt"]
            result = {
                "run_dir": prompt["run_dir"],
                "proposal_dir": prompt["proposal_dir"],
                "returncode": returncode,
                "stdout": str(proc_info["stdout_path"]),
                "stderr": str(proc_info["stderr_path"]),
            }
            agent_results.append(result)
            if progress:
                status = "ok" if returncode == 0 else "failed"
                print(
                    f"[spawn-agents] finished agent {prompt['index']}/{count}: "
                    f"{status} returncode={returncode} proposal={prompt['proposal_dir']}",
                    flush=True,
                )
        pending = remaining
        if pending:
            now = time.monotonic()
            if progress and now - last_progress >= 30:
                running = ", ".join(
                    f"{item['prompt']['index']}(pid={item['process'].pid})"
                    for item in pending
                )
                print(f"[spawn-agents] still running: {running}", flush=True)
                last_progress = now
            time.sleep(1)

    merge_report: dict[str, Any] | None = None
    if merge_output_dir is not None:
        if progress:
            print(f"[spawn-agents] merging proposals -> {merge_output_dir}", flush=True)
        merge_report = merge_proposals(
            proposal_dirs=[Path(item["proposal_dir"]) for item in prompts],
            output_dir=merge_output_dir,
        )

    result = {
        "summary": {
            "model": model_name,
            "count": count,
            "agents_started": len(processes),
            "agent_failures": sum(1 for item in agent_results if item["returncode"] != 0),
            "merged": merge_report is not None,
            "merge_ok": merge_report["summary"]["ok"] if merge_report is not None else None,
        },
        "run_prefix": str(run_base),
        "prompts": prompts,
        "agent_results": agent_results,
        "merge_report": str(merge_output_dir / "merge_report.json") if merge_output_dir is not None else None,
    }
    report_path = run_base.parent / f"{run_base.name}_spawn_agents.json"
    _write_json(report_path, result)
    result["report"] = str(report_path)
    if progress:
        print(f"[spawn-agents] report: {report_path}", flush=True)
    return result
