"""Generate initial proposal prompts and optionally run external agents."""

from __future__ import annotations

import tempfile

from ..common import *  # noqa: F403
from ..checks.fitrace import _required_sglang_engine_kwargs
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
    """Return the fixed initial prompt for one review-only agent run."""
    proposal_dir = run_dir / "proposal"
    hints = "\n".join(f"  - {item}" for item in sglang_model_hints) if sglang_model_hints else "  - none provided"
    runtime_guidance: list[str] = []
    hf_config = _load_json(hf_config_path) if hf_config_path.exists() else {}
    if isinstance(hf_config, dict):
        required_engine_kwargs = _required_sglang_engine_kwargs(hf_config)
        if required_engine_kwargs:
            runtime_guidance.extend([
                "Runtime config startup requirement:",
                "",
                "```json",
                json.dumps({"engine_kwargs": required_engine_kwargs}, indent=2, ensure_ascii=False),
                "```",
                "",
                "Include these `engine_kwargs` in `config/run_config.json`. They are required before SGLang can load this HF config.",
                "",
            ])
    check_cmd = "\n".join([
        "python3 -B -m flashinfer_bench.onboarding.proposal_tools check-proposal \\",
        f"  --proposal-dir {_display_path(proposal_dir)} \\",
        f"  --hf-config {_display_path(hf_config_path)} \\",
        f"  --flashinfer-root {_display_path(flashinfer_root)}",
    ])
    return _render_prompt_template(
        "first_pass.md",
        model_name=model_name,
        hf_config_path=_display_path(hf_config_path),
        sglang_root=_display_path(sglang_root),
        sglang_model_hints=hints,
        flashinfer_root=_display_path(flashinfer_root),
        cookbook_root=_display_path(cookbook_root),
        run_dir=_display_path(run_dir),
        architecture_path=_display_path(proposal_dir / "architecture.md"),
        candidate_targets_path=_display_path(proposal_dir / "candidate_targets.json"),
        review_checklist_path=_display_path(proposal_dir / "review_checklist.md"),
        definitions_path=_display_path(proposal_dir / "definitions/..."),
        definition_hints_path=_display_path(proposal_dir / "definition_hints/..."),
        run_config_path=_display_path(run_dir / "config" / "run_config.json"),
        runtime_guidance="\n".join(runtime_guidance).rstrip(),
        non_fi_definition_rules=_non_fitrace_definition_rules_markdown(),
        check_cmd=check_cmd,
    )


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
    """Generate initial proposal prompts and optionally run N external agents."""
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
        artifacts_dir = proposal_dir / "agent_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = artifacts_dir / "first_pass_prompt.md"
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
            stdout_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
            stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
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
                    f"pid={proc.pid}",
                    flush=True,
                )
            processes.append({
                "process": proc,
                "stdin": prompt_text,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
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
            prompt = proc_info["prompt"]
            result = {
                "run_dir": prompt["run_dir"],
                "proposal_dir": prompt["proposal_dir"],
                "returncode": returncode,
            }
            agent_results.append(result)
            if returncode != 0:
                proc_info["stdout_file"].seek(0)
                proc_info["stderr_file"].seek(0)
                stdout_text = proc_info["stdout_file"].read()
                stderr_text = proc_info["stderr_file"].read()
                if stdout_text.strip():
                    print(
                        f"[spawn-agents] agent {prompt['index']} stdout:\n{stdout_text}",
                        file=sys.stdout,
                        flush=True,
                    )
                if stderr_text.strip():
                    print(
                        f"[spawn-agents] agent {prompt['index']} stderr:\n{stderr_text}",
                        file=sys.stderr,
                        flush=True,
                    )
            proc_info["stdout_file"].close()
            proc_info["stderr_file"].close()
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
    return result
