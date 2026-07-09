"""CLI entry point for offline proposal onboarding tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flashinfer_bench.onboarding.proposal.common import (
    DEFAULT_COOKBOOK_CACHE_ROOT,
    DEFAULT_COOKBOOK_REPO,
    DEFAULT_COOKBOOK_ROOT,
    DEFAULT_FLASHINFER_ROOT,
    DEFAULT_SGLANG_ROOT,
    _agent_command_from_shortcut,
    _default_hf_config_path,
    _default_merge_output_dir,
    _default_run_prefix,
)
from flashinfer_bench.onboarding.proposal.gate import run_proposal_gate
from flashinfer_bench.onboarding.proposal.workflow.diagnose import diagnose_run
from flashinfer_bench.onboarding.proposal.workflow.merge import merge_proposals
from flashinfer_bench.onboarding.proposal.workflow.prepare import prepare_agent_inputs
from flashinfer_bench.onboarding.proposal.workflow.promote import promote_approved_targets
from flashinfer_bench.onboarding.proposal.workflow.repair import check_repair_loop, run_repair_loop
from flashinfer_bench.onboarding.proposal.workflow.spawn import spawn_agents


def _add_prepare_agent_inputs_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-agent-inputs",
        help="Refresh local HF config and sgl-cookbook material for review-only onboarding.",
    )
    parser.add_argument("--model", action="append", required=True, help="HF model name. Can be passed multiple times.")
    parser.add_argument("--output-root", type=Path, default=Path("agent_inputs"))
    parser.add_argument("--refresh", action="store_true", help="Update cookbook and re-download configs.")
    parser.add_argument("--cookbook-repo", default=DEFAULT_COOKBOOK_REPO)
    parser.add_argument("--cookbook-cache", type=Path, default=DEFAULT_COOKBOOK_CACHE_ROOT)
    parser.add_argument("--check-sglang-root", type=Path, default=Path("agent_inputs/sglang/python/sglang"))
    parser.add_argument("--check-flashinfer-root", type=Path, default=Path("agent_inputs/flashinfer/flashinfer"))


def _add_check_proposal_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "check-proposal",
        help="Run deterministic checks for a full proposal bundle. Does not invoke an agent.",
    )
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        help="Path to flashinfer/ source root for static fitrace checks.",
    )


def _add_diagnose_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "diagnose-run",
        help="Convert reports/run_report.json into the proposal/review_checklist.md tool status block.",
    )
    parser.add_argument("--run", type=Path, required=True, help="Run path or path relative to runs/.")


def _add_check_repair_loop_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "check-repair-loop",
        help="Repair proposal check failures before runtime run; optionally invoke an external agent.",
    )
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        help="Path to flashinfer/ source root for static fitrace checks.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum check/repair rounds when --agent-command is provided.",
    )
    parser.add_argument(
        "--agent",
        choices=["codex"],
        help="Shortcut external agent command. Currently supports codex.",
    )
    parser.add_argument(
        "--agent-command",
        nargs=argparse.REMAINDER,
        help="Optional external agent command. The check repair prompt is sent to stdin; pass this after check-repair-loop options.",
    )


def _add_run_repair_loop_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run-repair-loop",
        help="Repair proposal issues from run diagnostics; optionally invoke an external agent.",
    )
    parser.add_argument("--run", type=Path, required=True, help="Run path or path relative to runs/.")
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        help="Path to flashinfer/ source root for static fitrace checks.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum run repair/check rounds when --agent-command is provided.",
    )
    parser.add_argument(
        "--agent",
        choices=["codex"],
        help="Shortcut external agent command. Currently supports codex.",
    )
    parser.add_argument(
        "--agent-command",
        nargs=argparse.REMAINDER,
        help="Optional external agent command. The run repair prompt is sent to stdin; pass this after run-repair-loop options.",
    )


def _add_spawn_agents_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "spawn-agents",
        help="Generate initial proposal prompts and optionally run N external agents.",
    )
    parser.add_argument("--model", required=True, help="HF model name, for example microsoft/Phi-4-mini-instruct.")
    parser.add_argument(
        "--run-prefix",
        type=Path,
        help=(
            "Base run path or path relative to runs/. Defaults to <model_slug>/<YYYYMMDD>. "
            "With --count 1 this exact run is used; "
            "with --count N, sibling runs ending in _agent_a/_agent_b/... are used."
        ),
    )
    parser.add_argument("--hf-config", type=Path, help="Defaults to agent_inputs/config/<model_slug>.json.")
    parser.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    parser.add_argument("--flashinfer-root", type=Path, default=DEFAULT_FLASHINFER_ROOT)
    parser.add_argument("--cookbook-root", type=Path, default=DEFAULT_COOKBOOK_ROOT)
    parser.add_argument(
        "--sglang-model-hint",
        action="append",
        default=[],
        help="Relative SGLang source hint. Can be passed multiple times.",
    )
    parser.add_argument("--count", type=int, default=1, help="Number of initial proposal agents/prompts. Defaults to 1.")
    parser.add_argument(
        "--merge-output-dir",
        type=Path,
        help="Output proposal directory for merge-proposals after agents finish. Defaults to <run_prefix>_merged/proposal when --count > 1.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge multi-agent proposal outputs automatically.",
    )
    parser.add_argument(
        "--agent",
        choices=["codex"],
        help="Shortcut external agent command. Currently supports codex.",
    )
    parser.add_argument(
        "--agent-command",
        nargs=argparse.REMAINDER,
        help="Optional explicit external agent command. Overrides --agent. Each prompt is sent to stdin; pass this after spawn-agents options.",
    )


def _add_merge_proposals_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "merge-proposals",
        help="Union multiple review-only proposal bundles and report conflicts for human review.",
    )
    parser.add_argument(
        "--proposal-dir",
        type=Path,
        action="append",
        required=True,
        help="Proposal directory, or run directory containing proposal/. Pass at least two.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output proposal directory.")


def _add_promote_approved_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "promote-approved",
        help="Write config/approved_targets.json from proposal candidates marked status=approved.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="Run path or path relative to runs/.")
    source.add_argument("--proposal-dir", type=Path, help="Proposal directory, or run directory containing proposal/.")
    parser.add_argument("--config-dir", type=Path, help="Config directory. Defaults to <run>/config.")
    parser.add_argument("--output", type=Path, help="Output approved_targets.json path.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline proposal onboarding tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_agent_inputs_parser(subparsers)
    _add_check_proposal_parser(subparsers)
    _add_diagnose_run_parser(subparsers)
    _add_check_repair_loop_parser(subparsers)
    _add_run_repair_loop_parser(subparsers)
    _add_spawn_agents_parser(subparsers)
    _add_merge_proposals_parser(subparsers)
    _add_promote_approved_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare-agent-inputs":
        report = prepare_agent_inputs(
            models=args.model,
            output_root=args.output_root,
            refresh=args.refresh,
            cookbook_repo=args.cookbook_repo,
            cookbook_cache=args.cookbook_cache,
            check_sglang_root=args.check_sglang_root,
            check_flashinfer_root=args.check_flashinfer_root,
        )
        summary = report["summary"]
        print(f"models: {summary['models']}")
        print(f"configs ok: {summary['configs_ok']}")
        print(f"cookbook ok: {summary['cookbook_ok']}")
        print(f"cookbook matches: {summary['cookbook_matches']}")
        print(f"source checks ok: {summary['source_checks_ok']}")
        return 0

    if args.command == "check-proposal":
        result = run_proposal_gate(
            proposal_dir=args.proposal_dir,
            hf_config_path=args.hf_config,
            flashinfer_root=args.flashinfer_root,
        )
        summary = result["summary"]
        print(f"proposal check ok: {summary['ok']}")
        print(f"ready for human review: {summary['ready_for_human_review']}")
        print(f"errors: {summary['errors']}")
        print(f"warnings: {summary['warnings']}")
        print(f"proposal check: {result['outputs']['proposal_check']}")
        print(f"review checklist: {result['outputs']['review_checklist']}")
        return 0 if summary["ok"] else 1

    if args.command == "diagnose-run":
        result = diagnose_run(run=args.run)
        summary = result["summary"]
        print(f"run diagnostics ok: {summary['ok']}")
        print(f"errors: {summary['errors']}")
        print(f"warnings: {summary['warnings']}")
        print(f"action_required: {summary.get('action_required', 0)}")
        print(f"review checklist: {result['outputs']['review_checklist']}")
        return 0 if summary["ok"] else 1

    if args.command == "check-repair-loop":
        agent_command = args.agent_command
        agent_env = None
        if agent_command is None:
            agent_command, agent_env = _agent_command_from_shortcut(args.agent)
        result = check_repair_loop(
            proposal_dir=args.proposal_dir,
            hf_config_path=args.hf_config,
            flashinfer_root=args.flashinfer_root,
            agent_command=agent_command,
            agent_env=agent_env,
            max_rounds=args.max_rounds,
        )
        summary = result["summary"]
        print(f"ready for human review: {summary['ready_for_human_review']}")
        print(f"errors: {summary['errors']}")
        print(f"warnings: {summary['warnings']}")
        print(f"rounds: {summary['rounds']}/{summary['max_rounds']}")
        print(f"agent ran: {summary['agent_ran']}")
        print(f"proposal check: {result['outputs']['proposal_check']}")
        print(f"review checklist: {result['outputs']['review_checklist']}")
        print(f"check repair prompt: {result['outputs']['check_repair_prompt']}")
        return 0 if summary["ready_for_human_review"] or summary.get("needs_rerun", False) else 1

    if args.command == "run-repair-loop":
        agent_command = args.agent_command
        agent_env = None
        if agent_command is None:
            agent_command, agent_env = _agent_command_from_shortcut(args.agent)
        result = run_repair_loop(
            run=args.run,
            hf_config_path=args.hf_config,
            flashinfer_root=args.flashinfer_root,
            agent_command=agent_command,
            agent_env=agent_env,
            max_rounds=args.max_rounds,
        )
        summary = result["summary"]
        print(f"diagnostics ok: {summary['diagnostics_ok']}")
        print(f"diagnostics action_required: {summary.get('diagnostics_action_required', 0)}")
        print(f"proposal ready: {summary.get('proposal_ready', False)}")
        print(f"needs rerun: {summary.get('needs_rerun', False)}")
        print(f"ready for human review: {summary['ready_for_human_review']}")
        print(f"errors: {summary['errors']}")
        print(f"warnings: {summary['warnings']}")
        print(f"rounds: {summary['rounds']}/{summary['max_rounds']}")
        print(f"agent ran: {summary['agent_ran']}")
        print(f"run repair prompt: {result['outputs']['run_repair_prompt']}")
        print(f"review checklist: {result['outputs']['review_checklist']}")
        return 0 if summary["ready_for_human_review"] else 1

    if args.command == "spawn-agents":
        run_prefix = args.run_prefix or _default_run_prefix(args.model)
        hf_config_path = args.hf_config or _default_hf_config_path(args.model)
        agent_command = args.agent_command
        agent_env = None
        if agent_command is None:
            agent_command, agent_env = _agent_command_from_shortcut(args.agent)
        merge_output_dir = args.merge_output_dir
        if merge_output_dir is None and args.count > 1 and agent_command is not None and not args.no_merge:
            merge_output_dir = _default_merge_output_dir(run_prefix)
        result = spawn_agents(
            model_name=args.model,
            run_prefix=run_prefix,
            hf_config_path=hf_config_path,
            sglang_root=args.sglang_root,
            flashinfer_root=args.flashinfer_root,
            cookbook_root=args.cookbook_root,
            sglang_model_hints=args.sglang_model_hint,
            count=args.count,
            agent_command=agent_command,
            agent_env=agent_env,
            merge_output_dir=merge_output_dir,
            progress=True,
        )
        summary = result["summary"]
        print(f"model: {summary['model']}")
        print(f"prompts: {summary['count']}")
        print(f"agents started: {summary['agents_started']}")
        print(f"agent failures: {summary['agent_failures']}")
        for item in result["prompts"]:
            print(f"prompt: {item['prompt']}")
        if result["merge_report"]:
            print(f"merge report: {result['merge_report']}")
        ok = summary["agent_failures"] == 0
        if summary["merge_ok"] is not None:
            ok = ok and bool(summary["merge_ok"])
        return 0 if ok else 1

    if args.command == "merge-proposals":
        result = merge_proposals(
            proposal_dirs=args.proposal_dir,
            output_dir=args.output_dir,
        )
        summary = result["summary"]
        print(f"merged candidates: {summary['merged_candidates']}")
        print(f"conflicts: {summary['conflicts']}")
        print(f"candidate conflicts: {summary['candidate_conflicts']}")
        print(f"draft file conflicts: {summary['draft_file_conflicts']}")
        print(f"output: {args.output_dir}")
        print(f"review checklist: {args.output_dir / 'review_checklist.md'}")
        print(f"report: {args.output_dir / 'merge_report.json'}")
        return 0 if summary["ok"] else 1

    if args.command == "promote-approved":
        result = promote_approved_targets(
            run=args.run,
            proposal_dir=args.proposal_dir,
            config_dir=args.config_dir,
            output_path=args.output,
        )
        summary = result["summary"]
        print(f"approved targets: {summary['approved']}")
        print(f"skipped candidates: {summary['skipped']}")
        print(f"definitions: {summary['definitions']}")
        print(f"definition hints: {summary['definition_hints']}")
        print(f"output: {result['outputs']['approved_targets']}")
        return 0

    raise SystemExit(f"ERROR: unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
