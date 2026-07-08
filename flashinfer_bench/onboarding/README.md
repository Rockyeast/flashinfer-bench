# Model Onboarding

`flashinfer_bench.onboarding` is a workflow for discovering model runtime targets, collecting FlashInfer and non-FI workload artifacts from SGLang runs, and validating the generated FlashInfer-Bench trace layout.

Workflow:

0. Prepare local model/source inputs if they are missing:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools prepare-agent-inputs \
     --model <hf_model>
   ```

1. Generate a review-only proposal:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools spawn-agents \
     --model <hf_model> \
     --agent codex
   ```

   The generated prompt tells the agent to follow `.claude/skills/review-onboarding-proposal/`. For another non-interactive agent CLI, use `--agent-command`.

2. Optionally run the deterministic proposal check before review:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools check-proposal \
     --proposal-dir runs/<model>/<run_id>/proposal \
     --hf-config agent_inputs/config/<model_slug>.json \
     --flashinfer-root agent_inputs/flashinfer/flashinfer
   ```

3. Review the proposal. For each accepted target in `proposal/candidate_targets.json`, set `status` to `approved`; leave rejected or undecided targets unapproved.

   Promote the approved targets into runtime config:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools promote-approved \
     --run <model>/<run_id>
   ```

   `promote-approved` writes `config/approved_targets.json` by copying only runtime target fields. For approved `definition_source=agent` targets, it also copies the matching `proposal/definitions/` and `proposal/definition_hints/` JSON files into `config/`. Proposal-only fields such as `status`, `evidence`, and `review_note` are dropped automatically.

4. Run collect:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.cli run --run <model>/<run_id>
   ```

5. Validate the completed run:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.cli validate --run <model>/<run_id>
   ```

6. If static proposal checks fail before runtime, generate check repair input:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools check-repair-loop --proposal-dir runs/<model>/<run_id>/proposal --hf-config agent_inputs/config/<model_slug>.json
   ```

7. If the run exposes proposal-level issues, generate run repair input:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools run-repair-loop --run <model>/<run_id>
   ```

See `user_guide.md` for the full command reference, run directory layout, and troubleshooting notes.

## Module Layout

- `core/`: planning, hook capture, event/workload sanitization, and definition audit.
- `definition_repairs/`: kernel-specific definition repair and hint rules.
- `runners/`: Modal/SGLang runner integration.
- `cli.py`: command-line entrypoints for collect and validation.
- `proposal_tools.py`: proposal checks, multi-agent merge, diagnostics, and repair-loop helpers.
- `.claude/skills/review-onboarding-proposal/`: agent skill for first-pass and repair-pass proposal generation.
- `tests/onboarding/`: unit tests for the onboarding workflow.

Runtime inputs are explicit reviewed files. The onboarding runtime does not infer model runtime settings from model names, hidden defaults, or legacy external definition directories.
