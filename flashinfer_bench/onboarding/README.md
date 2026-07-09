# Model Onboarding

`flashinfer_bench.onboarding` is a workflow for discovering model runtime targets, collecting FlashInfer and non-FI workload artifacts from SGLang runs, and validating the generated FlashInfer-Bench trace layout.

Workflow:

0. Generate and check the merged review-only proposal:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools first-pass-loop \
     --model <hf_model> \
     --count 3 \
     --max-rounds 3 \
     --agent codex
   ```

   This prepares local inputs, runs the first-pass agents, merges their proposals, runs `check-proposal`, and invokes `check-repair-loop` if the merged proposal is not ready. It stops before human review; it does not promote config or run Modal.

1. For debugging, the same steps can be run separately:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools prepare-agent-inputs \
     --model <hf_model>

   python3 -B -m flashinfer_bench.onboarding.proposal_tools spawn-agents \
     --model <hf_model> \
     --count 3 \
     --agent codex

   python3 -B -m flashinfer_bench.onboarding.proposal_tools check-proposal \
     --proposal-dir runs/<model>/<run_id>_merged/proposal \
     --hf-config agent_inputs/config/<model_slug>.json
   ```

2. Review the proposal. For each accepted target in `proposal/candidate_targets.json`, set `status` to `approved`; leave rejected or undecided targets unapproved.

   Promote the approved targets into runtime config:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools promote-approved \
     --run <model>/<run_id>
   ```

   `promote-approved` writes `config/approved_targets.json` by copying only runtime target fields. For approved `definition_source=agent` targets, it also copies the matching `proposal/definitions/` and `proposal/definition_hints/` JSON files into `config/`. Proposal-only fields such as `status`, `evidence`, and `review_note` are dropped automatically.

3. Run collect:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.cli run --run <model>/<run_id>
   ```

4. Validate the completed run:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.cli validate --run <model>/<run_id>
   ```

5. If static proposal checks fail before runtime, generate check repair input:

   ```bash
   python3 -B -m flashinfer_bench.onboarding.proposal_tools check-repair-loop --proposal-dir runs/<model>/<run_id>/proposal --hf-config agent_inputs/config/<model_slug>.json
   ```

6. If the run exposes proposal-level issues, generate run repair input:

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
- `.claude/skills/review-onboarding-proposal/`: agent skill for initial and repair-pass proposal generation.
- `tests/onboarding/`: unit tests for the onboarding workflow.

Runtime inputs are explicit reviewed files. The onboarding runtime does not infer model runtime settings from model names, hidden defaults, or legacy external definition directories.
