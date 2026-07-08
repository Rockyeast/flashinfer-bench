# Onboarding User Guide

This document covers the reviewed workflow for onboarding a model and validating a trace run.

## Quick Start

Use this path for a new model. Replace `<hf_model>` and `<model_slug>` with your target model values.

### CLI Command Reference

| Command | Who Runs It | Purpose |
| --- | --- | --- |
| `run` | human main entry | Run the reviewed runtime pipeline from `config/` and write `output/` + `reports/`. |
| `validate` | human follow-up | Re-check an existing run and refresh `reports/`. |
| `prepare-agent-inputs` | human; `spawn-agents` can reuse its outputs | Prepare HF config and source/cookbook inputs for proposal generation. |
| `spawn-agents` | human | Generate first-pass proposal prompts and optionally invoke external agents. |
| `merge-proposals` | human; `spawn-agents --count > 1` can call it | Merge multiple proposal drafts and record conflicts. |
| `check-proposal` | read-only check; repair loops call the same gate | Validate proposal artifacts and update `review_checklist.md`. |
| `check-repair-loop` | common pre-run repair command | Run static proposal check, generate `check_repair_prompt.md` on failure, optionally invoke an agent, then re-check. |
| `diagnose-run` | manual debug entry; `run-repair-loop` calls `diagnose_run()` internally | Convert `reports/run_report.json` failures into the `review_checklist.md` tool status block. |
| `run-repair-loop` | common post-run repair command | Diagnose a failed run, generate `run_repair_prompt.md`, optionally invoke an agent, then re-check the proposal. |
| `promote-approved` | human after review | Copy approved proposal targets and non-FI drafts into reviewed `config/`. |

### 1. Prepare Inputs

Prepare the local agent input cache:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools prepare-agent-inputs \
  --model <hf_model>
```

This downloads `agent_inputs/config/<model_slug>.json` and links `agent_inputs/sgl-cookbook/` to the shared `.onboarding_cache/sgl-cookbook` cache.

```bash
# Required after prepare-agent-inputs.
ls agent_inputs/config/<model_slug>.json

# Optional but recommended for static source checks.
ls agent_inputs/flashinfer/flashinfer
ls agent_inputs/sglang/python/sglang

# Required for final validate.
pip install -e /path/to/flashinfer-bench
```

`prepare-agent-inputs` checks whether the FlashInfer and SGLang source roots exist, but it does not clone them. Prepare those source snapshots separately if you want stronger agent/source checks.

### 2. Generate Proposal

Generate one proposal prompt:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools spawn-agents \
  --model <hf_model>
```

Run Codex automatically:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools spawn-agents \
  --model <hf_model> \
  --agent codex
```

For multiple independent agents:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools spawn-agents \
  --model <hf_model> \
  --count 3 \
  --agent codex
```

`spawn-agents` writes each agent's prompt/run under `runs/<model>/<date>_firstpass...`. With `--count > 1`, it creates sibling agent runs and merges proposals into a review-only merged proposal.

This stage may leave these files. `spawn-agents` writes the prompt file; when an external agent command fails, its stdout/stderr are printed to the current terminal instead of being stored under `proposal/`.

```text
runs/<model>/<run_id>/
  proposal/
    architecture.md
    first_pass_prompt.md
    candidate_targets.json
    review_checklist.md
    definitions/
    definition_hints/
  config/
    run_config.json
```

The agent is expected to run `check-proposal` until the proposal is ready for human review.

### 3. Review And Approve

Review:

- `proposal/architecture.md`
- `proposal/candidate_targets.json`
- `proposal/review_checklist.md`
- `config/run_config.json`
- `proposal/definitions/` and `proposal/definition_hints/` for non-FI targets

Approve targets by marking accepted entries in `proposal/candidate_targets.json`:

```json
{
  "name": "gqa_decode",
  "status": "approved",
  "evidence": [
    {
      "kind": "source_location",
      "value": "flashinfer/decode.py:BatchDecodeWithPagedKVCacheWrapper.run"
    }
  ],
  "review_note": "Verified SGLang route reaches this wrapper."
}
```

The human approval action is the `status: approved` change. Then promote approved targets:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools promote-approved \
  --run <model>/<run_id>
```

`promote-approved` writes `config/approved_targets.json` with only runtime fields. For approved `definition_source=agent` targets, it also copies the matching `proposal/definitions/` and `proposal/definition_hints/` JSON files into `config/`. It drops proposal-only review fields such as `status`, `evidence`, and `review_note`.

Reviewed artifacts live under:

```text
runs/<model>/<run_id>/config/
  approved_targets.json
  run_config.json
  definitions/
  definition_hints/
```

Minimum approval checklist:

- Every approved target has explicit `target`, `module`, and `attr`.
- FlashInfer collect targets point to APIs decorated with `@flashinfer_api(trace=...)`.
- Attention wrappers usually target the decorated `.run`; use `.forward` only as a companion when it passes arguments into `.run`.
- `definition_name` for fitrace-backed targets is only a preview; the final name comes from the fitrace dump.
- Known collectable non-FI ops, currently `rmsnorm` and `silu_and_mul`, have review-only definition/hints drafts before approval.
- Non-FI drafts are promoted automatically for approved `definition_source=agent` targets.

Proposal tools do not approve anything. `check-proposal`, `check-repair-loop`, and `run-repair-loop` only validate or repair proposal artifacts.

### 4. Run Collect

```bash
python3 -B -m flashinfer_bench.onboarding.cli run \
  --run <model>/<run_id>
```

`run` executes the reviewed collect path:

```text
remote SGLang probe
-> hook events/captures
-> FlashInfer fitrace dump
-> definition audit/repair
-> workload collect
-> local materialization
```

If the local terminal disconnects, copy the Function call ID from Modal and resume:

```bash
python3 -B -m flashinfer_bench.onboarding.cli run \
  --run <model>/<run_id> \
  --resume-call-id fc-...
```

### 5. Validate

```bash
python3 -B -m flashinfer_bench.onboarding.cli validate \
  --run <model>/<run_id>
```

`validate` is the final acceptance command. It runs local consistency checks, validator layout/export checks, and `flashinfer-bench validate` with GPU disabled. The run is accepted only when it prints:

```text
run accepted: True
```

Review:

```text
runs/<model>/<run_id>/reports/review.md
runs/<model>/<run_id>/reports/run_report.json
```

### 6. Repair If Needed

Before runtime, use `check-repair-loop` when the static proposal check fails:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools check-repair-loop \
  --proposal-dir runs/<model>/<run_id>/proposal \
  --hf-config agent_inputs/config/<model_slug>.json \
  --flashinfer-root agent_inputs/flashinfer/flashinfer
```

Do not edit `output/` directly. Use `run-repair-loop` to update the proposal checklist status and optionally invoke an external agent:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools run-repair-loop \
  --run <model>/<run_id> \
  --hf-config agent_inputs/config/<model_slug>.json \
  --flashinfer-root agent_inputs/flashinfer/flashinfer
```

Automatic repair with an external agent:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools run-repair-loop \
  --run <model>/<run_id> \
  --hf-config agent_inputs/config/<model_slug>.json \
  --flashinfer-root agent_inputs/flashinfer/flashinfer \
  --max-rounds 3 \
  --agent-command "codex exec -C <REPO_ROOT> -s workspace-write --ephemeral" -
```

`run-repair-loop` only repairs `proposal/`; it does not edit `config/`, does not edit `output/`, and does not run Modal. After repair, human-review/promote again, then rerun collect and validate.

## Reference

### Run Directory

Each run lives under:

```text
runs/<model>/<run_id>/
```

Directory layout:

```text
runs/<model>/<run_id>/
  proposal/
    architecture.md
    candidate_targets.json
    review_checklist.md
    merge_review.md
    merge_report.json
    agent_artifacts/
      first_pass_prompt.md
      check_repair_prompt.md
      run_repair_prompt.md
    definitions/
    definition_hints/
  config/
    approved_targets.json
    run_config.json
    definitions/
    definition_hints/
  output/
    definitions/
    workloads/
    blob/
  reports/
    run_report.json
    review.md
```

- `proposal/` is review-only agent output.
- `config/` is human-reviewed input consumed by runtime.
- `output/` is validator-ready staging data produced by the run.
- `reports/` contains the machine-readable run report and human review digest.

### Run Config

Minimal `config/run_config.json`:

```json
{
  "model_name": "<hf_model>",
  "image": "lmsysorg/sglang:v0.5.12.post1",
  "gpu": "L40S",
  "tp_size": 1,
  "timeout": 3600,
  "disable_cuda_graph": true,
  "batch_sizes": [1, 2, 4, 8, 16, 32, 64],
  "max_new_tokens": 96,
  "max_captures_per_target": 128,
  "supplemental_runs": [
    {
      "name": "sampling_supplemental",
      "sampling_params": {"temperature": 0.7, "top_k": 50, "top_p": 0.9},
      "allowed_op_types": ["sampling"]
    }
  ]
}
```

Runtime choices and collect strategy are reviewed config. The core does not infer GPU, TP, image, or workload coverage from the model name.

Collect uses `sharegpt_100.json` from the working directory when present; otherwise it uses the packaged onboarding prompt fixture. Remote prompt scenarios are derived from `batch_sizes` and `max_new_tokens`.

### Definition Sources

Definition sources are explicit:

- FlashInfer targets use the fitrace dump produced during the same Modal inference run.
- Reviewed non-FI definitions live under `config/definitions/`.
- Accepted definitions are staged under `output/definitions/`.

### Proposal Commands

Manual proposal check:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools check-proposal \
  --proposal-dir runs/<model>/<run_id>/proposal \
  --hf-config agent_inputs/config/<model_slug>.json \
  --flashinfer-root agent_inputs/flashinfer/flashinfer
```

Manual multi-agent merge:

```bash
python3 -B -m flashinfer_bench.onboarding.proposal_tools merge-proposals \
  --proposal-dir runs/<model>/<agent_run_a>/proposal \
  --proposal-dir runs/<model>/<agent_run_b>/proposal \
  --proposal-dir runs/<model>/<agent_run_c>/proposal \
  --output-dir runs/<model>/<merged_run>/proposal
```

`merge-proposals` deduplicates candidates and unions evidence. Conflicts are written to `merge_review.md` and `merge_report.json`; they must be resolved by review.

### Result Artifacts

Human review usually starts here:

```text
reports/review.md
```

Machine-readable details live here:

```text
reports/run_report.json
```

Reviewable outputs:

```text
output/definitions/
output/workloads/
output/blob/
```

Captures are raw argument snapshots created when hooks fire. They are intermediate artifacts used by definition audit/repair and sanitization. Standard local results do not retain captures long-term.

## Troubleshooting

### Early Stop

Normal collect stops early when a target fails audit/sanitize. Later targets may show zero events because they were not executed. That does not mean those targets are invalid, and they should not be changed to `collect: false` just because of early stop.

`run-repair-loop` detects early-stop cases and includes the reason in the proposal checklist status.

To gather more diagnostics in one run:

```bash
python3 -B -m flashinfer_bench.onboarding.cli run \
  --run <model>/<run_id> \
  --diagnostic-full-scan
```

Diagnostic full scan keeps running after early failures to expose more issues. Its captures are for debugging, not final collect.

### Incremental Collect

The same run directory can be passed to `run` multiple times. Each round adds or overwrites the specified targets while preserving data for other targets.

- To skip a target and keep existing data: set `collect: false`.
- To re-collect a target and replace existing data: keep `collect: true`.
- The workload manifest is merged automatically.

### Modal Resume

If local execution disconnects after Modal has started, resume with the Modal Function call ID:

```bash
python3 -B -m flashinfer_bench.onboarding.cli run \
  --run <model>/<run_id> \
  --resume-call-id fc-...
```

Resume only materializes the existing remote result. It does not launch a second SGLang run.
