---
name: onboard-model
description: Run the two-stage FlashInfer-Bench model onboarding flow: dump and review definitions, then dump and validate workloads.
---

# Onboard Model

Use the repository's two public onboarding commands. Do not create a parallel proposal,
target, merge, promote, or collect-plan workflow.

## Inputs

- model HuggingFace ID;
- run name below `runs/`;
- Modal GPU and optional image/TP/runtime overrides.

## Agent References

Definition analysis must read these current two-stage references before editing files:

- [`references/definition_standards.md`](references/definition_standards.md): canonical
  naming, schema, axes, and reference rules;
- [`references/non_fi_capture.md`](references/non_fi_capture.md): source evidence,
  capture-backend, and SGLang input-mapping rules.

Do not use the legacy `review-onboarding-proposal`, `collect-workloads`, or
`collect-workloads-bench` skills for this flow.

## Stage 1: Dump Definitions

```bash
flashinfer-bench onboarding dump-definition \
  --run {run_name} \
  --model-name {hf_model_id} \
  --gpu {modal_gpu}
```

This performs one short SGLang pass with FlashInfer definition tracing, executed-module
inventory, and SGLang output-logger comparison. The logger is enabled by default in the
same pass; use `--no-compare-sglang-logger` only when its overhead must be disabled. It
writes:

```text
runs/{run_name}/definitions/
runs/{run_name}/reports/definition_report.json
runs/{run_name}/reports/definition_review.md
runs/{run_name}/reports/evidence/sglang_execution_inventory.json
runs/{run_name}/reports/evidence/sglang_logger.json
```

Stop for human review. Inspect source evidence and edit `definitions/` in place. Do not
copy the files through a proposal/config promotion layer. A collectable definition must
use exactly one source-backed capture tag:

- `fi_api:<exact decorated callable>` for native FlashInfer logging;
- `sglang_module:<exact observed torch.nn.Module class>` for non-FI module inputs;
- `sglang_callable:<exact callable>` for a source-backed non-FI function.

Input names default to runtime argument names. Use
`sglang_input:<definition_input>=arg:<runtime_argument>` or
`sglang_input:<definition_input>=attr:<module_attribute>` only when an explicit mapping is
needed. SGLang's built-in tensor logger is output-signature coverage evidence, not exact
module identity or complete workload input capture.

Use the same command with `--agent codex` when Agent source analysis should write FI and
non-FI definitions directly:

```bash
flashinfer-bench onboarding dump-definition \
  --run {run_name} \
  --model-name {hf_model_id} \
  --gpu {modal_gpu} \
  --agent codex
```

The agent may edit only `definitions/`; it cannot approve evidence or run Modal. It must
not create a separate proposal or patch artifact.

## Stage 2: Dump Workloads

After the human accepts the current definition snapshot:

```bash
flashinfer-bench onboarding dump-workload \
  --run {run_name}
```

The command validates definitions before GPU submission, derives native FlashInfer logger
filters from `fi_api:` tags, captures reviewed non-FI modules/callables through the shared
`TracingRuntime`, runs SGLang, and runs the canonical dataset validator. It writes:

```text
runs/{run_name}/output/definitions/
runs/{run_name}/output/workloads/
runs/{run_name}/output/blob/
runs/{run_name}/reports/run_report.json
runs/{run_name}/reports/review.md
```

Stop for the second human review. A run is complete only when `run_report.json` has
`accepted: true` and the human accepts `review.md`.

For a failed collection, `--agent codex` may analyze the report and rewrite definitions.
Review the edits and rerun `dump-workload`; a changed definition digest invalidates the old
run snapshot. Do not hide a failed workload behind a warning.

## Rules

- `definitions/` is the only reviewed definition source.
- Running `dump-workload` approves exactly the SHA-256 snapshot recorded in the report.
- Never edit `output/` or `reports/` by hand.
- Never invent an `fi_api:` tag for a missing FlashInfer implementation.
- A missing FI kernel may use an observed `sglang_module:`/`sglang_callable:` capture point;
  otherwise it remains non-collectable.
- Use `--resume-call-id` after a local disconnect; do not start a duplicate remote call.

## Submission Gate

Workload validation and publication validation are separate. After adding one reference
test per new definition under `output/tests/references/`, run:

```bash
flashinfer-bench onboarding check-submission \
  --run {run_name} \
  --upstream-dataset tmp/flashinfer-trace
```

The gate requires complete descriptions, allowed publication tags/status, no conflicting
upstream definition, and a reference test for every new definition. Follow the compatible
ground-truth hierarchy from `add-reference-tests`: FlashInfer first, exact SGLang source
only when FlashInfer has no equivalent kernel. Do not follow that legacy skill's old
`tmp/` staging or phase/manifest workflow.

See `flashinfer_bench/onboarding/README.md` and `user_guide.md` for the file and failure
contracts.
