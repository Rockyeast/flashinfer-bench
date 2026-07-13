# Two-Stage Onboarding Guide

## Contract

The public workflow is limited to two commands:

| Command | Input | Stable output | Human decision |
|---|---|---|---|
| `dump-definition` | model + runtime config | `definitions/`, definition and SGLang evidence reports | edit/accept definitions |
| `dump-workload` | reviewed `definitions/` | `output/`, `run_report.json`, `review.md` | accept or revise definitions and rerun |

There is no candidate list, merge gate, promote command, proposal/config copy, or separate
definition transformation stage.

## Complete Example

```bash
python3 -B -m flashinfer_bench.onboarding.cli dump-definition \
  --run phi_4_mini/20260711 \
  --model-name microsoft/Phi-4-mini-instruct \
  --gpu L40S \
  --agent codex

# Review reports/definition_review.md and edit definitions/ directly.

python3 -B -m flashinfer_bench.onboarding.cli dump-workload \
  --run phi_4_mini/20260711 \
  --inferencex-profile 1k1k

# Review reports/review.md.
```

The first command may initialize `--image`, `--tp-size`, and `--timeout`. Resolved values
are persisted in `config/run_config.json`, so the second command only needs `--run`.

## Run Directory

```text
runs/<model>/<run_id>/
├── config/
│   └── run_config.json
├── definitions/                       # single reviewed source
│   └── <op_type>/<name>.json
├── output/                            # latest complete workload snapshot
│   ├── definitions/
│   ├── workloads/
│   └── blob/
├── reports/
│   ├── definition_report.json
│   ├── definition_review.md
│   ├── sglang_modules.json            # executed classes/signatures/source snippets
│   ├── sglang_logger_report.json      # output-only comparison with SGLang logger
│   ├── run_report.json
│   └── review.md
└── .modal_tmp/                        # interrupted-call handoff; removed after success
```

`output/` is replaced as one snapshot; workload runs never append to stale output.

## Definition Evidence and Review

`dump-definition` uses one prompt and one generated token. It gathers:

1. Native FlashInfer definition JSON from `FLASHINFER_TRACE_DUMP=1`.
2. Exact SGLang module classes that execute in the Modal parent or SGLang workers.
3. An optional bounded SGLang built-in tensor logger dump for output-surface comparison,
   enabled explicitly with `--compare-sglang-logger` on compatible models.

The SGLang tensor logger is not a workload backend. It records module outputs, while an
executable workload needs the exact inputs described by a Definition.

The deterministic review verifies formal `Definition` schema, path/name consistency,
reference output count, GQA invariants, and one of these mutually exclusive capture tags:

```text
fi_api:<exact decorated callable>
sglang_module:<exact observed torch.nn.Module class>
sglang_callable:<exact source-backed callable>
```

For SGLang capture, each definition input defaults to the same-named runtime argument.
Mappings are explicit tags:

```text
sglang_input:q=arg:query
sglang_input:scale=attr:scaling
```

With `--agent codex`, Agent reads the reports and existing native definitions, then writes
Definition JSON directly. It may edit only `definitions/`; it cannot approve evidence,
edit runtime/output/reports, or launch Modal.

## Workload Collection

Before spending GPU time, `dump-workload` validates definitions and records their SHA-256.
The remote stage reconstructs that exact snapshot and runs both capture backends in the
same SGLang inference pass:

- FI: `flashinfer_bench.tracing.flashinfer_logging` configures the native logger and
  `flashinfer_bench.tracing.sanitize` converts dumps.
- non-FI: a temporary worker bootstrap installs reviewed SGLang module/callable hooks and
  sends captured inputs to the existing `TracingRuntime`.

Each process writes a private non-FI shard. The parent merges shards, keeps at most one
workload per unique axes combination, caps each definition at `max_new_workloads`, and
writes the standard `definitions/workloads/blob` layout.

`--inferencex-profile 1k1k` replaces ShareGPT prompts in the workload stage with deterministic
random token IDs using InferenceX's fixed 1024-input/1024-output shape, 0.8 length range, and
`ignore_eos=true`. Repeat the option with `8k1k` to add 8192-input/1024-output coverage. The
requests run directly through the in-process SGLang Engine; this is shape coverage, not the
official InferenceX HTTP concurrency and throughput benchmark.

The run is accepted only when every collectable definition has a workload and the canonical
dataset validator passes. A definition with no supported capture tag remains visible in
the review report but is skipped.

If `dump-workload --agent codex` fails, Agent may analyze the report and rewrite
`definitions/`. A changed definition digest invalidates that output. Human review and a
fresh `dump-workload` are required.

## Resume and Failure Rules

The Modal client prints `function_call_id` and an exact resume command before waiting.
After a local disconnect, rerun the same stage with `--resume-call-id <id>`.
The remote function mounts the persistent Modal Volume `flashinfer-bench-hf-cache` at
`/mnt/hf-cache` and points `HF_HOME`/`HF_HUB_CACHE` there, so later cold containers reuse
downloaded model weights. Override the Volume name with
`FLASHINFER_BENCH_MODAL_HF_CACHE_VOLUME`.

The remote result is accepted only if its definition digest matches the current local
snapshot. Common failures:

- `definitions failed review`: fix `reports/definition_review.md` findings;
- `no reviewed definition contains a supported capture tag`: add an evidenced FI or
  SGLang capture point;
- `missing definitions > 0`: a reviewed capture point did not execute or its inputs did
  not satisfy the definition;
- `dataset validation: fail`: inspect validator details in `run_report.json`.

## Internal Call Chain

```text
cli.py
  -> planning.py
  -> modal_client.py
  -> modal_app.py
  -> remote_pipeline.py
       -> sglang_runner.py
       -> tracing/flashinfer_logging.py -> tracing/sanitize.py
       -> tracing/sglang_logging.py -> tracing/TracingRuntime
  -> validation.py
```

`modal_app.py` is only the platform boundary. The two-stage runner chooses capture backends;
both backends converge on the standard FlashInfer-Bench dataset format.
