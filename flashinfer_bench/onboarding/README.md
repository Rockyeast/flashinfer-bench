# Model Onboarding

Model onboarding has two capture stages, two human review points, and one independent
submission gate:

```text
dump-definition -> review/edit definitions -> dump-workload -> review dataset
                                                       -> check-submission
```

`definitions/` is the single reviewed source. There is no proposal/config copy. Workload
export only removes capture-specific metadata from the publication copy under
`output/definitions/`; it does not create another reviewed definition source.

## 1. Dump and Analyze Definitions

```bash
flashinfer-bench onboarding dump-definition \
  --run qwen3/20260711 \
  --model-name Qwen/Qwen3-1.7B \
  --gpu L40S \
  --agent codex
```

The command performs one bounded SGLang pass and writes:

```text
runs/qwen3/20260711/
├── config/run_config.json
├── definitions/<op_type>/<definition_name>.json
└── reports/
    ├── definition_report.json
    ├── definition_review.md
    └── evidence/
        ├── sglang_execution_inventory.json
        └── sglang_logger.json
```

Three evidence sources are combined:

- FlashInfer native definition tracing writes `fi_api:` definitions.
- A worker bootstrap records the exact SGLang module classes, forward signatures, sample
  input shapes, and bounded source snippets that actually executed.
- SGLang's built-in generic dumper is enabled by default. In a bounded comparison pass it
  records representative-layer module inputs and outputs. Complete module paths are compared
  with tracing inventory first; tensor signatures are only a weak fallback. This remains
  comparison evidence and does not create workloads. Pass
  `--no-compare-sglang-logger` to disable it.

With `--agent codex`, the agent reads this evidence and writes Definition JSON directly.
It may preserve native FI definitions or add non-FI definitions; it does not create a
separate proposal or patch artifact.

Each collectable definition chooses exactly one capture backend:

```text
fi_api:<exact decorated FlashInfer callable>
sglang_module:<exact fully-qualified torch.nn.Module class>
sglang_callable:<exact fully-qualified plain callable>
```

For non-FI capture, definition input names default to runtime argument names. Explicit
mapping tags are available when they differ:

```text
sglang_input:<definition_input>=arg:<runtime_argument>
sglang_input:<definition_input>=attr:<module_attribute>
```

The deterministic review checks the formal Definition schema, path/name consistency,
reference `run(...)`, capture-tag exclusivity, observed SGLang module evidence, exact FI
callables, and GQA head/name invariants. Review and edit `definitions/` in place. The
definition stage refuses to overwrite existing reviewed files unless `--overwrite` is
explicit.

## 2. Dump Workloads

```bash
flashinfer-bench onboarding dump-workload \
  --run qwen3/20260711
```

Running this command approves the current definition snapshot. Before GPU submission it
validates every definition and records a SHA-256 digest. During the SGLang pass:

- `fi_api:` definitions use FlashInfer's native Level-10 logger and the packaged
  `flashinfer_bench.tracing.sanitize` converter;
- `sglang_module:` definitions reuse SGLang generic-dumper inputs plus a thin
  Definition/path/attribute binding adapter; `sglang_callable:` definitions keep the reviewed
  callable wrapper because the dumper only hooks modules. Both paths send inputs to the existing
  `TracingRuntime`;
- worker-local shards are deduplicated by definition axes and merged into the same
  FlashInfer-Bench dataset layout.

Outputs:

```text
runs/qwen3/20260711/
├── definitions/                 # reviewed input, unchanged
├── output/
│   ├── definitions/
│   ├── workloads/
│   ├── blob/
│   └── tests/references/        # added before submission for new definitions
└── reports/
    ├── evidence/capture_metadata.json
    ├── run_report.json
    └── review.md
```

The command fails when a collectable definition produces no workload or when the canonical
dataset validator fails. With `--agent codex`, the failure report is another source for
definition analysis. If the agent changes `definitions/`, the old output is stale; review
the new snapshot and rerun `dump-workload`.

`definitions/` keeps capture-only `sglang_*` metadata needed by the workload stage.
Published copies under `output/definitions/` exclude that metadata; the removed fields are
recorded in `reports/evidence/capture_metadata.json` instead.

## 3. Review Reference Tests

After successful workload validation, `dump-workload` automatically generates deterministic
tests for supported FlashInfer APIs. When `dump-workload` is run with `--agent codex`, it also
asks Agent to write source-backed non-FI tests. The pipeline then stops for human review.

The standalone command is retained for rerunning this step or selecting a custom tests directory:

```bash
flashinfer-bench onboarding prepare-reference-tests \
  --run qwen3/20260711 \
  --agent codex
```

Generated tests are written to `output/tests/references/`. FI tests compare the Definition's
PyTorch `reference.run()` with the exact `fi_api:` implementation. Non-FI tests must compare
against the exact SGLang module/callable or another independent source-backed implementation;
they still require human review.

## 4. Check Submission

After adding one reference test per new definition, run the independent publication gate:

```bash
flashinfer-bench onboarding check-submission \
  --run qwen3/20260711
```

The command always refreshes the remote `flashinfer-ai/flashinfer-trace` main branch before
checking; a network or refresh failure stops the gate instead of using stale data. This gate
rejects missing definition/axis/input/output descriptions, unsupported publication tags or
status values, name conflicts with the current upstream dataset, and missing
`output/tests/references/test_<definition_name>.py` files. Definitions already present upstream
are treated as reused dependencies rather than new submissions.

## Runtime Config

The first command writes `config/run_config.json`. Required fields are `model_name`,
`image`, `gpu`, and `tp_size`; CLI values override or initialize them. Common optional
fields are:

```json
{
  "batch_sizes": [1, 2, 4, 8],
  "max_new_tokens": 16,
  "max_new_workloads": 20,
  "compare_sglang_logger": true,
  "disable_cuda_graph": true,
  "force_flashinfer_backends": true,
  "mem_fraction_static": 0.7,
  "engine_kwargs": {},
  "isl": 1024,
  "osl": 8,
  "random_range_ratio": 1.0,
  "seed": 0
}
```

SGLang logger comparison is enabled by default and needs no separate command. The
definition pass enables SGLang's generic dumper through `DUMPER_*` environment variables,
summarizes its temporary input/output dumps into `reports/evidence/sglang_logger.json`,
compares complete module paths and tensor signatures, then deletes the raw dumps. Set
`compare_sglang_logger` to `false` or pass
`--no-compare-sglang-logger` to disable its additional logging overhead.

Both stages replay the same deterministic synthetic request matrix. It combines 128-token and
`isl` inputs across every configured batch size, one long request capped at 8192 tokens or one
quarter of the model context, and one 75% shared-prefix batch. `osl` is the generated-token
length. `random_range_ratio=1.0` uses exact lengths. The matrix is persisted by
`dump-definition`; `dump-workload` reuses it instead of accepting a second set of length flags.
This borrows controllable request shapes from the existing serving benchmark without running its
HTTP throughput/latency path.

Interrupted Modal calls print a call ID and resume command. Pass that ID through
`--resume-call-id`; completed `.modal_tmp/` handoff files are removed automatically.
Model downloads are persisted in the Modal Volume `flashinfer-bench-hf-cache`, mounted at
`/mnt/hf-cache` and selected through `HF_HOME`/`HF_HUB_CACHE`. Set
`FLASHINFER_BENCH_MODAL_HF_CACHE_VOLUME` to use another Volume name.

See [user_guide.md](user_guide.md) for file contracts and failure behavior.
