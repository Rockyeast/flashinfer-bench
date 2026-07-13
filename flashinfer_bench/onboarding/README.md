# Model Onboarding

Model onboarding has two public stages and two human review points:

```text
dump-definition -> review/edit definitions -> dump-workload -> review dataset
```

`definitions/` is the single reviewed source. There is no proposal/config copy and no
separate definition transformation layer.

## 1. Dump and Analyze Definitions

```bash
python3 -B -m flashinfer_bench.onboarding.cli dump-definition \
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
    ├── sglang_modules.json
    └── sglang_logger_report.json
```

Three evidence sources are combined:

- FlashInfer native definition tracing writes `fi_api:` definitions.
- A worker bootstrap records the exact SGLang module classes, forward signatures, sample
  input shapes, and bounded source snippets that actually executed.
- With `--compare-sglang-logger`, SGLang's built-in tensor logger records first-layer
  outputs as optional comparison evidence. It
  does not create workloads because it does not preserve complete operator inputs.

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
python3 -B -m flashinfer_bench.onboarding.cli dump-workload \
  --run qwen3/20260711
```

Running this command approves the current definition snapshot. Before GPU submission it
validates every definition and records a SHA-256 digest. During the SGLang pass:

- `fi_api:` definitions use FlashInfer's native Level-10 logger and the packaged
  `flashinfer_bench.tracing.sanitize` converter;
- `sglang_module:` and `sglang_callable:` definitions capture real inputs in SGLang
  workers and send them to the existing `TracingRuntime`;
- worker-local shards are deduplicated by definition axes and merged into the same
  FlashInfer-Bench dataset layout.

Outputs:

```text
runs/qwen3/20260711/
├── definitions/                 # reviewed input, unchanged
├── output/
│   ├── definitions/
│   ├── workloads/
│   └── blob/
└── reports/
    ├── run_report.json
    └── review.md
```

The command fails when a collectable definition produces no workload or when the canonical
dataset validator fails. With `--agent codex`, the failure report is another source for
definition analysis. If the agent changes `definitions/`, the old output is stale; review
the new snapshot and rerun `dump-workload`.

## Runtime Config

The first command writes `config/run_config.json`. Required fields are `model_name`,
`image`, `gpu`, and `tp_size`; CLI values override or initialize them. Common optional
fields are:

```json
{
  "batch_sizes": [1, 2, 4, 8],
  "max_new_tokens": 64,
  "max_new_workloads": 20,
  "compare_sglang_logger": true,
  "disable_cuda_graph": true,
  "force_flashinfer_backends": true,
  "mem_fraction_static": 0.7,
  "engine_kwargs": {},
  "inferencex_profiles": ["1k1k"],
  "inferencex_range_ratio": 0.8,
  "inferencex_seed": 0
}
```

SGLang logger comparison is enabled by default so each definition pass records independent
operator/output coverage evidence. Set `compare_sglang_logger` to `false` to disable its
additional logging overhead.

`inferencex_profiles` switches the workload stage from ShareGPT text to InferenceX-compatible
random token requests. Supported fixed-sequence profiles are `1k1k` and `8k1k`; each profile
is run at every configured `batch_sizes` value with `ignore_eos=true`. This reuses InferenceX
request shapes for capture coverage. It does not run the InferenceX HTTP throughput benchmark
or claim official InferenceX performance results.

Interrupted Modal calls print a call ID and resume command. Pass that ID through
`--resume-call-id`; completed `.modal_tmp/` handoff files are removed automatically.
Model downloads are persisted in the Modal Volume `flashinfer-bench-hf-cache`, mounted at
`/mnt/hf-cache` and selected through `HF_HOME`/`HF_HUB_CACHE`. Set
`FLASHINFER_BENCH_MODAL_HF_CACHE_VOLUME` to use another Volume name.

See [user_guide.md](user_guide.md) for file contracts and failure behavior.
