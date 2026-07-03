from .conftest import *


def test_modal_probe_collect_run_expands_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", [
        "Explain why GPU memory bandwidth matters for transformer inference.",
        "Summarize the tradeoff between latency and throughput in batch serving.",
        "Write a short Python function that checks whether a number is prime.",
        "Give three practical debugging steps for a CUDA kernel launch failure.",
        "Describe how paged KV cache helps long-context language model serving.",
        "Compare greedy decoding with top-p sampling in two concise paragraphs.",
        "Draft a polite email asking a teammate to review a pull request.",
        "List the main components of an attention layer and their tensor shapes.",
    ])
    plan = build_modal_probe_plan(
        probe_plan=ProbePlan(targets=[], skipped=[]),
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir=tmp_path,
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        batch_sizes=[1, 2, 4, 8, 16, 32, 64],
        max_new_tokens=96,
        supplemental_runs=[_supplemental_run()],
        max_captures_per_target=128,
    )

    assert plan["sampling"]["max_new_tokens"] == 96
    assert plan["capture_limits"]["max_captures_per_target"] == 128
    assert "batch_sizes" not in plan
    assert "capture_output" not in plan
    assert [len(scenario["prompts"]) for scenario in _prompt_scenarios(plan)] == [1, 2, 4, 8, 16, 32, 64]
    assert [scenario["max_new_tokens"] for scenario in _prompt_scenarios(plan)] == [96, 64, 32, 32, 16, 8, 8]
    assert all("bucket" not in scenario for scenario in _prompt_scenarios(plan))
    assert plan["sglang"]["disable_cuda_graph"] is True

def test_modal_plan_enables_remote_fitrace_collect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", ["alpha", "beta"])
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "batch_sizes": [1, 2],
            "max_new_tokens": 16,
            "supplemental_runs": [_supplemental_run()],
            "max_captures_per_target": 8,
        },
        approved=[
                {
                    "name": "approved_target",
                    "target": "flashinfer.norm.rmsnorm",
                    "module": "flashinfer.norm",
                    "attr": "rmsnorm",
                    "backend": "flashinfer",
                    "collect": True,
                }
        ],
    )

    monkeypatch.chdir(tmp_path)
    modal_plan = _build_modal_plan_from_run(
        run_dir,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        batch_sizes=[1, 2],
        max_new_tokens=16,
        supplemental_runs=[_supplemental_run()],
        max_captures_per_target=8,
    )

    assert modal_plan["capture_limits"]["max_captures_per_target"] == 8
    assert "remote_collect" not in modal_plan
    assert "definitions_dir" not in modal_plan

def test_collect_plan_from_probe_plan_uses_fitrace_definitions(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    definition_path = _write_definition(
        definitions_dir,
        "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
        op_type="gqa_paged",
    )
    definitions = {
        "gqa_paged_prefill_causal_h32_kv8_d128_ps1": DefinitionRef(
            name="gqa_paged_prefill_causal_h32_kv8_d128_ps1",
            op_type="gqa_paged",
            path=definition_path,
        )
    }
    plan = ProbePlan(
        targets=[
            ProbeTarget(
                name="llama31_prefill",
                target="flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                module="flashinfer.prefill",
                attr="BatchPrefillWithPagedKVCacheWrapper.run",
                definition_name="gqa_paged_prefill_causal_h32_kv8_d128_ps1",
                op_type="gqa_paged",
                backend="flashinfer",
                collect=True,
                capture=_capture_spec(),
            )
        ],
        skipped=[],
    )

    events = [
        {
            "name": "llama31_prefill",
            "definition_name": "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
            "is_warmup": False,
        }
    ]

    collect_plan = build_collect_plan_from_probe_plan(definitions=definitions, probe_plan=plan, events=events)

    assert collect_plan.skipped == []
    assert len(collect_plan.targets) == 1
    assert collect_plan.targets[0].name == "llama31_prefill"
    assert collect_plan.targets[0].definition_name == "gqa_paged_prefill_causal_h32_kv8_d128_ps1"
    assert collect_plan.targets[0].definition_path == definition_path

def test_collect_plan_accepts_reviewed_non_fitrace_definition(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    definition_path = _write_definition(
        definitions_dir,
        "torch_rotary_embedding",
        op_type="rotary_embedding",
    )
    definitions = {
        "torch_rotary_embedding": DefinitionRef(
            name="torch_rotary_embedding",
            op_type="rotary_embedding",
            path=definition_path,
        )
    }
    plan = ProbePlan(
        targets=[
            ProbeTarget(
                name="rotary_target",
                target="sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward",
                module="sglang.srt.layers.rotary_embedding",
                attr="RotaryEmbedding.forward",
                definition_name="torch_rotary_embedding",
                op_type="rotary_embedding",
                backend="torch",
                collect=True,
                capture=_capture_spec(),
            )
        ],
        skipped=[],
    )
    events = [
        {
            "name": "rotary_target",
            "definition_name": "torch_rotary_embedding",
            "is_warmup": False,
        }
    ]

    collect_plan = build_collect_plan_from_probe_plan(
        definitions=definitions,
        probe_plan=plan,
        events=events,
    )

    assert collect_plan.skipped == []
    assert len(collect_plan.targets) == 1
    assert collect_plan.targets[0].backend == "torch"
    assert collect_plan.targets[0].definition_name == "torch_rotary_embedding"
    assert collect_plan.targets[0].definition_path == definition_path

def test_build_probe_plan_routes_warmup_role_into_warmup_hooks() -> None:
    approved = [
        ApprovedTarget(
            name="decode",
            target="pkg.mod.kernel",
            module="pkg.mod",
            attr="kernel",
            backend="flashinfer",
            collect=True,
            op_type="gqa_paged",
            capture=_capture_spec(),
        ),
        ApprovedTarget(
            name="sglang_warmup",
            role="warmup",
            module="pkg.mod",
            attr="warmup",
        ),
    ]

    plan = build_probe_plan(approved)

    assert [t.name for t in plan.targets] == ["decode"]
    assert [h.name for h in plan.warmup_hooks] == ["sglang_warmup"]
    assert plan.warmup_hooks[0].module == "pkg.mod"
    assert plan.warmup_hooks[0].attr == "warmup"

    # Round-trips through the modal probe-plan payload.
    rebuilt = ProbePlan.from_jsonable(plan.to_jsonable())
    assert [h.attr for h in rebuilt.warmup_hooks] == ["warmup"]

def test_probe_plan_round_trips_companion_attrs() -> None:
    plan = ProbePlan(
        targets=[
            ProbeTarget(
                name="decode",
                target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                module="flashinfer.decode",
                attr="BatchDecodeWithPagedKVCacheWrapper.run",
                op_type="gqa_paged",
                companion_attrs=["forward", "begin_forward"],
                capture=_capture_spec(),
            )
        ],
        skipped=[],
    )

    rebuilt = ProbePlan.from_jsonable(plan.to_jsonable())

    assert rebuilt.targets[0].companion_attrs == ["forward", "begin_forward"]

