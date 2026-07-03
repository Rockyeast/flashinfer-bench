from .conftest import *


def test_definition_audit_repairs_gqa_paged_3d_cache_shape(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_definitions"
    raw_path = _write_definition(
        raw_dir,
        "gqa_paged_prefill_h32_kv128_d128_ps8",
        op_type="gqa_paged",
    )
    raw_definition = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_definition.update(
        {
            "tags": [
                "fi_api:flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                "stage:prefill",
                "status:verified",
            ],
            "axes": {
                "num_qo_heads": {"type": "const", "value": 32},
                "num_kv_heads": {"type": "const", "value": 128},
                "head_dim": {"type": "const", "value": 128},
                "page_size": {"type": "const", "value": 8},
            },
            "inputs": {
                "q": {"shape": ["total_q", "num_qo_heads", "head_dim"], "dtype": "bfloat16"},
                "k_cache": {"shape": ["num_pages", "page_size", "num_kv_heads", "head_dim"], "dtype": "bfloat16"},
                "v_cache": {"shape": ["num_pages", "page_size", "num_kv_heads", "head_dim"], "dtype": "bfloat16"},
            },
        }
    )
    raw_path.write_text(json.dumps(raw_definition), encoding="utf-8")
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "name": "llama31_8b_gqa_paged_prefill_ps1",
                "target": "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                "variant": "prefill",
                "page_size": 1,
                "args": [
                    {"type": "BatchPrefillWithPagedKVCacheWrapper"},
                    {"type": "Tensor", "shape": [1755, 32, 128], "dtype": "torch.bfloat16"},
                    {
                        "type": "tuple",
                        "elements": [
                            {"type": "Tensor", "shape": [182823, 8, 128], "dtype": "torch.bfloat16"},
                            {"type": "Tensor", "shape": [182823, 8, 128], "dtype": "torch.bfloat16"},
                        ],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_and_repair_definitions(
        raw_definitions_dir=raw_dir,
        output_definitions_dir=tmp_path / "definitions",
        output_hints_dir=tmp_path / "definition_hints",
        events_path=events_path,
        report_dir=tmp_path / "definition_audit",
    )

    repaired_path = tmp_path / "definitions" / "gqa_paged" / "gqa_paged_prefill_causal_h32_kv8_d128_ps1.json"
    repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
    assert report["summary"]["repaired"] == 1
    assert report["summary"]["rejected"] == 0
    assert report["aliases"] == {
        "gqa_paged_prefill_h32_kv128_d128_ps8": "gqa_paged_prefill_causal_h32_kv8_d128_ps1"
    }
    assert repaired["axes"]["num_kv_heads"]["value"] == 8
    assert repaired["axes"]["page_size"]["value"] == 1
    assert "status:repaired" in repaired["tags"]
    hints_path = tmp_path / "definition_hints" / "gqa_paged" / "gqa_paged_prefill_causal_h32_kv8_d128_ps1.json"
    hints = json.loads(hints_path.read_text(encoding="utf-8"))
    assert hints["inputs"]["k_cache"][0]["source"] == "arg_tuple"
    assert hints["axes"]["num_pages"]["source"] == "tensor_max_plus_one"
    assert not (tmp_path / "definition_audit" / "definition_audit_report.json").exists()

def test_definition_audit_writes_gdn_hints(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "gdn"
    raw_dir.mkdir(parents=True)
    raw_definition = raw_dir / "gdn_prefill_qk16_v16_d128.json"
    raw_definition.write_text(
        json.dumps({
            "name": "gdn_prefill_qk16_v16_d128",
            "op_type": "gdn",
            "axes": {
                "total_seq_len": {"type": "var"},
                "num_seqs": {"type": "var"},
                "num_q_heads": {"type": "const", "value": 16},
                "num_v_heads": {"type": "const", "value": 16},
                "head_size": {"type": "const", "value": 128},
                "len_cu_seqlens": {"type": "var"},
            },
                "inputs": {
                    "q": {"shape": ["total_seq_len", "num_q_heads", "head_size"], "dtype": "bfloat16"},
                "state": {
                    "shape": ["num_seqs", "num_v_heads", "head_size", "head_size"],
                    "dtype": "float32",
                    "optional": True,
                },
                "A_log": {"shape": ["num_v_heads"], "dtype": "unknown", "optional": True},
                "a": {"shape": ["total_seq_len", "num_v_heads"], "dtype": "float32"},
                "dt_bias": {"shape": ["num_v_heads"], "dtype": "unknown", "optional": True},
                "b": {"shape": ["total_seq_len", "num_v_heads"], "dtype": "float32"},
                    "cu_seqlens": {"shape": ["len_cu_seqlens"], "dtype": "int64"},
                    "scale": {"shape": None, "dtype": "float32", "optional": True},
                },
                "outputs": {
                    "output": {"shape": ["total_seq_len", "num_v_heads", "head_size"], "dtype": "bfloat16"}
                },
                "reference": "def run(q, state, a, b, cu_seqlens, scale=None):\n    return q\n",
            }),
            encoding="utf-8",
        )
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    report = audit_and_repair_definitions(
        raw_definitions_dir=tmp_path / "raw",
        output_definitions_dir=tmp_path / "definitions",
        output_hints_dir=tmp_path / "definition_hints",
        events_path=events_path,
        report_dir=tmp_path / "reports",
    )

    assert report["summary"]["hints"] == 1
    assert report["summary"]["repaired"] == 1
    output_definition = json.loads(
        (tmp_path / "definitions" / "gdn" / "gdn_prefill_qk16_v16_d128.json").read_text(encoding="utf-8")
    )
    assert "A_log" not in output_definition["inputs"]
    assert "dt_bias" not in output_definition["inputs"]
    assert "scale" not in output_definition["inputs"]
    hints_path = tmp_path / "definition_hints" / "gdn" / "gdn_prefill_qk16_v16_d128.json"
    hints = json.loads(hints_path.read_text(encoding="utf-8"))
    assert hints["inputs"]["state"] == [{"source": "kwarg", "name": "initial_state"}]
    assert hints["inputs"]["a"] == [{"source": "kwarg", "name": "g"}]
    assert hints["inputs"]["b"] == [{"source": "kwarg", "name": "beta"}]
    assert "A_log" not in hints["inputs"]
    assert "dt_bias" not in hints["inputs"]
    assert "scale" not in hints["inputs"]
    assert hints["axes"]["num_seqs"] == {"source": "tensor_numel_minus_one", "input": "cu_seqlens"}

def test_sanitizer_randomizes_float_and_stores_structural_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    definition = {
        "name": "demo_decode",
        "op_type": "gqa_paged",
        "axes": {
            "batch_size": {"type": "var"},
            "num_qo_heads": {"type": "const", "value": 2},
            "head_dim": {"type": "const", "value": 4},
            "len_indptr": {"type": "var"},
        },
        "inputs": {
            "q": {"shape": ["batch_size", "num_qo_heads", "head_dim"], "dtype": "bfloat16"},
            "kv_indptr": {"shape": ["len_indptr"], "dtype": "int32"},
            "sm_scale": {"shape": None, "dtype": "float32"},
        },
    }
    captured = {
        "payload": {
            "args": [
                {
                    "saved": False,
                    "summary": {"type": "Wrapper"},
                    "attrs": {
                        "_paged_kv_indptr_buf": {
                            "saved": True,
                            "summary": {"type": "Tensor", "shape": [4], "dtype": "torch.int32"},
                            "value": torch.tensor([0, 1, 3, 5], dtype=torch.int32),
                        }
                    },
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [3, 2, 4], "dtype": "torch.bfloat16"},
                    "value": torch.zeros((3, 2, 4), dtype=torch.bfloat16),
                },
            ],
            "kwargs": {
                "sm_scale": {"saved": True, "summary": {"type": "float", "value": 0.5}, "value": 0.5}
            },
        }
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints=_gqa_paged_hints(),
        output_dir=tmp_path,
    )
    assert diagnostic == {}
    assert entry is not None

    assert entry["workload"]["axes"] == {"batch_size": 3, "len_indptr": 4}
    assert entry["workload"]["inputs"]["q"] == {"type": "random"}
    assert entry["workload"]["inputs"]["sm_scale"] == {"type": "scalar", "value": 0.5}
    kv_indptr = entry["workload"]["inputs"]["kv_indptr"]
    assert kv_indptr["type"] == "safetensors"
    assert Path(tmp_path / kv_indptr["path"][2:]).exists()

def test_sanitizer_preserves_real_inputs_from_definition_hints(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    definition = {
        "name": "top_k_top_p_sampling_v8",
        "op_type": "sampling",
        "axes": {
            "batch_size": {"type": "var"},
            "vocab_size": {"type": "const", "value": 8},
        },
        "inputs": {
            "probs": {"shape": ["batch_size", "vocab_size"], "dtype": "float32"},
            "top_k": {"shape": ["batch_size"], "dtype": "int32"},
            "top_p": {"shape": ["batch_size"], "dtype": "float32"},
        },
    }
    captured = {
        "payload": {
            "args": [
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [2, 8], "dtype": "torch.float32"},
                    "value": torch.ones((2, 8), dtype=torch.float32),
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [2], "dtype": "torch.int32"},
                    "value": torch.tensor([4, 4], dtype=torch.int32),
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [2], "dtype": "torch.float32"},
                    "value": torch.tensor([0.9, 0.95], dtype=torch.float32),
                },
            ],
            "kwargs": {},
        }
    }
    hints = {
        "inputs": {
            "probs": [{"source": "arg", "arg_index": 0}],
            "top_k": [{"source": "arg", "arg_index": 1}],
            "top_p": [{"source": "arg", "arg_index": 2}],
        },
        "real_inputs": ["probs", "top_k", "top_p"],
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints=hints,
        output_dir=tmp_path,
    )

    assert diagnostic == {}
    assert entry is not None
    for input_name in ("probs", "top_k", "top_p"):
        input_payload = entry["workload"]["inputs"][input_name]
        assert input_payload["type"] == "safetensors"
        assert Path(tmp_path / input_payload["path"][2:]).exists()

def test_non_fitrace_reviewed_definition_hints_build_workload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    capture_path = tmp_path / "captures" / "000001_nonfi.pt"
    capture_path.parent.mkdir(parents=True)
    torch.save(
        {
            "schema_version": 1,
            "name": "nonfi_rotary",
            "definition_name": "torch_rotary_embedding",
            "target": "sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward",
            "op_type": "rotary_embedding",
            "is_warmup": False,
            "payload": {
                "args": [
                    {
                        "saved": True,
                        "summary": {"type": "Tensor", "shape": [2, 4], "dtype": "torch.float32"},
                        "value": torch.arange(8, dtype=torch.float32).reshape(2, 4),
                    }
                ],
                "kwargs": {},
            },
        },
        capture_path,
    )
    events = [
        {
            "name": "nonfi_rotary",
            "definition_name": "torch_rotary_embedding",
            "is_warmup": False,
            "capture_path": str(capture_path),
        }
    ]
    definitions_dir = tmp_path / "definitions" / "rotary_embedding"
    definitions_dir.mkdir(parents=True)
    definition_path = definitions_dir / "torch_rotary_embedding.json"
    definition_path.write_text(
        json.dumps({
            "name": "torch_rotary_embedding",
            "op_type": "rotary_embedding",
            "axes": {
                "batch_size": {"type": "var"},
                "hidden_size": {"type": "const", "value": 4},
            },
            "inputs": {
                "x": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
            },
        }),
        encoding="utf-8",
    )
    hints_dir = tmp_path / "definition_hints" / "rotary_embedding"
    hints_dir.mkdir(parents=True)
    (hints_dir / "torch_rotary_embedding.json").write_text(
        json.dumps({
            "schema_version": 1,
            "definition_name": "torch_rotary_embedding",
            "op_type": "rotary_embedding",
            "inputs": {"x": [{"source": "arg", "arg_index": 0}]},
            "real_inputs": ["x"],
        }),
        encoding="utf-8",
    )
    collect_plan = CollectPlan(
        targets=[
            CollectTarget(
                name="nonfi_rotary",
                definition_name="torch_rotary_embedding",
                op_type="rotary_embedding",
                target="sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward",
                backend="torch",
                collect=True,
                definition_path=definition_path,
            )
        ],
        skipped=[],
    )

    manifest = build_workload_manifest(
        events,
        collect_plan,
        output_dir=tmp_path / "collect",
        hints_dir=tmp_path / "definition_hints",
    )

    assert manifest["summary"]["workloads"] == 1
    assert manifest["summary"]["sanitized"] == 1
    assert manifest["skipped"] == []
    workload_path = tmp_path / "collect" / "workloads" / "rotary_embedding" / "torch_rotary_embedding.jsonl"
    assert workload_path.exists()
    entry = json.loads(workload_path.read_text(encoding="utf-8").strip())
    assert entry["workload"]["axes"] == {"batch_size": 2}
    input_spec = entry["workload"]["inputs"]["x"]
    assert input_spec["type"] == "safetensors"
    assert (tmp_path / "collect" / input_spec["path"][2:]).exists()

def test_workload_manifest_matches_repaired_definition_alias(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.pt"
    capture_path.write_bytes(b"not a torch payload")
    definition_path = tmp_path / "definitions" / "gqa_paged_decode_h32_kv8_d128_ps1.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(
        json.dumps({
            "name": "gqa_paged_decode_h32_kv8_d128_ps1",
            "op_type": "gqa_paged",
            "inputs": {},
        }),
        encoding="utf-8",
    )

    manifest = build_workload_manifest(
        [
            {
                "name": "decode",
                "definition_name": "gqa_paged_decode_h32_kv128_d128_ps8",
                "is_warmup": False,
                "capture_path": str(capture_path),
            }
        ],
        CollectPlan(
            targets=[
                CollectTarget(
                    name="decode",
                    definition_name="gqa_paged_decode_h32_kv8_d128_ps1",
                    op_type="gqa_paged",
                    target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                    backend="flashinfer",
                    collect=True,
                    definition_path=definition_path,
                    page_size=1,
                )
            ],
            skipped=[],
        ),
        output_dir=tmp_path / "collect",
        definition_aliases={
            "gqa_paged_decode_h32_kv128_d128_ps8": "gqa_paged_decode_h32_kv8_d128_ps1",
        },
    )

    assert manifest["summary"]["workloads"] == 0
    assert manifest["skipped"][0]["name"] == "decode"
    assert manifest["skipped"][0]["reason"] == "sanitize_failed"

