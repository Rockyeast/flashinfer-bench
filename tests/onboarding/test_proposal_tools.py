from .conftest import *


def test_check_proposal_rejects_agent_guessed_collect_definition(tmp_path: Path) -> None:
    targets = tmp_path / "candidate_targets.json"
    targets.write_text(
        json.dumps([
            {
                "name": "bad_fi",
                "target": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                "module": "flashinfer.decode",
                "attr": "BatchDecodeWithPagedKVCacheWrapper.run",
                "backend": "flashinfer",
                "collect": True,
                "definition_source": "agent",
            },
            {
                "name": "manual_non_fi",
                "status": "candidate",
                "target": "sglang.srt.layers.attention.RadixAttention.forward",
                "backend": "sglang_kernel",
                "collect": False,
            },
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    report = check_proposal(proposal_dir=tmp_path, hf_config_path=hf_config)

    assert report["summary"]["ok"] is False
    assert any(
        item["reason"] == "collectable FlashInfer target must not use agent-guessed final definition"
        for item in report["findings"]
    )

def test_check_proposal_allows_reviewed_non_fitrace_collect_candidate(tmp_path: Path) -> None:
    targets = tmp_path / "candidate_targets.json"
    targets.write_text(
        json.dumps([
            {
                "name": "rotary_collect",
                "status": "candidate",
                "target": "sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward",
                "module": "sglang.srt.layers.rotary_embedding",
                "attr": "RotaryEmbedding.forward",
                "backend": "torch",
                "collect": True,
                "definition_source": "agent",
                "definition_name": "torch_rotary_embedding",
                "op_type": "rotary_embedding",
                "capture": _capture_json(full_args=[1, 2]),
                "evidence": [{"kind": "source_location", "value": "sglang/srt/layers/rotary_embedding.py"}],
            }
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")
    definition_dir = tmp_path / "definitions" / "rotary_embedding"
    definition_dir.mkdir(parents=True)
    (definition_dir / "torch_rotary_embedding.json").write_text(
        json.dumps({
            "name": "torch_rotary_embedding",
            "op_type": "rotary_embedding",
            "axes": {"num_tokens": {"type": "var"}, "head_dim": {"type": "const", "value": 128}},
            "inputs": {"query": {"shape": ["num_tokens", "head_dim"], "dtype": "bfloat16"}},
            "outputs": {"output": {"shape": ["num_tokens", "head_dim"], "dtype": "bfloat16"}},
            "reference": "def run(query):\n    return query\n",
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
            "inputs": {"query": [{"source": "arg", "arg_index": 1}]},
        }),
        encoding="utf-8",
    )

    report = check_proposal(proposal_dir=tmp_path, hf_config_path=hf_config)

    assert report["summary"]["ok"] is True
    assert report["summary"]["fitrace_targets"] == 0
    assert report["findings"] == []

def test_check_proposal_rejects_op_type_mismatch_with_trace_template(tmp_path: Path) -> None:
    flashinfer_root = tmp_path / "flashinfer"
    templates_dir = flashinfer_root / "trace" / "templates"
    templates_dir.mkdir(parents=True)
    (flashinfer_root / "gdn_prefill.py").write_text(
        "\n".join([
            "from .trace.templates.gdn import gdn_prefill_trace",
            "def flashinfer_api(func=None, *, trace=None):",
            "    def deco(f): return f",
            "    return deco(func) if func is not None else deco",
            "@flashinfer_api(trace=gdn_prefill_trace)",
            "def chunk_gated_delta_rule():",
            "    pass",
        ]),
        encoding="utf-8",
    )
    (templates_dir / "gdn.py").write_text(
        "gdn_prefill_trace = TraceTemplate(op_type='gdn', name_prefix='gdn_prefill')\n",
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text(json.dumps({"hidden_size": 1024}), encoding="utf-8")
    candidates = tmp_path / "candidate_targets.json"
    base_candidate = {
        "name": "gdn_prefill",
        "role": "target",
        "target": "flashinfer.gdn_prefill.chunk_gated_delta_rule",
        "module": "flashinfer.gdn_prefill",
        "attr": "chunk_gated_delta_rule",
        "backend": "flashinfer",
        "variant": "prefill",
        "definition_source": "fitrace",
        "collect": True,
        "capture": _capture_json(),
    }

    candidates.write_text(json.dumps([dict(base_candidate, op_type="gdn_prefill")]), encoding="utf-8")
    failed = check_proposal(
        proposal_dir=tmp_path,
        hf_config_path=hf_config,
        flashinfer_root=flashinfer_root,
    )

    assert failed["summary"]["ok"] is False
    assert any("does not match FlashInfer trace template op_type 'gdn'" in item["reason"] for item in failed["findings"])

    candidates.write_text(json.dumps([dict(base_candidate, op_type="gdn")]), encoding="utf-8")
    passed = check_proposal(
        proposal_dir=tmp_path,
        hf_config_path=hf_config,
        flashinfer_root=flashinfer_root,
    )

    assert passed["summary"]["ok"] is True
    assert passed["fitrace_eval"]["targets"][0]["trace_op_type"] == "gdn"

def test_merge_proposals_unions_candidates_and_evidence(tmp_path: Path) -> None:
    proposal_a = tmp_path / "agent_a" / "proposal"
    proposal_b = tmp_path / "agent_b" / "proposal"
    proposal_a.mkdir(parents=True)
    proposal_b.mkdir(parents=True)
    base = {
        "name": "decode_a",
        "status": "candidate",
        "target": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        "module": "flashinfer.decode",
        "attr": "BatchDecodeWithPagedKVCacheWrapper.run",
        "backend": "flashinfer",
        "definition_source": "fitrace",
        "op_type": "gqa_paged",
        "variant": "decode",
        "collect": True,
        "companion_attrs": ["forward"],
        "capture": _capture_json(),
        "evidence": [{"kind": "source_location", "value": "a.py:1"}],
        "review_note": "agent a note",
    }
    same_from_b = dict(
        base,
        name="decode_b",
        evidence=[{"kind": "source_location", "value": "b.py:2"}],
        review_note="agent b note",
    )
    unique = {
        "name": "sampling",
        "status": "candidate",
        "target": "flashinfer.sampling.top_k_top_p_sampling_from_probs",
        "module": "flashinfer.sampling",
        "attr": "top_k_top_p_sampling_from_probs",
        "backend": "flashinfer",
        "definition_source": "fitrace",
        "op_type": "sampling",
        "variant": None,
        "collect": True,
        "capture": _capture_json(full_args=[0]),
    }
    (proposal_a / "candidate_targets.json").write_text(json.dumps([base]), encoding="utf-8")
    (proposal_b / "candidate_targets.json").write_text(json.dumps([same_from_b, unique]), encoding="utf-8")

    output = tmp_path / "merged" / "proposal"
    report = merge_proposals(proposal_dirs=[proposal_a, proposal_b], output_dir=output)

    assert report["summary"]["ok"] is True
    assert report["summary"]["input_candidates"] == 3
    assert report["summary"]["merged_candidates"] == 2
    merged = json.loads((output / "candidate_targets.json").read_text(encoding="utf-8"))
    decode = next(item for item in merged if item["op_type"] == "gqa_paged")
    assert decode["name"] == "decode_a"
    assert decode["evidence"] == [
        {"kind": "source_location", "value": "a.py:1"},
        {"kind": "source_location", "value": "b.py:2"},
    ]
    assert "agent a note" in decode["review_note"]
    assert "agent b note" in decode["review_note"]
    assert (output / "review_checklist.md").exists()

def test_merge_proposals_reports_candidate_conflicts(tmp_path: Path) -> None:
    proposal_a = tmp_path / "agent_a" / "proposal"
    proposal_b = tmp_path / "agent_b" / "proposal"
    proposal_a.mkdir(parents=True)
    proposal_b.mkdir(parents=True)
    candidate = {
        "name": "rmsnorm",
        "status": "candidate",
        "target": "sglang.srt.layers.layernorm.RMSNorm.forward",
        "module": "sglang.srt.layers.layernorm",
        "attr": "RMSNorm.forward",
        "backend": "sglang_kernel",
        "definition_source": "agent",
        "definition_name": "rmsnorm_h2560",
        "op_type": "rmsnorm",
        "variant": None,
        "collect": True,
        "capture": _capture_json(full_args=[1]),
    }
    conflicting = dict(candidate, collect=False)
    (proposal_a / "candidate_targets.json").write_text(json.dumps([candidate]), encoding="utf-8")
    (proposal_b / "candidate_targets.json").write_text(json.dumps([conflicting]), encoding="utf-8")

    output = tmp_path / "merged" / "proposal"
    report = merge_proposals(proposal_dirs=[proposal_a, proposal_b], output_dir=output)

    assert report["summary"]["ok"] is False
    assert report["summary"]["candidate_conflicts"] == 1
    assert report["summary"]["merged_candidates"] == 0
    assert report["conflicts"][0]["fields"] == ["collect"]
    merged = json.loads((output / "candidate_targets.json").read_text(encoding="utf-8"))
    assert merged == []
    checklist = (output / "review_checklist.md").read_text(encoding="utf-8")
    assert "conflicting candidate fields" in checklist

def test_repair_loop_passes_prompt_to_agent_stdin_and_rechecks(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "model" / "run"
    reports_dir = run_dir / "reports"
    proposal_dir = run_dir / "proposal"
    reports_dir.mkdir(parents=True)
    proposal_dir.mkdir()
    (proposal_dir / "candidate_targets.json").write_text("[]", encoding="utf-8")
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")
    (reports_dir / "run_report.json").write_text(
        json.dumps({
            "diagnostics": {
                "uncollected_definitions": [
                    {
                        "name": "extra_definition",
                        "reason": "audited definition was not selected by collect_plan",
                    }
                ]
            }
        }),
        encoding="utf-8",
    )
    agent_script = tmp_path / "fake_agent.py"
    agent_script.write_text(
        "\n".join([
            "import json",
            "import sys",
            "from pathlib import Path",
            "proposal_dir = Path(sys.argv[1])",
            "(proposal_dir / 'seen_prompt.md').write_text(sys.stdin.read(), encoding='utf-8')",
            "(proposal_dir / 'ignored_definitions.json').write_text(json.dumps([",
            "    {'name': 'extra_definition', 'reason': 'not part of reviewed workload surface'}",
            "]), encoding='utf-8')",
        ]),
        encoding="utf-8",
    )

    result = repair_loop(
        run=run_dir,
        hf_config_path=hf_config,
        agent_command=[sys.executable, str(agent_script), str(proposal_dir)],
        max_rounds=2,
    )

    assert result["summary"]["agent_ran"] is True
    assert result["summary"]["rounds"] == 2
    assert result["summary"]["diagnostics_ok"] is True
    assert result["summary"]["ready_for_human_review"] is True
    assert result["agent_rounds"][0]["returncode"] == 0
    seen_prompt = (proposal_dir / "seen_prompt.md").read_text(encoding="utf-8")
    assert "FIX_REQUIRED" in seen_prompt
    assert "extra_definition" in seen_prompt
    assert Path(result["outputs"]["repair_prompt"]).exists()

def test_spawn_agents_generates_isolated_first_pass_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = spawn_agents(
        model_name="microsoft/Phi-4-mini-instruct",
        run_prefix=Path("phi_4_mini_instruct/first_pass"),
        hf_config_path=Path("agent_inputs/config/phi_4_mini_instruct.json"),
        sglang_root=Path("agent_inputs/sglang/python/sglang"),
        flashinfer_root=Path("agent_inputs/flashinfer/flashinfer"),
        cookbook_root=Path("agent_inputs/sgl-cookbook"),
        sglang_model_hints=["srt/models/phi.py"],
        count=3,
    )

    assert result["summary"]["count"] == 3
    assert result["summary"]["agents_started"] == 0
    run_dirs = [Path(item["run_dir"]) for item in result["prompts"]]
    assert run_dirs == [
        Path("runs/phi_4_mini_instruct/first_pass_agent_a"),
        Path("runs/phi_4_mini_instruct/first_pass_agent_b"),
        Path("runs/phi_4_mini_instruct/first_pass_agent_c"),
    ]
    for item in result["prompts"]:
        prompt = Path(item["prompt"])
        text = prompt.read_text(encoding="utf-8")
        assert "microsoft/Phi-4-mini-instruct" in text
        assert item["run_dir"] in text
        assert "Do not write `config/approved_targets.json`" in text

def test_spawn_agents_runs_external_agents_with_prompt_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent_script = tmp_path / "fake_first_pass_agent.py"
    agent_script.write_text(
        "\n".join([
            "import json",
            "import sys",
            "from pathlib import Path",
            "prompt = sys.stdin.read()",
            "print('agent stdout noise')",
            "print('agent stderr noise', file=sys.stderr)",
            "run_dir = None",
            "for line in prompt.splitlines():",
            "    if line.startswith('run dir: '):",
            "        run_dir = Path(line.split(': ', 1)[1])",
            "        break",
            "assert run_dir is not None",
            "proposal = run_dir / 'proposal'",
            "proposal.mkdir(parents=True, exist_ok=True)",
            "(proposal / 'seen_prompt.md').write_text(prompt, encoding='utf-8')",
            "(proposal / 'candidate_targets.json').write_text(json.dumps([]), encoding='utf-8')",
            "(proposal / 'architecture.md').write_text('# Architecture\\n', encoding='utf-8')",
            "(proposal / 'review_checklist.md').write_text('# Review\\n', encoding='utf-8')",
            "(run_dir / 'config').mkdir(parents=True, exist_ok=True)",
            "(run_dir / 'config' / 'run_config.json').write_text('{}', encoding='utf-8')",
        ]),
        encoding="utf-8",
    )

    result = spawn_agents(
        model_name="demo/model",
        run_prefix=Path("demo/parallel"),
        hf_config_path=Path("agent_inputs/config/demo.json"),
        sglang_root=Path("agent_inputs/sglang/python/sglang"),
        flashinfer_root=Path("agent_inputs/flashinfer/flashinfer"),
        cookbook_root=Path("agent_inputs/sgl-cookbook"),
        sglang_model_hints=[],
        count=2,
        agent_command=[sys.executable, str(agent_script)],
    )

    assert result["summary"]["agents_started"] == 2
    assert result["summary"]["agent_failures"] == 0
    for item in result["agent_results"]:
        proposal = Path(item["proposal_dir"])
        assert item["returncode"] == 0
        assert (proposal / "seen_prompt.md").exists()
