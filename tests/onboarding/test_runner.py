from .conftest import *


def test_remote_diagnostic_full_scan_disables_early_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import flashinfer_bench.onboarding.runners.modal_runner as modal_runner

    def fake_prepare_fitrace_dump(output_dir: Path) -> Path:
        path = output_dir / "definitions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fake_prepare_worker_injection(**kwargs) -> None:
        return None

    def fake_stage(**kwargs):
        output_dir = kwargs["output_dir"]
        collect_dir = output_dir / "collect"
        definitions_dir = output_dir / "audited_definitions"
        hints_dir = output_dir / "definition_hints"
        collect_dir.mkdir(parents=True, exist_ok=True)
        definitions_dir.mkdir(parents=True, exist_ok=True)
        hints_dir.mkdir(parents=True, exist_ok=True)
        return {
            "collect_dir": collect_dir,
            "audited_definitions_dir": definitions_dir,
            "definition_hints_dir": hints_dir,
            "definition_audit_report": {"summary": {"raw": 1, "passed": 1, "rejected": 0}},
            "collect_plan": {"targets": [], "skipped": []},
            "workload_manifest": {
                "summary": {"workloads": 0, "skipped": 1, "captures": 0, "sanitized": 0},
                "skipped": [{"name": "bad", "reason": "sanitize_failed"}],
            },
        }

    monkeypatch.setattr(modal_runner, "_prepare_fitrace_dump", fake_prepare_fitrace_dump)
    monkeypatch.setattr(modal_runner, "prepare_worker_injection", fake_prepare_worker_injection)
    monkeypatch.setattr(modal_runner, "_build_remote_post_capture_outputs", fake_stage)

    seen_early_checks = []
    modal_probe_plan = {
        "probe_plan": ProbePlan(targets=[], skipped=[], warmup_hooks=[]).to_jsonable(),
        "capture_limits": {"max_captures_per_target": 1},
        "diagnostics": {"full_scan": True},
    }

    result = run_remote_probe_entrypoint(
        modal_probe_plan=modal_probe_plan,
        output_dir=tmp_path / "remote",
        run_model=lambda early_check: seen_early_checks.append(early_check) or True,
    )

    assert seen_early_checks == [None]
    assert "early_stop" not in result
    assert result["summary"]["diagnostic_full_scan"] is True
    assert result["workload_manifest"]["summary"]["skipped"] == 1

def test_materialize_modal_result_extracts_remote_collect_and_redacts_archives(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    collect_dir = source_root / "collect"
    collect_dir.mkdir(parents=True)
    workload_path = collect_dir / "workloads" / "gqa_paged" / "demo.jsonl"
    workload_path.parent.mkdir(parents=True)
    workload_path.write_text("{}\n", encoding="utf-8")
    blob_path = collect_dir / "blob" / "workloads" / "gqa_paged" / "demo" / "x.safetensors"
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(b"blob")
    (collect_dir / "workload_manifest.json").write_text(
        json.dumps({
            "summary": {"workloads": 1},
            "workloads": [
                {
                    "name": "demo",
                    "definition_path": "/tmp/flashinfer-trace-probe/audited_definitions/gqa_paged/demo.json",
                    "capture_paths": ["/tmp/flashinfer-trace-probe/captures/x.pt"],
                    "workload_paths": [
                        "/tmp/flashinfer-trace-probe/collect/workloads/gqa_paged/demo.jsonl"
                    ],
                    "blob_paths": [
                        "/tmp/flashinfer-trace-probe/collect/blob/workloads/gqa_paged/demo/x.safetensors"
                    ],
                }
            ],
        }),
        encoding="utf-8",
    )
    (collect_dir / "collect_plan.json").write_text(
        json.dumps({
            "targets": [
                {
                    "name": "demo",
                    "definition_name": "demo",
                    "definition_path": "/tmp/flashinfer-trace-probe/audited_definitions/gqa_paged/demo.json",
                }
            ]
        }),
        encoding="utf-8",
    )
    archive_path = source_root / "collect.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(collect_dir, arcname="collect")
    definitions_dir = source_root / "definitions" / "gqa_paged"
    definitions_dir.mkdir(parents=True)
    (definitions_dir / "demo.json").write_text('{"name":"demo","op_type":"gqa_paged"}\n', encoding="utf-8")
    definitions_archive_path = source_root / "definitions.tar.gz"
    with tarfile.open(definitions_archive_path, "w:gz") as archive:
        archive.add(source_root / "definitions", arcname="definitions")
    output_dir = tmp_path / "run" / ".modal_tmp"
    materialize_modal_result(
        {
            "parse_report": {"summary": {"events": 1, "missing_targets": 0}},
            "collect_archive_b64": base64.b64encode(archive_path.read_bytes()).decode("ascii"),
            "definitions_archive_b64": base64.b64encode(definitions_archive_path.read_bytes()).decode("ascii"),
            "definition_audit_report": {"summary": {"raw": 1, "passed": 1, "repaired": 0, "rejected": 0}},
            "summary": {"workloads": 1, "sanitized": 1},
        },
        output_dir,
    )

    assert not (output_dir / "events.jsonl").exists()
    assert not (output_dir / "parse_report.json").exists()
    result = json.loads((output_dir / "modal_result.json").read_text(encoding="utf-8"))
    assert result["parse_report"] == {"summary": {"events": 1, "missing_targets": 0}}
    local_manifest = result["workload_manifest"]
    workload = local_manifest["workloads"][0]
    assert workload["definition_path"] == str(tmp_path / "run" / "output" / "definitions" / "gqa_paged" / "demo.json")
    assert workload["capture_paths"] == ["/tmp/flashinfer-trace-probe/captures/x.pt"]
    assert workload["workload_paths"] == [
        str(tmp_path / "run" / "output" / "workloads" / "gqa_paged" / "demo.jsonl")
    ]
    assert workload["blob_paths"] == [
        str(
            tmp_path
            / "run"
            / "output"
            / "blob"
            / "workloads"
            / "gqa_paged"
            / "demo"
            / "x.safetensors"
        )
    ]
    collect_plan = result["collect_plan"]
    assert collect_plan["targets"][0]["definition_path"] == str(
        tmp_path / "run" / "output" / "definitions" / "gqa_paged" / "demo.json"
    )
    assert (tmp_path / "run" / "output" / "definitions" / "gqa_paged" / "demo.json").exists()
    assert (tmp_path / "run" / "output" / "workloads" / "gqa_paged" / "demo.jsonl").exists()
    assert not (output_dir / "captures").exists()

    assert result["collect_archive_b64"]["redacted"] is True
    assert result["definitions_archive_b64"]["redacted"] is True
    assert not (tmp_path / "run" / "collect.tar.gz").exists()
    assert not (tmp_path / "run" / "definitions.tar.gz").exists()

def test_run_collect_run_requires_remote_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flashinfer_bench.onboarding import cli

    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", ["alpha"])
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "batch_sizes": [1],
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

    def fake_run_modal_probe(*, modal_probe_plan, output_dir, timeout, resume_call_id=None):
        return {
            "parse_report": {"summary": {"events": 0, "missing_targets": 1}},
            "summary": {"events_written": 0},
        }

    monkeypatch.setattr(cli, "run_modal_probe", fake_run_modal_probe)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--run", "demo"])

    assert "remote collect was enabled but no workload manifest was returned" in str(exc_info.value)
    assert "definition audit under the run directory" in str(exc_info.value)
    assert not (run_dir / "reports" / "run_report.json").exists()

def test_run_diagnostic_full_scan_sets_modal_plan_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flashinfer_bench.onboarding import cli

    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", ["alpha"])
    run_dir = tmp_path / "runs" / "demo"
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "batch_sizes": [1],
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

    def fake_run_modal_probe(*, modal_probe_plan, output_dir, timeout, resume_call_id=None):
        assert modal_probe_plan["diagnostics"] == {"full_scan": True}
        raise RuntimeError("stop after plan assertion")

    monkeypatch.setattr(cli, "run_modal_probe", fake_run_modal_probe)

    with pytest.raises(RuntimeError, match="stop after plan assertion"):
        cli.main(["run", "--run", "demo", "--diagnostic-full-scan"])
