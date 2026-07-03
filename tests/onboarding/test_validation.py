from .conftest import *


def test_validate_run_writes_reviewable_summary(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    _write_definition(definitions_dir, "approved_target")
    approved_targets = tmp_path / "approved_targets.json"
    approved_targets.write_text(
        json.dumps([
                {
                    "name": "approved_target",
                    "target": "flashinfer.norm.rmsnorm",
                    "module": "flashinfer.norm",
                    "attr": "rmsnorm",
                    "collect": True,
                    "capture": _capture_json(),
                    }
        ]),
        encoding="utf-8",
    )
    parse_report = tmp_path / "parse_report.json"
    parse_report.write_text(
        json.dumps({
            "summary": {"events": 1, "non_warmup_events": 1, "missing_targets": 0},
            "observed_targets": [{"name": "approved_target"}],
            "missing_targets": [],
        }),
        encoding="utf-8",
    )
    manifest = tmp_path / "workload_manifest.json"
    workload_path = tmp_path / "workloads" / "rmsnorm" / "approved_target.jsonl"
    blob_path = tmp_path / "blob" / "workloads" / "rmsnorm" / "approved_target" / "000001.pt"
    workload_path.parent.mkdir(parents=True)
    blob_path.parent.mkdir(parents=True)
    workload_path.write_text("{}\n", encoding="utf-8")
    blob_path.write_bytes(b"capture")
    manifest.write_text(
        json.dumps({
            "summary": {"workloads": 1, "captures": 1, "workload_files": 2, "sanitized": 1},
            "workloads": [
                {
                    "name": "approved_target",
                    "definition_name": "approved_target",
                    "capture_paths": [str(blob_path)],
                    "workload_paths": [str(workload_path)],
                    "blob_paths": [str(blob_path)],
                    "sanitized_count": 1,
                }
            ],
        }),
        encoding="utf-8",
    )

    report = validate_run(
        approved_targets_path=approved_targets,
        definitions_dir=definitions_dir,
        parse_report_path=parse_report,
        workload_manifest_path=manifest,
    )
    markdown = render_run_review_markdown({
        "collect": {"manifest": {"summary": report["workload_summary"]}},
        "internal_validation": report,
    })

    assert report["summary"]["ok"] is True
    assert "sanitized entries: 1" in markdown

def test_validate_run_accepts_fitrace_dump_name_over_preview(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    definition_path = _write_definition(
        definitions_dir,
        "top_k_top_p_sampling_v128256",
        op_type="sampling",
    )
    payload = json.loads(definition_path.read_text(encoding="utf-8"))
    payload["tags"] = ["fi_api:flashinfer.sampling.top_k_top_p_sampling_from_probs", "status:verified"]
    definition_path.write_text(json.dumps(payload), encoding="utf-8")
    approved_targets = tmp_path / "approved_targets.json"
    approved_targets.write_text(
        json.dumps([
            {
                "name": "sampling",
                "target": "flashinfer.sampling.top_k_top_p_sampling_from_probs",
                "module": "flashinfer.sampling",
                "attr": "top_k_top_p_sampling_from_probs",
                "definition_source": "fitrace",
                "backend": "flashinfer",
                "collect": True,
                "definition_name": "top_k_top_p_sampling_from_probs_v128256",
                "op_type": "sampling",
                "capture": _capture_json(full_args=[0, 2]),
            }
        ]),
        encoding="utf-8",
    )

    report = validate_run(
        approved_targets_path=approved_targets,
        definitions_dir=definitions_dir,
    )

    assert report["summary"]["ok"] is True
    assert report["findings"] == []

def test_run_review_markdown_lists_repaired_definitions() -> None:
    markdown = render_run_review_markdown(
        {
            "summary": {"accepted": True, "internal_ok": True, "official_ok": True, "export_ok": True},
            "collect": {"manifest": {"summary": {"workloads": 1, "captures": 2, "workload_files": 3, "sanitized": 2}}},
            "definition_audit": {
                "summary": {"raw": 2, "passed": 1, "repaired": 1, "rejected": 0},
                "repaired": [
                    {
                        "source_name": "gqa_paged_decode_h32_kv128_d128_ps8",
                        "name": "gqa_paged_decode_h32_kv8_d128_ps1",
                        "repair": "gqa_paged_3d_cache_axes",
                        "reason": "gqa_paged num_kv_heads larger than num_qo_heads: 128>32",
                    }
                ],
            },
            "export": {"summary": {"definitions": 1, "missing_definitions": 0}},
            "internal_validation": {"summary": {"errors": 0, "warnings": 0}, "findings": []},
            "official_validation": {"ok": True, "returncode": 0},
        }
    )

    assert "### Repaired Definitions" in markdown
    assert "gqa_paged_decode_h32_kv128_d128_ps8 -> gqa_paged_decode_h32_kv8_d128_ps1" in markdown
    assert "gqa_paged_3d_cache_axes" in markdown

def test_export_run_dataset_copies_official_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    definition = tmp_path / "defs" / "rmsnorm" / "demo.json"
    definition.parent.mkdir(parents=True)
    definition.write_text(json.dumps({"name": "demo", "op_type": "rmsnorm"}), encoding="utf-8")
    (run_dir / "output" / "workloads" / "rmsnorm").mkdir(parents=True)
    (run_dir / "output" / "workloads" / "rmsnorm" / "demo.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "output" / "blob" / "workloads" / "rmsnorm" / "demo").mkdir(parents=True)
    (run_dir / "output" / "blob" / "workloads" / "rmsnorm" / "demo" / "x.safetensors").write_bytes(b"x")
    _write_collect_report(run_dir, plan={"targets": [{"definition_name": "demo", "op_type": "rmsnorm", "definition_path": str(definition)}]})

    report = export_run_dataset(run_dir=run_dir)
    dataset_dir = Path(report["dataset_dir"])

    assert report["summary"]["ok"] is True
    assert dataset_dir == run_dir / "output"
    assert (dataset_dir / "definitions" / "rmsnorm" / "demo.json").exists()
    assert (dataset_dir / "workloads" / "rmsnorm" / "demo.jsonl").exists()
    assert (dataset_dir / "blob" / "workloads" / "rmsnorm" / "demo" / "x.safetensors").exists()

def test_validate_cli_runs_internal_and_official_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fake_bench = tmp_path / "flashinfer_bench" / "cli"
    fake_bench.mkdir(parents=True)
    (tmp_path / "flashinfer_bench" / "__init__.py").write_text("", encoding="utf-8")
    (fake_bench / "__init__.py").write_text("", encoding="utf-8")
    (fake_bench / "main.py").write_text(
        "def cli():\n"
        "    print('flashinfer-bench fake validator ok')\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "demo"
    definitions_dir = run_dir / "output" / "definitions"
    definition = _write_definition(definitions_dir, "demo")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(
        run_dir,
        config={},
        approved=[
                {
                    "name": "demo",
                    "target": "flashinfer.norm.rmsnorm",
                    "module": "flashinfer.norm",
                    "attr": "rmsnorm",
                    "definition_name": "demo",
                    "collect": True,
                    }
        ],
    )
    update_run_report(
        run_dir,
        parse_report={
            "summary": {"events": 1, "non_warmup_events": 1, "missing_targets": 0},
            "observed_targets": [{"name": "demo"}],
            "missing_targets": [],
        },
    )
    workload_path = run_dir / "output" / "workloads" / "rmsnorm" / "demo.jsonl"
    blob_path = run_dir / "output" / "blob" / "workloads" / "rmsnorm" / "demo" / "000001.safetensors"
    workload_path.parent.mkdir(parents=True)
    blob_path.parent.mkdir(parents=True)
    workload_path.write_text("{}\n", encoding="utf-8")
    blob_path.write_bytes(b"blob")
    _write_collect_report(
        run_dir,
        plan={"targets": [{"definition_name": "demo", "op_type": "rmsnorm", "definition_path": str(definition)}]},
        manifest={
            "summary": {"workloads": 1, "captures": 1, "workload_files": 2, "sanitized": 1},
            "workloads": [
                {
                    "name": "demo",
                    "definition_name": "demo",
                    "capture_paths": [str(blob_path)],
                    "workload_paths": [str(workload_path)],
                    "blob_paths": [str(blob_path)],
                    "sanitized_count": 1,
                }
            ],
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashinfer_bench.onboarding.cli",
            "validate",
            "--run",
            "demo",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[1])]),
        },
    )

    assert "internal validation ok: True" in result.stdout
    assert "official validation ok: True" in result.stdout
    assert "run accepted: True" in result.stdout
    assert "run report: runs/demo/reports/run_report.json" in result.stdout
    run_report = json.loads((run_dir / "reports" / "run_report.json").read_text(encoding="utf-8"))
    assert run_report["summary"]["accepted"] is True
    assert run_report["collect"]["manifest"]["summary"]["workloads"] == 1
    assert run_report["internal_validation"]["summary"]["ok"] is True
    assert run_report["official_validation"]["ok"] is True
    assert run_report["official_validation"]["returncode"] == 0
    assert (run_dir / "reports" / "review.md").exists()
    assert not (run_dir / "validate" / "validation_report.json").exists()
    assert (run_dir / "output" / "definitions" / "rmsnorm" / "demo.json").exists()
    assert not (run_dir / "official_validate" / "official_validation_report.json").exists()

