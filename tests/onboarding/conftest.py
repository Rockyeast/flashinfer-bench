import json
import os
import subprocess
import sys
import base64
import tarfile
import types
from pathlib import Path

import pytest

from flashinfer_bench.onboarding.core.capture import CaptureSession
from flashinfer_bench.onboarding.core.capture import infer_dispatch_value
from flashinfer_bench.onboarding.core.capture import HF_CONFIG_OVERRIDE_ENV, install_hf_config_override, install_hf_config_override_from_env
from flashinfer_bench.onboarding.core.definition_audit import audit_and_repair_definitions, prepare_definition_for_output
from flashinfer_bench.onboarding.core.events import (
    build_event_report,
    build_workload_manifest,
    build_sanitized_workload_entry,
    load_jsonl,
)
from flashinfer_bench.onboarding.runners.remote_runner import (
    _filter_supported_engine_kwargs,
    _materialize_reviewed_artifacts,
    _probe_passes,
    _prompt_scenarios,
    _supplemental_runs,
    run_remote_probe_entrypoint,
)
from flashinfer_bench.onboarding.runners.modal_client import materialize_modal_result
from flashinfer_bench.onboarding.cli import _reviewed_definition_artifacts
from flashinfer_bench.onboarding.core.collect_planning import (
    build_collect_plan_from_probe_plan,
    load_definitions,
)
from flashinfer_bench.onboarding.core.probe_planning import (
    build_modal_probe_plan,
    build_probe_plan,
    load_approved_targets,
)
from flashinfer_bench.onboarding.validation import (
    export_run_dataset,
    render_run_review_markdown,
    run_dataset_validator,
    update_run_report,
    validate_run,
)
from flashinfer_bench.onboarding.proposal.checks.fitrace import _required_sglang_engine_kwargs
from flashinfer_bench.onboarding.proposal.gate import check_proposal, run_proposal_gate
from flashinfer_bench.onboarding.proposal.workflow.diagnose import diagnose_run
from flashinfer_bench.onboarding.proposal.workflow.merge import merge_proposals
from flashinfer_bench.onboarding.proposal.workflow.prepare import slug_model_name
from flashinfer_bench.onboarding.proposal.workflow.repair import repair_loop
from flashinfer_bench.onboarding.proposal.workflow.spawn import spawn_agents
from flashinfer_bench.onboarding.core.schemas import ApprovedTarget, CaptureSpec, CollectPlan, CollectTarget, DefinitionRef, DispatchSpec, ProbePlan, ProbeTarget, WarmupHook

def _capture_json(
    *,
    full_args: list[int] | None = None,
    full_kwargs: list[str] | None = None,
    structural_attr_tokens: list[str] | None = None,
) -> dict:
    tokens = structural_attr_tokens or [
        "indptr",
        "indices",
        "last_page",
        "page_len",
        "seq_len",
        "offset",
        "mask",
        "block_table",
    ]
    return {
        "full_args": sorted(set(full_args or [])),
        "full_kwargs": sorted(set(full_kwargs or [])),
        "structural_attr_tokens": sorted(set(tokens)),
    }

def _capture_spec(
    *,
    full_args: list[int] | None = None,
    full_kwargs: list[str] | None = None,
    structural_attr_tokens: list[str] | None = None,
) -> CaptureSpec:
    return CaptureSpec(**_capture_json(
        full_args=full_args,
        full_kwargs=full_kwargs,
        structural_attr_tokens=structural_attr_tokens,
    ))

def _gqa_paged_hints(definition_name: str = "demo_decode") -> dict:
    return {
        "schema_version": 1,
        "definition_name": definition_name,
        "op_type": "gqa_paged",
        "inputs": {
            "q": [{"source": "arg", "arg_index": 1}],
            "k_cache": [{"source": "arg_tuple", "arg_index": 2, "tuple_index": 0}],
            "v_cache": [{"source": "arg_tuple", "arg_index": 2, "tuple_index": 1}],
            "kv_indptr": [{"source": "attr", "pattern": "paged_kv_indptr"}],
            "kv_indices": [{"source": "attr", "pattern": "paged_kv_indices"}],
            "qo_indptr": [{"source": "attr", "pattern": "qo_indptr"}],
        },
        "shape_overrides": {
            "k_cache": {"squeezed_axes": ["page_size"]},
            "v_cache": {"squeezed_axes": ["page_size"]},
        },
        "axes": {
            "num_kv_indices": {"source": "tensor_last", "input": "kv_indptr"},
            "total_q": {"source": "tensor_last", "input": "qo_indptr"},
            "num_pages": {
                "source": "tensor_max_plus_one",
                "input": "kv_indices",
                "limit_axis": "num_kv_indices",
            },
        },
        "tensor_slices": {
            "kv_indices": {"limit_axis": "num_kv_indices"},
        },
    }

def _gqa_ragged_hints(definition_name: str = "demo_ragged") -> dict:
    return {
        "schema_version": 1,
        "definition_name": definition_name,
        "op_type": "gqa_ragged",
        "inputs": {
            "q": [{"source": "arg", "arg_index": 1}],
            "k": [{"source": "arg", "arg_index": 2}],
            "v": [{"source": "arg", "arg_index": 3}],
        },
    }

def _page_size_dispatch_json() -> dict:
    return {
        "field": "page_size",
        "rules": [
            {"kind": "arg_attr", "arg_index": 0, "attrs": ["page_size", "_page_size"]},
            {"kind": "arg_shape", "arg_index": 2, "tuple_index": 0, "min_rank": 4, "shape_index": 1},
            {"kind": "arg_shape", "arg_index": 2, "tuple_index": 0, "rank": 3, "value": 1},
            {"kind": "arg_shape", "arg_index": 2, "min_rank": 5, "shape_index": 2},
            {"kind": "arg_shape", "arg_index": 2, "rank": 4, "equals_index": 1, "equals": 2, "value": 1},
        ],
    }

def _page_size_dispatch_spec() -> DispatchSpec:
    from flashinfer_bench.onboarding.core.schemas import dispatch_spec_from_jsonable

    dispatch = dispatch_spec_from_jsonable(_page_size_dispatch_json(), context="test target")
    assert dispatch is not None
    return dispatch

def _write_run_inputs(run_dir: Path, *, config: dict, approved: list[dict]) -> None:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    normalized = []
    for item in approved:
        copied = dict(item)
        if copied.get("role", "target") == "target" and "capture" not in copied:
            copied["capture"] = _capture_json()
        if copied.get("role", "target") == "target" and copied.get("page_size") is not None and "dispatch" not in copied:
            copied["dispatch"] = _page_size_dispatch_json()
        normalized.append(copied)
    (config_dir / "approved_targets.json").write_text(json.dumps(normalized), encoding="utf-8")

def _write_sharegpt_fixture(path: Path, prompts: list[str] | None = None) -> None:
    payload = prompts or [
        "short prompt",
        "medium length prompt for collect testing",
        "a substantially longer prompt used to exercise the collect length bucket planner",
        "another realistic prompt about GPU inference and batch scheduling",
        "write a concise explanation of paged attention",
        "summarize sampling configuration tradeoffs",
        "draft a review request for a pull request",
        "list the main tensor shapes in an attention layer",
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

def _build_modal_plan_from_run(run_dir: Path, **kwargs: object) -> dict:
    return build_modal_probe_plan(
        probe_plan=build_probe_plan(load_approved_targets(run_dir / "config" / "approved_targets.json")),
        output_dir=run_dir / ".modal_tmp",
        **kwargs,
    )

def _collect_strategy() -> dict:
    return {
        "batch_sizes": [1],
        "max_new_tokens": 16,
        "supplemental_runs": [_supplemental_run()],
        "max_captures_per_target": 8,
    }

def _supplemental_run(
    *,
    name: str = "sampling_supplemental",
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    allowed_op_types: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "sampling_params": {"temperature": temperature, "top_k": top_k, "top_p": top_p},
        "allowed_op_types": ["sampling"] if allowed_op_types is None else allowed_op_types,
    }

def _write_collect_report(run_dir: Path, *, plan: dict, manifest: dict | None = None) -> None:
    update_run_report(run_dir, collect={"plan": plan, "manifest": manifest or {"summary": {}}})

class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

class _FakeWrapper:
    def __init__(self, page_size: int | None = None) -> None:
        if page_size is not None:
            self.page_size = page_size

def _write_definition(root: Path, name: str, op_type: str = "rmsnorm") -> Path:
    path = root / op_type / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "name": name,
            "op_type": op_type,
            "axes": {},
            "inputs": {},
            "outputs": {},
            "reference": "def run(): pass\n",
        }),
        encoding="utf-8",
    )
    return path


__all__ = [name for name in globals() if not name.startswith('__')]
