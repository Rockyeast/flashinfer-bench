"""Prepare independent reference tests for dataset submission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from flashinfer_bench.data.definition import Definition

_FI_GENERATORS: dict[str, Callable[[Path, Definition], str]] = {}


def _fi_generator(
    api_path: str,
) -> Callable[[Callable[[Path, Definition], str]], Callable[[Path, Definition], str]]:
    def register(function: Callable[[Path, Definition], str]) -> Callable[[Path, Definition], str]:
        _FI_GENERATORS[api_path] = function
        return function

    return register


def prepare_reference_tests(
    *, run_dir: Path, tests_dir: Path | None = None, overwrite: bool = False
) -> dict[str, Any]:
    """Generate deterministic FI tests and report non-FI tests needing Agent review."""
    definitions_dir = run_dir / "output" / "definitions"
    destination = tests_dir or run_dir / "output" / "tests" / "references"
    generated: list[str] = []
    existing: list[str] = []
    agent_required: list[str] = []
    unsupported_fi: list[dict[str, str]] = []

    if not definitions_dir.exists():
        raise FileNotFoundError(f"definitions directory missing: {definitions_dir}")

    destination.mkdir(parents=True, exist_ok=True)
    for definition_path in sorted(definitions_dir.rglob("*.json")):
        definition = Definition.model_validate_json(definition_path.read_text(encoding="utf-8"))
        test_path = destination / f"test_{definition.name}.py"
        if test_path.exists() and not overwrite:
            existing.append(str(test_path))
            continue

        fi_apis = [
            tag.removeprefix("fi_api:") for tag in definition.tags if tag.startswith("fi_api:")
        ]
        if not fi_apis:
            agent_required.append(str(definition_path))
            continue
        if len(fi_apis) != 1:
            unsupported_fi.append(
                {
                    "definition": str(definition_path),
                    "reason": f"expected exactly one fi_api tag; found {fi_apis}",
                }
            )
            continue

        api_path = fi_apis[0]
        generator = _FI_GENERATORS.get(api_path)
        if generator is None:
            unsupported_fi.append(
                {
                    "definition": str(definition_path),
                    "reason": f"no deterministic test generator for {api_path}",
                }
            )
            continue
        relative = definition_path.relative_to(run_dir / "output")
        test_path.write_text(generator(relative, definition), encoding="utf-8")
        generated.append(str(test_path))

    return {
        "schema_version": 1,
        "tests_dir": str(destination),
        "generated_fi_tests": generated,
        "existing_tests": existing,
        "agent_required_definitions": agent_required,
        "unsupported_fi_definitions": unsupported_fi,
    }


def write_reference_test_report(run_dir: Path, report: dict[str, Any]) -> Path:
    path = run_dir / "reports" / "reference_test_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_reference_test_preparation(
    *,
    run_dir: Path,
    tests_dir: Path | None = None,
    overwrite: bool = False,
    agent: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Prepare reference tests and optionally ask Agent to fill non-FI cases."""
    report = prepare_reference_tests(run_dir=run_dir, tests_dir=tests_dir, overwrite=overwrite)
    if agent and report["agent_required_definitions"]:
        if agent != "codex":
            raise ValueError(f"unsupported reference-test agent: {agent}")

        from flashinfer_bench.onboarding.agent_analysis import run_reference_test_analysis

        requested = list(report["agent_required_definitions"])
        report["agent_returncode"] = run_reference_test_analysis(
            run_dir=run_dir, definitions=requested, tests_dir=Path(report["tests_dir"])
        )
        generated_by_agent: list[str] = []
        still_missing: list[str] = []
        for definition_path in requested:
            test_path = Path(report["tests_dir"]) / f"test_{Path(definition_path).stem}.py"
            if test_path.exists():
                generated_by_agent.append(str(test_path))
            else:
                still_missing.append(definition_path)
        report["agent_requested_definitions"] = requested
        report["agent_generated_tests"] = generated_by_agent
        report["agent_required_definitions"] = still_missing

    return report, write_reference_test_report(run_dir, report)


def _const_axis(definition: Definition, name: str) -> int:
    axis = definition.axes.get(name)
    value = getattr(axis, "value", None)
    if not isinstance(value, int):
        raise ValueError(f"{definition.name} requires const axis {name}")
    return value


def _definition_loader(relative: Path) -> str:
    return f"""\
DEFINITION_PATH = Path(__file__).parents[2] / {str(relative)!r}


def _load_reference():
    definition = Definition.model_validate_json(DEFINITION_PATH.read_text(encoding="utf-8"))
    namespace = {{"torch": torch}}
    exec(definition.reference, namespace)
    return namespace["run"]
"""


@_fi_generator("flashinfer.norm.rmsnorm")
def _render_rmsnorm_test(relative: Path, definition: Definition) -> str:
    hidden_size = _const_axis(definition, "hidden_size")
    return f'''\
"""Generated independent reference test for {definition.name}."""

from pathlib import Path

import flashinfer
import pytest
import torch

from flashinfer_bench.data import Definition


{_definition_loader(relative)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_{definition.name}():
    batch_size = 8
    hidden_states = torch.randn(
        batch_size, {hidden_size}, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn({hidden_size}, device="cuda", dtype=torch.bfloat16)

    reference_run = _load_reference()
    expected = reference_run(hidden_states.clone(), weight.clone())
    actual = flashinfer.norm.rmsnorm(
        hidden_states.clone().contiguous(), weight.contiguous(), eps=1e-6
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=8e-3, rtol=1e-2)
'''


@_fi_generator("flashinfer.norm.fused_add_rmsnorm")
def _render_fused_add_rmsnorm_test(relative: Path, definition: Definition) -> str:
    hidden_size = _const_axis(definition, "hidden_size")
    return f'''\
"""Generated independent reference test for {definition.name}."""

from pathlib import Path

import flashinfer
import pytest
import torch

from flashinfer_bench.data import Definition


{_definition_loader(relative)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_{definition.name}():
    batch_size = 8
    hidden_states = torch.randn(
        batch_size, {hidden_size}, device="cuda", dtype=torch.bfloat16
    )
    residual = torch.randn_like(hidden_states)
    weight = torch.randn({hidden_size}, device="cuda", dtype=torch.bfloat16)

    reference_run = _load_reference()
    expected_output, expected_residual = reference_run(
        hidden_states.clone(), residual.clone(), weight.clone()
    )

    actual_output = hidden_states.clone().contiguous()
    actual_residual = residual.clone().contiguous()
    flashinfer.norm.fused_add_rmsnorm(
        actual_output, actual_residual, weight.contiguous(), eps=1e-6
    )

    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=8e-3, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_residual.float(), expected_residual.float(), atol=8e-3, rtol=1e-2
    )
'''
