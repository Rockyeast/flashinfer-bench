"""Publication cleanup and deterministic submission checks."""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from flashinfer_bench.data.definition import Definition

INTERNAL_TAG_PREFIXES = (
    "sglang_module:",
    "sglang_callable:",
    "sglang_input:",
)
PUBLIC_STATUS_TAGS = {
    "status:verified",
    "status:unverified",
    "status:reference",
}
PUBLIC_TAG_PREFIXES = (
    "fi_api:",
    "model:",
    "quantization:",
    "quant:",
    "stage:",
    "tp:",
    "ep:",
    "gpu:",
    "layout:",
    "routing:",
    "sparse:",
    "sparsity:",
)
PUBLIC_BARE_TAGS = {"fused"}
UPSTREAM_DATASET_URL = "https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace"


def publication_definition(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a publishable definition and capture-only metadata for its sidecar."""
    published = copy.deepcopy(value)
    tags = published.get("tags")
    tags = list(tags) if isinstance(tags, list) else []
    capture_tags = [
        tag
        for tag in tags
        if isinstance(tag, str) and tag.startswith(INTERNAL_TAG_PREFIXES)
    ]
    source_statuses = [
        tag for tag in tags if isinstance(tag, str) and tag.startswith("status:")
    ]
    published_tags = [
        tag
        for tag in tags
        if isinstance(tag, str)
        and not tag.startswith(INTERNAL_TAG_PREFIXES)
        and tag != "status:source_reviewed"
    ]
    if "status:source_reviewed" in source_statuses and not any(
        tag in PUBLIC_STATUS_TAGS for tag in published_tags
    ):
        published_tags.append("status:unverified")
    published["tags"] = published_tags
    sidecar = {
        "name": published.get("name"),
        "op_type": published.get("op_type"),
        "capture_tags": capture_tags,
        "source_status_tags": source_statuses,
    }
    return published, sidecar


def check_submission(
    *,
    output_dir: Path,
    upstream_dataset: Path,
    reference_tests_dir: Path | None = None,
) -> dict[str, Any]:
    """Check only artifacts that would be submitted to flashinfer-trace."""
    definitions_dir = output_dir / "definitions"
    tests_dir = reference_tests_dir or output_dir / "tests" / "references"
    upstream_definitions = _definitions_root(upstream_dataset)
    errors: list[dict[str, str]] = []
    new_definitions: list[str] = []
    existing_definitions: list[str] = []

    if not definitions_dir.exists():
        errors.append({"path": str(definitions_dir), "reason": "definitions directory missing"})
    if not upstream_definitions.exists():
        errors.append(
            {
                "path": str(upstream_definitions),
                "reason": (
                    "upstream definitions missing from the refreshed remote snapshot"
                ),
            }
        )

    upstream_by_name: dict[str, list[Path]] = {}
    if upstream_definitions.exists():
        for path in sorted(upstream_definitions.rglob("*.json")):
            upstream_by_name.setdefault(path.stem, []).append(path)

    if definitions_dir.exists():
        for path in sorted(definitions_dir.rglob("*.json")):
            relative = path.relative_to(definitions_dir)
            value = _load_definition(path, relative, errors)
            if value is None:
                continue
            _public_tag_errors(value, relative, errors)

            upstream_path = upstream_definitions / relative
            if upstream_path.exists():
                upstream_value = _read_json(upstream_path)
                if upstream_value == value:
                    existing_definitions.append(str(relative))
                    continue
                errors.append(
                    {
                        "path": str(relative),
                        "reason": f"definition conflicts with upstream {upstream_path}",
                    }
                )
                continue
            duplicate_names = upstream_by_name.get(path.stem, [])
            if duplicate_names:
                errors.append(
                    {
                        "path": str(relative),
                        "reason": (
                            "definition name already exists upstream at "
                            + ", ".join(str(item) for item in duplicate_names)
                        ),
                    }
                )
                continue

            new_definitions.append(str(relative))
            _description_errors(value, relative, errors)
            _reference_test_errors(path.stem, tests_dir, errors)

    return {
        "schema_version": 1,
        "summary": {
            "ok": not errors and bool(new_definitions or existing_definitions),
            "definitions": len(new_definitions) + len(existing_definitions),
            "new_definitions": len(new_definitions),
            "existing_definitions": len(existing_definitions),
            "errors": len(errors),
        },
        "output_dir": str(output_dir),
        "upstream_dataset": str(upstream_dataset),
        "reference_tests_dir": str(tests_dir),
        "new_definitions": new_definitions,
        "existing_definitions": existing_definitions,
        "errors": errors,
    }


def check_submission_against_latest(
    *,
    output_dir: Path,
    reference_tests_dir: Path | None = None,
) -> dict[str, Any]:
    """Refresh upstream main and check artifacts against that exact revision."""
    with _latest_upstream_dataset() as (upstream_dataset, revision):
        report = check_submission(
            output_dir=output_dir,
            upstream_dataset=upstream_dataset,
            reference_tests_dir=reference_tests_dir,
        )
    report["upstream_dataset"] = UPSTREAM_DATASET_URL
    report["upstream_revision"] = revision
    return report


@contextmanager
def _latest_upstream_dataset() -> Iterator[tuple[Path, str]]:
    cache = Path.home() / ".cache" / "flashinfer_bench" / "flashinfer-trace-upstream-v2"
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not cache.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "--branch=main",
                UPSTREAM_DATASET_URL,
                str(cache),
            ],
            check=True,
            env=environment,
        )
    elif not (cache / ".git").is_dir():
        raise RuntimeError(f"upstream cache is not a git checkout: {cache}")

    remote_url = subprocess.run(
        ["git", "-C", str(cache), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote_url != UPSTREAM_DATASET_URL:
        raise RuntimeError(f"unexpected upstream cache remote: {remote_url}")

    subprocess.run(
        [
            "git",
            "-C",
            str(cache),
            "fetch",
            "--force",
            "--depth=1",
            "origin",
            "refs/heads/main:refs/remotes/origin/main",
        ],
        check=True,
        env=environment,
    )
    revision = subprocess.run(
        ["git", "-C", str(cache), "rev-parse", "origin/main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="flashinfer-trace-upstream-") as temp:
        root = Path(temp)
        archive = root / "definitions.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(cache),
                "archive",
                "--format=tar",
                f"--output={archive}",
                "origin/main",
                "definitions",
            ],
            check=True,
            env=environment,
        )
        with tarfile.open(archive, mode="r:") as bundle:
            bundle.extractall(root, filter="data")
        archive.unlink()
        yield root, revision


def _definitions_root(dataset: Path) -> Path:
    return dataset if dataset.name == "definitions" else dataset / "definitions"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_definition(
    path: Path, relative: Path, errors: list[dict[str, str]]
) -> dict[str, Any] | None:
    value = _read_json(path)
    if not isinstance(value, dict):
        errors.append({"path": str(relative), "reason": "invalid definition JSON"})
        return None
    try:
        Definition.model_validate(value)
    except Exception as exc:  # noqa: BLE001 - preserve complete schema error
        errors.append({"path": str(relative), "reason": f"invalid definition: {exc}"})
        return None
    return value


def _public_tag_errors(
    value: dict[str, Any], relative: Path, errors: list[dict[str, str]]
) -> None:
    tags = value.get("tags")
    if not isinstance(tags, list):
        errors.append({"path": str(relative), "reason": "tags must be a list"})
        return
    internal = [
        tag
        for tag in tags
        if isinstance(tag, str) and tag.startswith(INTERNAL_TAG_PREFIXES)
    ]
    if internal:
        errors.append(
            {
                "path": str(relative),
                "reason": f"capture-only tags must stay in sidecar: {internal}",
            }
        )
    statuses = [tag for tag in tags if isinstance(tag, str) and tag.startswith("status:")]
    if len(statuses) != 1 or statuses[0] not in PUBLIC_STATUS_TAGS:
        errors.append(
            {
                "path": str(relative),
                "reason": (
                    "published definition requires exactly one status tag from "
                    f"{sorted(PUBLIC_STATUS_TAGS)}; found {statuses}"
                ),
            }
        )
    unsupported = [
        tag
        for tag in tags
        if isinstance(tag, str)
        and tag not in PUBLIC_STATUS_TAGS
        and tag not in PUBLIC_BARE_TAGS
        and not tag.startswith(PUBLIC_TAG_PREFIXES)
    ]
    if unsupported:
        errors.append(
            {"path": str(relative), "reason": f"unsupported publication tags: {unsupported}"}
        )


def _description_errors(
    value: dict[str, Any], relative: Path, errors: list[dict[str, str]]
) -> None:
    fields: list[tuple[str, Any]] = [("description", value.get("description"))]
    for section in ("axes", "inputs", "outputs"):
        entries = value.get(section)
        if isinstance(entries, dict):
            fields.extend(
                (f"{section}.{name}.description", spec.get("description"))
                for name, spec in entries.items()
                if isinstance(spec, dict)
            )
    for field, description in fields:
        if not isinstance(description, str) or not description.strip():
            errors.append(
                {"path": str(relative), "reason": f"missing required {field}"}
            )


def _reference_test_errors(
    definition_name: str, tests_dir: Path, errors: list[dict[str, str]]
) -> None:
    path = tests_dir / f"test_{definition_name}.py"
    if not path.exists():
        errors.append({"path": str(path), "reason": "missing reference test"})
        return
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        errors.append({"path": str(path), "reason": f"invalid reference test: {exc}"})
        return
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in module.body
    ):
        errors.append({"path": str(path), "reason": "reference test defines no test_* function"})
