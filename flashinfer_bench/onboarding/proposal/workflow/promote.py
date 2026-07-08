"""Promote human-approved proposal targets into runtime config."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import shutil
import tempfile

from flashinfer_bench.onboarding.core.probe_planning import load_approved_targets
from flashinfer_bench.onboarding.core.schemas import ApprovedTarget

from ..common import *  # noqa: F403

APPROVED_TARGET_FIELD_ORDER = tuple(field.name for field in fields(ApprovedTarget))


def _resolve_config_dir_from_proposal(proposal_dir: Path) -> Path:
    if proposal_dir.name == "proposal":
        return proposal_dir.parent / "config"
    return proposal_dir / "config"


def _validate_approved_targets(payload: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        load_approved_targets(path)
    finally:
        path.unlink(missing_ok=True)


def _draft_pair_paths(proposal_dir: Path, item: dict[str, Any]) -> tuple[Path, Path] | None:
    if item.get("definition_source") != "agent":
        return None
    op_type = item.get("op_type")
    definition_name = item.get("definition_name")
    name = str(item.get("name") or definition_name or "unknown")
    if not isinstance(op_type, str) or not op_type:
        raise ValueError(f"approved agent target {name} has no op_type")
    if not isinstance(definition_name, str) or not definition_name:
        raise ValueError(f"approved agent target {name} has no definition_name")
    return (
        proposal_dir / "definitions" / op_type / f"{definition_name}.json",
        proposal_dir / "definition_hints" / op_type / f"{definition_name}.json",
    )


def _copy_reviewed_draft(src: Path, dst: Path, *, label: str) -> dict[str, str]:
    if not src.exists():
        raise FileNotFoundError(f"approved {label} draft not found: {src}")
    data = _load_json(src)
    if not isinstance(data, dict):
        raise ValueError(f"approved {label} draft must be a JSON object: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": str(src), "destination": str(dst)}


def promote_approved_targets(
    *,
    run: Path | None = None,
    proposal_dir: Path | None = None,
    config_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Write reviewed config artifacts from proposal candidates marked approved."""
    if proposal_dir is None:
        if run is None:
            raise ValueError("promote-approved requires --run or --proposal-dir")
        run_dir = _resolve_run_dir(run)
        proposal_dir = _resolve_proposal_dir(run_dir)
        resolved_config_dir = config_dir or (run_dir / "config")
    else:
        proposal_dir = _resolve_proposal_dir(proposal_dir)
        resolved_config_dir = config_dir or _resolve_config_dir_from_proposal(proposal_dir)

    candidates_path = proposal_dir / "candidate_targets.json"
    raw = _load_json(candidates_path)
    if not isinstance(raw, list):
        raise ValueError(f"candidate_targets.json must be a list: {candidates_path}")

    approved: list[dict[str, Any]] = []
    approved_items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            skipped.append({"index": index, "name": f"#{index}", "status": "invalid"})
            continue
        name = str(item.get("name") or f"#{index}")
        status = item.get("status")
        if status != "approved":
            skipped.append({"index": index, "name": name, "status": status or "unapproved"})
            continue
        approved_items.append(item)
        approved.append({
            key: item[key]
            for key in APPROVED_TARGET_FIELD_ORDER
            if key in item
        })

    if not approved:
        raise ValueError(f"no candidates with status=approved found in: {candidates_path}")

    _validate_approved_targets(approved)
    approved_targets_path = output_path or (resolved_config_dir / "approved_targets.json")
    _write_json(approved_targets_path, approved)

    copied_definitions: list[dict[str, str]] = []
    copied_hints: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for item in approved_items:
        paths = _draft_pair_paths(proposal_dir, item)
        if paths is None:
            continue
        definition_src, hints_src = paths
        op_type = str(item["op_type"])
        definition_name = str(item["definition_name"])
        pair = (op_type, definition_name)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        definition_dst = resolved_config_dir / "definitions" / op_type / f"{definition_name}.json"
        hints_dst = resolved_config_dir / "definition_hints" / op_type / f"{definition_name}.json"
        copied_definitions.append(_copy_reviewed_draft(definition_src, definition_dst, label="definition"))
        copied_hints.append(_copy_reviewed_draft(hints_src, hints_dst, label="definition hints"))

    return {
        "summary": {
            "approved": len(approved),
            "skipped": len(skipped),
            "definitions": len(copied_definitions),
            "definition_hints": len(copied_hints),
            "output": str(approved_targets_path),
        },
        "inputs": {
            "proposal_dir": str(proposal_dir),
            "candidates": str(candidates_path),
        },
        "outputs": {
            "approved_targets": str(approved_targets_path),
            "definitions": copied_definitions,
            "definition_hints": copied_hints,
        },
        "skipped": skipped,
    }
