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
    seen_targets: dict[tuple[Any, ...], str] = {}
    for item in payload:
        name = str(item.get("name") or "<unknown>")
        role = item.get("role", "target")
        if role == "target":
            target = item.get("target")
            module = item.get("module")
            attr = item.get("attr")
            backend = item.get("backend")
            definition_source = item.get("definition_source")
            if not isinstance(target, str) or not target:
                raise ValueError(f"approved target {name} has no hook target")
            if not isinstance(module, str) or not module or not isinstance(attr, str) or not attr:
                raise ValueError(f"approved target {name} has no explicit module/attr")
            if backend == "flashinfer" and definition_source not in {"fitrace", "manual"}:
                raise ValueError(f"approved FlashInfer target {name} must use definition_source=fitrace or manual")
            runtime_key = (
                target,
                module,
                attr,
                backend,
                item.get("op_type"),
                item.get("variant"),
                item.get("probe_mode", "default"),
                item.get("page_size"),
                item.get("dispatch_value"),
            )
            previous = seen_targets.get(runtime_key)
            if previous is not None:
                raise ValueError(f"approved target {name} duplicates runtime target {previous}")
            seen_targets[runtime_key] = name
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json_files(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {}
    files: dict[str, Any] = {}
    for path in sorted(root.rglob("*.json")):
        files[str(path.relative_to(root))] = _load_json(path)
    return files


def _reset_reviewed_draft_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _approved_from_candidates(raw: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
        promoted = {
            key: item[key]
            for key in APPROVED_TARGET_FIELD_ORDER
            if key in item
        }
        if promoted.get("role", "target") == "target":
            promoted["collect"] = True
        approved.append(promoted)
    return approved, approved_items, skipped


def promoted_config_sync_errors(
    *,
    run: Path | None = None,
    proposal_dir: Path | None = None,
    config_dir: Path | None = None,
) -> list[str]:
    """Return config/proposal sync errors without writing files."""
    if proposal_dir is None:
        if run is None:
            raise ValueError("sync check requires --run or --proposal-dir")
        run_dir = _resolve_run_dir(run)
        proposal_dir = _resolve_proposal_dir(run_dir)
        resolved_config_dir = config_dir or (run_dir / "config")
    else:
        if not proposal_dir.exists():
            return []
        proposal_dir = _resolve_proposal_dir(proposal_dir)
        resolved_config_dir = config_dir or _resolve_config_dir_from_proposal(proposal_dir)

    candidates_path = proposal_dir / "candidate_targets.json"
    if not candidates_path.exists():
        return []
    raw = _load_json(candidates_path)
    if not isinstance(raw, list):
        return [f"candidate_targets.json must be a list: {candidates_path}"]

    approved, approved_items, _skipped = _approved_from_candidates(raw)
    if not approved:
        return [f"no candidates with status=approved found in: {candidates_path}"]

    errors: list[str] = []
    try:
        _validate_approved_targets(approved)
    except Exception as exc:
        errors.append(str(exc))

    approved_targets_path = resolved_config_dir / "approved_targets.json"
    if not approved_targets_path.exists():
        errors.append(f"missing promoted approved targets: {approved_targets_path}")
    else:
        current = _load_json(approved_targets_path)
        if _canonical_json(current) != _canonical_json(approved):
            errors.append(f"config/approved_targets.json is out of sync with proposal/candidate_targets.json")

    expected_definitions: dict[str, Any] = {}
    expected_hints: dict[str, Any] = {}
    for item in approved_items:
        paths = _draft_pair_paths(proposal_dir, item)
        if paths is None:
            continue
        definition_src, hints_src = paths
        op_type = str(item["op_type"])
        definition_name = str(item["definition_name"])
        rel = f"{op_type}/{definition_name}.json"
        expected_definitions[rel] = _load_json(definition_src)
        expected_hints[rel] = _load_json(hints_src)

    current_definitions = _load_json_files(resolved_config_dir / "definitions")
    current_hints = _load_json_files(resolved_config_dir / "definition_hints")
    if _canonical_json(current_definitions) != _canonical_json(expected_definitions):
        errors.append("config/definitions is out of sync with approved proposal definitions")
    if _canonical_json(current_hints) != _canonical_json(expected_hints):
        errors.append("config/definition_hints is out of sync with approved proposal definition_hints")
    return errors


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

    approved, approved_items, skipped = _approved_from_candidates(raw)

    if not approved:
        raise ValueError(f"no candidates with status=approved found in: {candidates_path}")

    _validate_approved_targets(approved)
    approved_targets_path = output_path or (resolved_config_dir / "approved_targets.json")
    _write_json(approved_targets_path, approved)

    _reset_reviewed_draft_dir(resolved_config_dir / "definitions")
    _reset_reviewed_draft_dir(resolved_config_dir / "definition_hints")

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
