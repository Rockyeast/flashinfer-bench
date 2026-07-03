"""Merge review-only proposal bundles."""

from __future__ import annotations

from ..common import *  # noqa: F403

def _candidate_merge_key(item: dict[str, Any]) -> tuple[Any, ...]:
    def key_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return _json_key(value)

    role = item.get("role", "target")
    if role == "warmup":
        return ("warmup", key_value(item.get("module")), key_value(item.get("attr")))

    dispatch = item.get("dispatch")
    dispatch_field = dispatch.get("field") if isinstance(dispatch, dict) else None
    dispatch_value = item.get("page_size") if item.get("page_size") is not None else item.get("dispatch_value")
    module = item.get("module")
    attr = item.get("attr")
    target = item.get("target")
    hook_key = ("module_attr", module, attr) if module and attr else ("target", target)
    return (
        "target",
        tuple(key_value(value) for value in hook_key),
        key_value(item.get("backend")),
        key_value(item.get("definition_source")),
        key_value(item.get("op_type")),
        key_value(item.get("variant")),
        key_value(item.get("definition_name")),
        key_value(dispatch_field),
        key_value(dispatch_value),
    )


def _candidate_core(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in CANDIDATE_MERGE_META_FIELDS
    }


def _unique_json_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    output: list[Any] = []
    for value in values:
        key = _json_key(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _candidate_conflict_fields(items: list[dict[str, Any]]) -> list[str]:
    keys = sorted({key for item in items for key in item if key not in CANDIDATE_MERGE_META_FIELDS})
    fields: list[str] = []
    for key in keys:
        values = [_json_key(item.get(key)) for item in items]
        if len(set(values)) > 1:
            fields.append(key)
    return fields


def _merge_candidate_group(entries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidates = [entry["candidate"] for entry in entries]
    first_core = _candidate_core(candidates[0])
    if any(_candidate_core(candidate) != first_core for candidate in candidates[1:]):
        return None, {
            "kind": "candidate",
            "key": list(_candidate_merge_key(candidates[0])),
            "reason": "conflicting candidate fields",
            "fields": _candidate_conflict_fields(candidates),
            "variants": [
                {
                    "proposal": entry["proposal"],
                    "name": entry["candidate"].get("name"),
                    "candidate": entry["candidate"],
                }
                for entry in entries
            ],
        }

    merged = dict(candidates[0])
    evidence = []
    notes = []
    for entry in entries:
        candidate = entry["candidate"]
        raw_evidence = candidate.get("evidence")
        if isinstance(raw_evidence, list):
            evidence.extend(raw_evidence)
        note = candidate.get("review_note")
        if isinstance(note, str) and note.strip():
            notes.append(f"[{entry['label']}:{candidate.get('name', 'unknown')}] {note.strip()}")
    if evidence:
        merged["evidence"] = _unique_json_values(evidence)
    if notes:
        merged["review_note"] = "\n".join(dict.fromkeys(notes))
    return merged, None


def _draft_payload_without_description(payload: Any) -> Any:
    if not isinstance(payload, dict) or "description" not in payload:
        return payload
    stripped = dict(payload)
    stripped.pop("description", None)
    return stripped


def _copy_proposal_draft_files(
    *,
    kind: str,
    proposal_dirs: list[Path],
    output_dir: Path,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    copied = 0
    conflicts: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    by_relative: dict[Path, list[dict[str, Any]]] = {}
    for proposal_dir in proposal_dirs:
        root = proposal_dir / kind
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            rel = path.relative_to(root)
            by_relative.setdefault(rel, []).append({
                "proposal": str(proposal_dir),
                "path": path,
                "payload": _load_json(path),
            })

    for rel, entries in sorted(by_relative.items(), key=lambda item: str(item[0])):
        payload_keys = {_json_key(entry["payload"]) for entry in entries}
        if len(payload_keys) > 1:
            stripped_keys = {_json_key(_draft_payload_without_description(entry["payload"])) for entry in entries}
            if len(stripped_keys) == 1:
                warnings.append({
                    "kind": kind,
                    "path": str(rel),
                    "reason": "draft file descriptions differ",
                    "variants": [
                        {
                            "proposal": entry["proposal"],
                            "path": str(entry["path"]),
                        }
                        for entry in entries
                    ],
                })
                dst = output_dir / kind / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entries[0]["path"], dst)
                copied += 1
                continue
            conflicts.append({
                "kind": kind,
                "path": str(rel),
                "reason": "conflicting draft file payloads",
                "variants": [
                    {
                        "proposal": entry["proposal"],
                        "path": str(entry["path"]),
                    }
                    for entry in entries
                ],
            })
            continue
        dst = output_dir / kind / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entries[0]["path"], dst)
        copied += 1
    return copied, conflicts, warnings


def _merge_review_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Merged Proposal Review",
        "",
        f"- input proposals: {summary['proposals']}",
        f"- input candidates: {summary['input_candidates']}",
        f"- merged candidates: {summary['merged_candidates']}",
        f"- conflicts: {summary['conflicts']}",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- {path}" for path in report["inputs"])
    lines.extend(["", "## Conflicts", ""])
    if not report["conflicts"]:
        lines.append("- none")
    else:
        for item in report["conflicts"]:
            name = item.get("path") or item.get("key") or "unknown"
            fields = item.get("fields")
            detail = f"; fields={fields}" if fields else ""
            lines.append(f"- {item.get('kind', 'unknown')}: {name} ({item.get('reason', 'conflict')}{detail})")
    warnings = report.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings", ""])
        for item in warnings:
            name = item.get("path") or item.get("key") or "unknown"
            lines.append(f"- {item.get('kind', 'unknown')}: {name} ({item.get('reason', 'warning')})")
    lines.extend([
        "",
        "## Human Action",
        "",
        "Review conflicts before promoting anything into config/. This merged proposal is review-only and does not approve targets.",
        "",
    ])
    return "\n".join(lines)


def merge_proposals(*, proposal_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Merge multiple review-only proposal bundles into a union proposal.

    The merge is intentionally conservative: identical candidates are deduped,
    evidence is unioned, and conflicting candidate/definition/hints payloads are
    reported for human review instead of being auto-resolved.
    """
    if len(proposal_dirs) < 2:
        raise ValueError("merge-proposals requires at least two proposal dirs")
    resolved_dirs = [_resolve_proposal_dir(path) for path in proposal_dirs]
    output_resolved = output_dir.resolve()
    for proposal_dir in resolved_dirs:
        if output_resolved == proposal_dir.resolve():
            raise ValueError("merge-proposals output-dir must be distinct from input proposal dirs")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir / "definitions", ignore_errors=True)
    shutil.rmtree(output_dir / "definition_hints", ignore_errors=True)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    input_candidates = 0
    for index, proposal_dir in enumerate(resolved_dirs, start=1):
        raw = _load_json(proposal_dir / "candidate_targets.json")
        if not isinstance(raw, list):
            raise ValueError(f"candidate_targets.json must be a list: {proposal_dir}")
        label = proposal_dir.parent.name if proposal_dir.name == "proposal" else proposal_dir.name
        for item_index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"candidate #{item_index} must be an object: {proposal_dir}")
            input_candidates += 1
            entry = {
                "proposal": str(proposal_dir),
                "label": f"{label or f'proposal{index}'}",
                "candidate": item,
            }
            groups.setdefault(_candidate_merge_key(item), []).append(entry)

    merged_candidates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for entries in groups.values():
        merged, conflict = _merge_candidate_group(entries)
        if conflict is not None:
            conflicts.append(conflict)
            continue
        if merged is not None:
            merged_candidates.append(merged)
    merged_candidates.sort(key=lambda item: str(item.get("name") or _candidate_merge_key(item)))

    definitions_copied, definition_conflicts, definition_warnings = _copy_proposal_draft_files(
        kind="definitions",
        proposal_dirs=resolved_dirs,
        output_dir=output_dir,
    )
    hints_copied, hint_conflicts, hint_warnings = _copy_proposal_draft_files(
        kind="definition_hints",
        proposal_dirs=resolved_dirs,
        output_dir=output_dir,
    )
    conflicts.extend(definition_conflicts)
    conflicts.extend(hint_conflicts)
    warnings = definition_warnings + hint_warnings

    _write_json(output_dir / "candidate_targets.json", merged_candidates)
    (output_dir / "architecture.md").write_text(
        "# Merged Proposal\n\n"
        "This proposal was generated by `flashinfer_bench.onboarding.proposal_tools merge-proposals`.\n"
        "Use `merge_review.md` before promoting anything into config/.\n",
        encoding="utf-8",
    )
    report = {
        "summary": {
            "ok": not conflicts,
            "proposals": len(resolved_dirs),
            "input_candidates": input_candidates,
            "merged_candidates": len(merged_candidates),
            "candidate_conflicts": sum(1 for item in conflicts if item.get("kind") == "candidate"),
            "draft_file_conflicts": sum(1 for item in conflicts if item.get("kind") in {"definitions", "definition_hints"}),
            "conflicts": len(conflicts),
            "definitions_copied": definitions_copied,
            "definition_hints_copied": hints_copied,
            "warnings": len(warnings),
        },
        "inputs": [str(path) for path in resolved_dirs],
        "output_dir": str(output_dir),
        "conflicts": conflicts,
        "warnings": warnings,
    }
    _write_json(output_dir / "merge_report.json", report)
    (output_dir / "merge_review.md").write_text(_merge_review_markdown(report), encoding="utf-8")
    (output_dir / "review_checklist.md").write_text(
        "# Review Checklist\n\n"
        "- Read merge_review.md.\n"
        "- Resolve every conflict before promoting candidates into config/.\n"
        "- Run check-proposal on the merged proposal after manual conflict resolution.\n",
        encoding="utf-8",
    )
    return report


