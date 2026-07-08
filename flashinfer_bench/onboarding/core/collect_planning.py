"""Collect planning: observed events and definitions into collect targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flashinfer_bench.onboarding.core.schemas import (
    CollectPlan,
    CollectTarget,
    DefinitionRef,
    ProbePlan,
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_definitions(definitions_dir: Path) -> dict[str, DefinitionRef]:
    """Load minimal definition metadata by definition name."""
    definitions: dict[str, DefinitionRef] = {}
    for path in sorted(definitions_dir.rglob("*.json")):
        data = _load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"definition must be an object: {path}")
        name = data.get("name")
        op_type = data.get("op_type")
        if not isinstance(name, str) or not name:
            raise ValueError(f"definition has invalid name: {path}")
        if not isinstance(op_type, str) or not op_type:
            raise ValueError(f"definition {name} has invalid op_type: {path}")
        raw_tags = data.get("tags", [])
        if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
            raise ValueError(f"definition {name} has invalid tags: {path}")
        raw_axes = data.get("axes", {})
        if not isinstance(raw_axes, dict):
            raise ValueError(f"definition {name} has invalid axes: {path}")
        if name in definitions:
            raise ValueError(f"duplicate definition name {name}: {path}")
        definitions[name] = DefinitionRef(
            name=name,
            op_type=op_type,
            path=path,
            tags=raw_tags,
            axes=raw_axes,
        )
    return definitions


def build_collect_plan_from_probe_plan(
    *,
    definitions: dict[str, DefinitionRef],
    probe_plan: ProbePlan,
    events: list[dict[str, Any]],
    definition_aliases: dict[str, str] | None = None,
) -> CollectPlan:
    """Build collect targets from already-reviewed probe targets."""
    collect_targets: list[CollectTarget] = []
    skipped: list[dict[str, str]] = list(probe_plan.skipped)
    seen: set[tuple[str, str]] = set()
    aliases = definition_aliases or {}
    events_by_target = _events_by_target_definition(events)

    for target in probe_plan.targets:
        if not target.collect:
            skipped.append({
                "name": target.name,
                "reason": "collect is false",
            })
            continue
        if target.backend != "flashinfer" and not target.definition_name:
            skipped.append({
                "name": target.name,
                "reason": f"non-fitrace collect target has no reviewed definition_name: {target.backend}",
            })
            continue
        matched_definitions = []
        for raw_definition_name in sorted(events_by_target.get(target.name, set())):
            definition_name = aliases.get(raw_definition_name, raw_definition_name)
            definition = definitions.get(definition_name)
            if definition is None:
                skipped.append({
                    "name": target.name,
                    "reason": f"event referenced missing definition: {raw_definition_name}",
                })
                continue
            page_size_axis = definition.axes.get("page_size")
            matched_page_size = (
                int(page_size_axis["value"])
                if isinstance(page_size_axis, dict) and isinstance(page_size_axis.get("value"), int)
                else target.page_size
            )
            matched_definitions.append((definition, matched_page_size))
        if not matched_definitions:
            skipped.append({"name": target.name, "reason": "no event-linked definition"})
            continue

        for definition, matched_page_size in matched_definitions:
            key = (target.name, definition.name)
            if key in seen:
                skipped.append({
                    "name": target.name,
                    "reason": f"duplicate collect target for definition: {definition.name}",
                })
                continue
            seen.add(key)
            collect_targets.append(
                CollectTarget(
                    name=target.name,
                    definition_name=definition.name,
                    op_type=definition.op_type,
                    target=target.target,
                    backend=target.backend,
                    collect=target.collect,
                    definition_path=definition.path,
                    page_size=matched_page_size,
                )
            )

    collect_targets.sort(key=lambda item: (item.name, item.definition_name))
    return CollectPlan(targets=collect_targets, skipped=skipped)


def _events_by_target_definition(events: list[dict[str, Any]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for event in events:
        if bool(event.get("is_warmup")):
            continue
        name = event.get("name")
        definition_name = event.get("definition_name")
        if isinstance(name, str) and name and isinstance(definition_name, str) and definition_name:
            grouped.setdefault(name, set()).add(definition_name)
    return grouped
