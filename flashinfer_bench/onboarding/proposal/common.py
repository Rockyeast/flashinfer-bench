"""Shared helpers for offline onboarding proposal tools."""

from __future__ import annotations

import ast
import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import string
import time
from datetime import datetime
from dataclasses import fields
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from flashinfer_bench.onboarding.core.schemas import CaptureSpec


DEFAULT_COOKBOOK_REPO = "https://github.com/sgl-project/sgl-cookbook.git"
DEFAULT_FLASHINFER_ROOTS = (
    Path("agent_inputs/flashinfer/flashinfer"),
)
DEFAULT_SGLANG_ROOT = Path("agent_inputs/sglang/python/sglang")
DEFAULT_FLASHINFER_ROOT = Path("agent_inputs/flashinfer/flashinfer")
DEFAULT_COOKBOOK_ROOT = Path("agent_inputs/sgl-cookbook")
CAPTURE_SPEC_FIELDS = {field_info.name for field_info in fields(CaptureSpec)}
DEFINITION_REQUIRED_FIELDS = {"name", "op_type", "axes", "inputs", "outputs"}
HINT_REQUIRED_FIELDS = {"schema_version", "definition_name", "op_type", "inputs"}
KNOWN_NON_FITRACE_COLLECTABLE_OPS = {"rmsnorm", "silu_and_mul"}
COMPANION_REQUIRED_WRAPPER_SUFFIXES = (
    "BatchDecodeWithPagedKVCacheWrapper.run",
    "BatchPrefillWithPagedKVCacheWrapper.run",
    "BatchPrefillWithRaggedKVCacheWrapper.run",
)
FLASHINFER_ATTENTION_TARGET_PREFIXES = ("flashinfer.decode.", "flashinfer.prefill.")
CANDIDATE_MERGE_META_FIELDS = {"name", "status", "evidence", "review_note"}

def slug_model_name(model_name: str) -> str:
    """Return a stable local filename slug for a HF model name."""
    last = model_name.strip().split("/")[-1]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", last).strip("_").lower()
    return slug or "model"


def _default_run_prefix(model_name: str) -> Path:
    return Path(slug_model_name(model_name)) / f"{datetime.now().strftime('%Y%m%d')}_firstpass"


def _default_hf_config_path(model_name: str) -> Path:
    return Path("agent_inputs/config") / f"{slug_model_name(model_name)}.json"


def _default_merge_output_dir(run_prefix: Path) -> Path:
    base = _resolve_run_dir(run_prefix)
    return base.with_name(f"{base.name}_merged") / "proposal"


def _find_codex_binary() -> Path:
    resolved = shutil.which("codex")
    if resolved:
        return Path(resolved)
    candidates = sorted(Path.home().glob(".vscode-server/extensions/*/bin/linux-x86_64/codex"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("codex binary not found on PATH or under ~/.vscode-server/extensions")


def _agent_command_from_shortcut(agent: str | None) -> tuple[list[str] | None, dict[str, str] | None]:
    if agent is None:
        return None, None
    if agent != "codex":
        raise ValueError(f"unsupported agent shortcut: {agent}")
    codex_bin = _find_codex_binary()
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(codex_bin.parent), env.get("PATH", "")])
    return [
        str(codex_bin),
        "exec",
        "-C",
        str(Path.cwd()),
        "-s",
        "workspace-write",
        "--ephemeral",
        "-",
    ], env



def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _resolve_proposal_dir(path: Path) -> Path:
    if (path / "candidate_targets.json").exists():
        return path
    nested = path / "proposal"
    if (nested / "candidate_targets.json").exists():
        return nested
    raise FileNotFoundError(f"proposal candidate_targets.json not found under: {path}")


def _resolve_run_dir(run: Path) -> Path:
    return run if run.exists() else Path("runs") / run


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)

def _brief(value: Any, *, limit: int = 500) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."



__all__ = [name for name in globals() if not name.startswith("__")]
