"""Prepare local inputs for proposal agents."""

from __future__ import annotations

from ..common import *  # noqa: F403

def _run_git(args: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": ["git", *args],
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _ensure_cookbook_cache(*, cache_root: Path, refresh: bool, repo_url: str) -> dict[str, Any]:
    if cache_root.exists():
        if not (cache_root / ".git").exists():
            return {"path": str(cache_root), "status": "cache_exists_not_git", "ok": False}
        if not refresh:
            return {"path": str(cache_root), "status": "cache_exists", "ok": True}
        result = _run_git(["pull", "--ff-only"], cwd=cache_root)
        return {
            "path": str(cache_root),
            "status": "cache_updated" if result["returncode"] == 0 else "cache_update_failed",
            "ok": result["returncode"] == 0,
            "git": result,
        }
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    result = _run_git(["clone", "--depth", "1", repo_url, str(cache_root)])
    return {
        "path": str(cache_root),
        "status": "cache_cloned" if result["returncode"] == 0 else "cache_clone_failed",
        "ok": result["returncode"] == 0,
        "git": result,
    }


def ensure_sgl_cookbook(
    *,
    root: Path,
    refresh: bool,
    repo_url: str = DEFAULT_COOKBOOK_REPO,
    cache_root: Path = DEFAULT_COOKBOOK_CACHE_ROOT,
) -> dict[str, Any]:
    """Ensure root/sgl-cookbook is a lightweight reference to the shared cache."""
    target = root / "sgl-cookbook"
    cache = _ensure_cookbook_cache(cache_root=cache_root, refresh=refresh, repo_url=repo_url)
    if not cache.get("ok"):
        return {"path": str(target), "cache_path": str(cache_root), "status": cache["status"], "ok": False, "cache": cache}

    root.mkdir(parents=True, exist_ok=True)
    resolved_cache = cache_root.resolve()
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            if target.resolve() != resolved_cache:
                target.unlink()
                target.symlink_to(resolved_cache, target_is_directory=True)
        elif target.resolve() != resolved_cache:
            return {
                "path": str(target),
                "cache_path": str(cache_root),
                "status": "target_exists_not_link",
                "ok": False,
                "cache": cache,
                "error": "remove or move the existing agent_inputs/sgl-cookbook directory before linking the shared cache",
            }
    else:
        target.symlink_to(resolved_cache, target_is_directory=True)

    return {
        "path": str(target),
        "cache_path": str(cache_root),
        "status": "linked",
        "ok": True,
        "cache": cache,
    }


def fetch_hf_config(*, model_name: str, config_dir: Path, refresh: bool) -> dict[str, Any]:
    """Download one HuggingFace config.json unless an existing copy is accepted."""
    config_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_model_name(model_name)
    path = config_dir / f"{slug}.json"
    if path.exists() and not refresh:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        return {
            "model": model_name,
            "slug": slug,
            "path": str(path),
            "status": "exists",
            "ok": True,
            "model_type": data.get("model_type"),
        }

    url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    req = Request(url, headers={"User-Agent": "flashinfer-bench-onboarding-config-fetch"})
    try:
        with urlopen(req, timeout=60) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "model": model_name,
            "slug": slug,
            "path": str(path),
            "status": "downloaded",
            "ok": True,
            "model_type": data.get("model_type"),
        }
    except Exception as exc:  # noqa: BLE001 - report diagnostics for CLI users
        return {
            "model": model_name,
            "slug": slug,
            "path": str(path),
            "status": "download_failed",
            "ok": False,
            "error": str(exc),
        }


def _version_key(path: Path) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", path.name)
    return tuple(int(number) for number in numbers)


def _latest_generated_dirs(cookbook_root: Path) -> list[Path]:
    generated = cookbook_root / "data/models/generated"
    if not generated.exists():
        return []
    dirs = [path for path in generated.iterdir() if path.is_dir()]
    return sorted(dirs, key=_version_key, reverse=True)


def _tokens_for_model(model_name: str) -> set[str]:
    text = model_name.split("/")[-1].lower()
    pieces = re.split(r"[^a-z0-9]+", text)
    tokens = {piece for piece in pieces if len(piece) >= 3}
    if "qwen3" in text:
        tokens.add("qwen3")
    if "tinyllama" in text:
        tokens.add("tinyllama")
        tokens.add("llama")
    return tokens


def find_cookbook_candidates(*, cookbook_root: Path, model_name: str, limit: int = 8) -> dict[str, Any]:
    """Find likely sgl-cookbook YAMLs. Missing is diagnostic, not fatal."""
    latest_dirs = _latest_generated_dirs(cookbook_root)
    tokens = _tokens_for_model(model_name)
    matches: list[str] = []
    searched: list[str] = []
    for directory in latest_dirs:
        searched.append(str(directory))
        for yaml_path in sorted(directory.glob("*.yaml")):
            name = yaml_path.stem.lower()
            if any(token in name for token in tokens):
                matches.append(str(yaml_path))
                if len(matches) >= limit:
                    return {
                        "model": model_name,
                        "status": "found",
                        "ok": True,
                        "matches": matches,
                        "searched_latest_first": searched,
                    }
    status = "found" if matches else "missing"
    return {"model": model_name, "status": status, "ok": True, "matches": matches, "searched_latest_first": searched}


def prepare_agent_inputs(
    *,
    models: list[str],
    output_root: Path,
    refresh: bool = False,
    cookbook_repo: str = DEFAULT_COOKBOOK_REPO,
    cookbook_cache: Path = DEFAULT_COOKBOOK_CACHE_ROOT,
    check_sglang_root: Path | None = None,
    check_flashinfer_root: Path | None = None,
) -> dict[str, Any]:
    """Prepare external inputs and return a reproducible report."""
    output_root.mkdir(parents=True, exist_ok=True)
    cookbook = ensure_sgl_cookbook(root=output_root, refresh=refresh, repo_url=cookbook_repo, cache_root=cookbook_cache)
    configs = [
        fetch_hf_config(model_name=model, config_dir=output_root / "config", refresh=refresh)
        for model in models
    ]

    cookbook_root = output_root / "sgl-cookbook"
    cookbook_matches = [
        find_cookbook_candidates(cookbook_root=cookbook_root, model_name=model)
        for model in models
    ]

    source_checks = []
    for label, path in (("sglang", check_sglang_root), ("flashinfer", check_flashinfer_root)):
        if path is None:
            source_checks.append({"name": label, "path": None, "status": "not_checked", "ok": True})
        else:
            source_checks.append({
                "name": label,
                "path": str(path),
                "status": "exists" if path.exists() else "missing",
                "ok": path.exists(),
            })

    return {
        "summary": {
            "models": len(models),
            "configs_ok": sum(1 for item in configs if item.get("ok")),
            "cookbook_ok": cookbook.get("ok", False),
            "cookbook_matches": sum(1 for item in cookbook_matches if item.get("matches")),
            "source_checks_ok": sum(1 for item in source_checks if item.get("ok")),
        },
        "output_root": str(output_root),
        "refresh": refresh,
        "sgl_cookbook": cookbook,
        "configs": configs,
        "cookbook_candidates": cookbook_matches,
        "source_checks": source_checks,
    }
