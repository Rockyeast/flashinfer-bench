"""Build the small serializable plan shared by both onboarding stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SHAREGPT_PATH = Path("sharegpt_100.json")
PACKAGE_SHAREGPT_PATH = Path(__file__).resolve().parent / "sharegpt_100.json"
INFERENCEX_PROFILES = {"1k1k": (1024, 1024), "8k1k": (8192, 1024)}
DEFAULT_INFERENCEX_RANGE_RATIO = 0.8


def build_stage_plan(
    *,
    stage: str,
    config: dict[str, Any],
    reviewed_definitions: list[dict[str, Any]] | None = None,
    pass_modes: list[str] | None = None,
    page_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible Modal/SGLang plan."""
    if stage not in {"definitions", "workloads"}:
        raise ValueError("stage must be definitions or workloads")
    model_name = _non_empty_string(config.get("model_name"), "model_name")
    image = _non_empty_string(config.get("image"), "image")
    gpu = _non_empty_string(config.get("gpu"), "gpu")
    tp_size = _positive_int(config.get("tp_size"), "tp_size")
    max_new_tokens = _positive_int(config.get("max_new_tokens"), "max_new_tokens")
    batch_sizes = _positive_int_list(config.get("batch_sizes"), "batch_sizes")
    inferencex_profiles = _inferencex_profiles(config.get("inferencex_profiles"))
    if stage == "workloads" and inferencex_profiles:
        prompt_scenarios = _build_inferencex_scenarios(
            profiles=inferencex_profiles,
            batch_sizes=batch_sizes,
            range_ratio=_ratio(
                config.get("inferencex_range_ratio", DEFAULT_INFERENCEX_RANGE_RATIO),
                "inferencex_range_ratio",
            ),
            seed=_integer(config.get("inferencex_seed", 0), "inferencex_seed"),
        )
    else:
        prompt_scenarios = _build_prompt_scenarios(
            prompts=_load_prompts(_sharegpt_path()),
            batch_sizes=batch_sizes,
            max_new_tokens=max_new_tokens,
        )
    plan = {
        "stage": stage,
        "model_name": model_name,
        "runtime": {
            "backend": "modal",
            "model_name": model_name,
            "image": image,
            "gpu": gpu,
            "tp_size": tp_size,
        },
        "pass_modes": pass_modes or ["default"],
        "page_sizes": page_sizes or [],
        "prompt_scenarios": prompt_scenarios,
        "sampling": {"max_new_tokens": max_new_tokens},
        "supplemental_runs": _supplemental_runs(config.get("supplemental_runs")),
        "sglang": {
            "disable_cuda_graph": bool(config.get("disable_cuda_graph", True)),
            "enable_piecewise_cuda_graph": config.get("enable_piecewise_cuda_graph"),
            "force_flashinfer_backends": bool(config.get("force_flashinfer_backends", True)),
            "mem_fraction_static": float(config.get("mem_fraction_static", 0.7)),
            "cuda_graph_max_bs": config.get("cuda_graph_max_bs"),
            "engine_kwargs": _engine_kwargs(config.get("engine_kwargs")),
            "compare_tensor_logger": bool(config.get("compare_sglang_logger", True)),
        },
    }
    if reviewed_definitions is not None:
        plan["reviewed_definitions"] = reviewed_definitions
        plan["max_new_workloads"] = _positive_int(
            config.get("max_new_workloads"), "max_new_workloads"
        )
    return plan


def _sharegpt_path() -> Path:
    return DEFAULT_SHAREGPT_PATH if DEFAULT_SHAREGPT_PATH.exists() else PACKAGE_SHAREGPT_PATH


def _load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"prompt dataset does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"prompt dataset must be a JSON list: {path}")
    prompts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if not prompts:
        raise ValueError(f"prompt dataset contains no prompts: {path}")
    return prompts


def _build_prompt_scenarios(
    *, prompts: list[str], batch_sizes: list[int], max_new_tokens: int
) -> list[dict[str, Any]]:
    ranked = sorted(prompts, key=len)
    scenarios = []
    for scenario_index, batch_size in enumerate(batch_sizes):
        if batch_size <= 1:
            pool = ranked[-max(len(ranked) // 4, 1) :]
            token_limit = 96
        elif batch_size <= 4:
            start = len(ranked) // 4
            end = max(len(ranked) * 3 // 4, start + batch_size)
            pool = ranked[start : min(end, len(ranked))] or ranked
            token_limit = 64
        elif batch_size <= 16:
            pool = ranked[: max(batch_size, len(ranked) // 2, 1)]
            token_limit = 32
        else:
            pool = ranked[: max(len(ranked) // 4, 1)]
            token_limit = 8
        batch = [pool[(scenario_index * 7 + index) % len(pool)] for index in range(batch_size)]
        scenarios.append(
            {
                "name": f"batch_{batch_size}",
                "prompts": batch,
                "max_new_tokens": min(max_new_tokens, token_limit),
            }
        )
    return scenarios


def _build_inferencex_scenarios(
    *, profiles: list[str], batch_sizes: list[int], range_ratio: float, seed: int
) -> list[dict[str, Any]]:
    scenarios = []
    for profile_index, profile in enumerate(profiles):
        input_len, output_len = INFERENCEX_PROFILES[profile]
        for batch_index, batch_size in enumerate(batch_sizes):
            scenarios.append(
                {
                    "name": f"inferencex_{profile}_bs{batch_size}",
                    "source": "inferencex",
                    "profile": profile,
                    "input_len": input_len,
                    "output_len": output_len,
                    "batch_size": batch_size,
                    "range_ratio": range_ratio,
                    "seed": seed + profile_index * len(batch_sizes) + batch_index,
                }
            )
    return scenarios


def _inferencex_profiles(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ValueError("inferencex_profiles must be a non-empty list")
    profiles = []
    for index, item in enumerate(value):
        if item not in INFERENCEX_PROFILES:
            choices = ", ".join(sorted(INFERENCEX_PROFILES))
            raise ValueError(f"inferencex_profiles[{index}] must be one of: {choices}")
        if item not in profiles:
            profiles.append(item)
    return profiles


def _supplemental_runs(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("supplemental_runs must be a list")
    runs = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"supplemental_runs[{index}] must be an object")
        name = _non_empty_string(item.get("name"), f"supplemental_runs[{index}].name")
        params = item.get("sampling_params")
        if not isinstance(params, dict):
            raise ValueError(f"supplemental_runs[{index}].sampling_params must be an object")
        runs.append(
            {
                "name": name,
                "sampling_params": dict(params),
                "use_scenario_tokens": bool(item.get("use_scenario_tokens", False)),
            }
        )
    return runs


def _engine_kwargs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("engine_kwargs must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("engine_kwargs keys must be non-empty strings")
        if type(item) not in {str, int, float, bool} and item is not None:
            raise ValueError(f"engine_kwargs.{key} must be a JSON scalar")
    return dict(value)


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _ratio(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    ratio = float(value)
    if not 0 < ratio <= 1:
        raise ValueError(f"{field} must be greater than 0 and at most 1")
    return ratio


def _positive_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return [_positive_int(item, f"{field}[{index}]") for index, item in enumerate(value)]
