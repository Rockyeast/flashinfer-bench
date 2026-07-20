"""Run reviewed prompt scenarios against an SGLang Engine."""

from __future__ import annotations

import inspect
import json
import os
import random
from typing import Any

_HF_CONFIG_OVERRIDE: dict[str, Any] | None = None
_HF_CONFIG_PATCHED = False


def run_sglang_model(plan: dict[str, Any]) -> None:
    """Run all SGLang passes requested by a serializable stage plan."""
    for paged, page_size in _execution_passes(plan):
        _run_sglang_pass(plan, paged=paged, page_size=page_size)


def _execution_passes(plan: dict[str, Any]) -> list[tuple[bool, int | None]]:
    modes = set(plan.get("pass_modes") or ["default"])
    page_sizes = plan.get("page_sizes")
    if not isinstance(page_sizes, list) or not page_sizes:
        page_sizes = [None]
    paged = [(True, int(size) if size is not None else None) for size in page_sizes]
    if "both" in modes:
        return [(False, None), *paged]
    passes = []
    if "default" in modes:
        passes.append((False, None))
    if "paged" in modes:
        passes.extend(paged)
    return passes or [(False, None)]


def _run_sglang_pass(plan: dict[str, Any], *, paged: bool, page_size: int | None) -> None:
    runtime = plan.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("stage plan missing runtime")
    sglang_config = plan.get("sglang") if isinstance(plan.get("sglang"), dict) else {}
    reviewed_kwargs = sglang_config.get("engine_kwargs") or {}
    if not isinstance(reviewed_kwargs, dict):
        raise ValueError("sglang.engine_kwargs must be an object")
    reviewed_kwargs = dict(reviewed_kwargs)
    hf_config_override = _pop_decrypted_config_json(reviewed_kwargs)

    enable_piecewise = sglang_config.get("enable_piecewise_cuda_graph")
    enable_piecewise = bool(paged if enable_piecewise is None else enable_piecewise)
    engine_kwargs: dict[str, Any] = {
        "model_path": str(runtime.get("model_name") or plan.get("model_name")),
        "tp_size": int(runtime.get("tp_size") or 1),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "attention_backend": "flashinfer",
        "disable_cuda_graph": bool(sglang_config.get("disable_cuda_graph", True)),
        "disable_piecewise_cuda_graph": not enable_piecewise,
        "log_level": "info",
    }
    if bool(sglang_config.get("force_flashinfer_backends", True)):
        engine_kwargs.update(
            {
                "prefill_attention_backend": "flashinfer",
                "decode_attention_backend": "flashinfer",
                "sampling_backend": "flashinfer",
            }
        )
    if enable_piecewise:
        engine_kwargs["enable_deterministic_inference"] = True
    if paged and page_size is not None:
        engine_kwargs["page_size"] = page_size
    if type(sglang_config.get("cuda_graph_max_bs")) is int:
        engine_kwargs["cuda_graph_max_bs"] = int(sglang_config["cuda_graph_max_bs"])
    if isinstance(sglang_config.get("mem_fraction_static"), (int, float)):
        engine_kwargs["mem_fraction_static"] = float(sglang_config["mem_fraction_static"])
    logger_output = sglang_config.get("debug_tensor_dump_output_folder")
    if isinstance(logger_output, str) and logger_output:
        engine_kwargs["debug_tensor_dump_output_folder"] = logger_output
        logger_layers = sglang_config.get("debug_tensor_dump_layers")
        if isinstance(logger_layers, list) and all(type(item) is int for item in logger_layers):
            engine_kwargs["debug_tensor_dump_layers"] = list(logger_layers)

    protected = {
        "model_path",
        "tp_size",
        "trust_remote_code",
        "disable_cuda_graph",
        "disable_piecewise_cuda_graph",
        "page_size",
        "debug_tensor_dump_output_folder",
        "debug_tensor_dump_layers",
    }
    overlap = sorted(protected & set(reviewed_kwargs))
    if overlap:
        raise ValueError(f"engine_kwargs cannot override managed fields: {overlap}")
    engine_kwargs.update(reviewed_kwargs)
    engine_kwargs = _filter_supported_engine_kwargs(
        engine_kwargs,
        optional={
            "disable_piecewise_cuda_graph",
            "enable_deterministic_inference",
            "cuda_graph_max_bs",
            "mem_fraction_static",
            "debug_tensor_dump_output_folder",
            "debug_tensor_dump_layers",
        },
    )

    old_paged = os.environ.get("SGLANG_FLASHINFER_USE_PAGED")
    old_mode = os.environ.get("FLASHINFER_TRACE_ACTIVE_PROBE_MODE")
    if paged:
        os.environ["SGLANG_FLASHINFER_USE_PAGED"] = "1"
        os.environ["FLASHINFER_TRACE_ACTIVE_PROBE_MODE"] = (
            f"paged_ps{page_size}" if page_size else "paged"
        )
    else:
        os.environ.pop("SGLANG_FLASHINFER_USE_PAGED", None)
        os.environ["FLASHINFER_TRACE_ACTIVE_PROBE_MODE"] = "default"
    os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        print(
            "[flashinfer_bench.onboarding] sglang pass: "
            + json.dumps(
                {"paged": paged, "page_size": page_size, "engine_kwargs": engine_kwargs},
                sort_keys=True,
            ),
            flush=True,
        )
        import sglang as sgl

        _install_hf_config_override(hf_config_override)
        engine = sgl.Engine(**engine_kwargs)
        try:
            _run_generation_requests(engine, plan)
        finally:
            _shutdown_engine(engine)
    finally:
        _restore_env("SGLANG_FLASHINFER_USE_PAGED", old_paged)
        _restore_env("FLASHINFER_TRACE_ACTIVE_PROBE_MODE", old_mode)


def _run_generation_requests(engine: Any, plan: dict[str, Any]) -> None:
    scenarios = _request_scenarios(plan)
    base = plan.get("sampling")
    if not isinstance(base, dict) or type(base.get("max_new_tokens")) is not int:
        raise ValueError("stage plan missing sampling.max_new_tokens")
    runs = [("base", dict(base), True)]
    runs.extend(
        (item["name"], item["sampling_params"], item["use_scenario_tokens"])
        for item in _supplemental_runs(plan)
    )
    for name, parameters, use_scenario_tokens in runs:
        for scenario in scenarios:
            input_ids, sampling_params = _synthetic_token_batch(
                engine,
                scenario,
                dict(parameters),
                use_scenario_tokens=use_scenario_tokens,
            )
            print(f"[flashinfer_bench.onboarding] request: {name}/{scenario['name']}", flush=True)
            engine.generate(input_ids=input_ids, sampling_params=sampling_params)


def _request_scenarios(plan: dict[str, Any]) -> list[dict[str, Any]]:
    value = plan.get("request_scenarios")
    if not isinstance(value, list) or not value:
        raise ValueError("stage plan missing request_scenarios")
    scenarios = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"request_scenarios[{index}] must be an object")
        source = str(item.get("source") or "synthetic")
        if source != "synthetic":
            raise ValueError(f"request_scenarios[{index}].source is unsupported: {source}")
        scenario = {
            "name": str(item.get("name") or f"scenario_{index + 1}"),
            "source": source,
            "input_len": _positive_int(
                item.get("input_len"), f"request_scenarios[{index}].input_len"
            ),
            "output_len": _positive_int(
                item.get("output_len"), f"request_scenarios[{index}].output_len"
            ),
            "batch_size": _positive_int(
                item.get("batch_size"), f"request_scenarios[{index}].batch_size"
            ),
            "range_ratio": _range_ratio(
                item.get("range_ratio"), f"request_scenarios[{index}].range_ratio"
            ),
            "seed": _integer(item.get("seed"), f"request_scenarios[{index}].seed"),
        }
        context_fraction = item.get("context_fraction")
        if context_fraction is not None:
            scenario["context_fraction"] = _range_ratio(
                context_fraction, f"request_scenarios[{index}].context_fraction"
            )
        shared_prefix_len = item.get("shared_prefix_len")
        if shared_prefix_len is not None:
            shared_prefix_len = _positive_int(
                shared_prefix_len, f"request_scenarios[{index}].shared_prefix_len"
            )
            if shared_prefix_len >= scenario["input_len"]:
                raise ValueError(
                    f"request_scenarios[{index}].shared_prefix_len must be smaller than input_len"
                )
            scenario["shared_prefix_len"] = shared_prefix_len
        scenarios.append(scenario)
    return scenarios


def _synthetic_token_batch(
    engine: Any, scenario: dict[str, Any], parameters: dict[str, Any], *, use_scenario_tokens: bool
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    rng = random.Random(int(scenario["seed"]))
    batch_size = int(scenario["batch_size"])
    ratio = float(scenario["range_ratio"])
    input_len = _effective_input_len(engine, scenario)
    input_lens = _sample_lengths(rng, input_len, ratio, batch_size)
    output_lens = _sample_lengths(rng, int(scenario["output_len"]), ratio, batch_size)
    vocab_size = _engine_vocab_size(engine)
    shared_prefix_len = min(int(scenario.get("shared_prefix_len") or 0), min(input_lens))
    shared_prefix = [rng.randrange(vocab_size) for _ in range(shared_prefix_len)]
    input_ids = []
    for length in input_lens:
        suffix = [rng.randrange(vocab_size) for _ in range(length - shared_prefix_len)]
        input_ids.append([*shared_prefix, *suffix])
    sampling_params = []
    for output_len in output_lens:
        item = dict(parameters)
        if use_scenario_tokens:
            item["max_new_tokens"] = output_len
        else:
            item.setdefault("max_new_tokens", output_len)
        item["ignore_eos"] = True
        sampling_params.append(item)
    return input_ids, sampling_params


def _effective_input_len(engine: Any, scenario: dict[str, Any]) -> int:
    target = int(scenario["input_len"])
    context_len = _engine_context_length(engine)
    if context_len is not None:
        target = min(target, max(context_len - int(scenario["output_len"]) - 1, 1))
    fraction = scenario.get("context_fraction")
    if fraction is None:
        return target
    if context_len is None:
        return target
    return max(min(target, int(context_len * float(fraction))), 1)


def _engine_context_length(engine: Any) -> int | None:
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    model_config = getattr(tokenizer_manager, "model_config", None)
    candidates = [model_config, getattr(model_config, "hf_config", None)]
    for candidate in candidates:
        for name in ("context_len", "context_length", "max_position_embeddings"):
            value = getattr(candidate, name, None)
            if type(value) is int and value > 0:
                return value
    return None


def _sample_lengths(rng: random.Random, target: int, ratio: float, count: int) -> list[int]:
    lower = max(int(target * ratio), 1)
    return [rng.randint(lower, target) for _ in range(count)]


def _engine_vocab_size(engine: Any) -> int:
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    model_config = getattr(tokenizer_manager, "model_config", None)
    vocab_size = getattr(model_config, "vocab_size", None)
    if type(vocab_size) is not int or vocab_size < 2:
        raise RuntimeError("SGLang Engine did not expose a valid model vocab_size")
    return vocab_size


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _range_ratio(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    ratio = float(value)
    if not 0 < ratio <= 1:
        raise ValueError(f"{field} must be greater than 0 and at most 1")
    return ratio


def _supplemental_runs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    value = plan.get("supplemental_runs")
    if not isinstance(value, list):
        raise ValueError("stage plan missing supplemental_runs")
    return [item for item in value if isinstance(item, dict)]


def _supported_engine_kwarg_names() -> set[str]:
    from sglang.srt.server_args import ServerArgs

    return set(inspect.signature(ServerArgs).parameters)


def _filter_supported_engine_kwargs(
    values: dict[str, Any], *, optional: set[str]
) -> dict[str, Any]:
    supported = _supported_engine_kwarg_names()
    unsupported = sorted(set(values) - supported)
    required = sorted(set(unsupported) - optional)
    if required:
        raise RuntimeError(f"SGLang Engine does not support kwargs: {required}")
    return {key: value for key, value in values.items() if key not in unsupported}


def _pop_decrypted_config_json(engine_kwargs: dict[str, Any]) -> dict[str, Any] | None:
    raw = engine_kwargs.pop("decrypted_config_json", None)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError("engine_kwargs.decrypted_config_json must be a non-empty JSON string")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("engine_kwargs.decrypted_config_json must decode to an object")
    return value


def _install_hf_config_override(value: dict[str, Any] | None) -> None:
    if value is None:
        return
    global _HF_CONFIG_OVERRIDE, _HF_CONFIG_PATCHED
    _HF_CONFIG_OVERRIDE = value
    if _HF_CONFIG_PATCHED:
        return
    from transformers import PretrainedConfig

    original = PretrainedConfig.from_dict.__func__

    @classmethod
    def patched(cls, config_dict: dict[str, Any], **kwargs: Any) -> Any:
        return original(cls, dict(_HF_CONFIG_OVERRIDE or config_dict), **kwargs)

    PretrainedConfig.from_dict = patched
    _HF_CONFIG_PATCHED = True


def _shutdown_engine(engine: Any) -> None:
    for name in ("shutdown", "release", "close"):
        method = getattr(engine, name, None)
        if callable(method):
            method()
            return


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
