"""InferenceX fixed-sequence request generation and dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional


INFERENCEX_REVISION = "b1f05e9ac71859b83ff316e904289becf8dcd670"
INFERENCEX_GENERATOR = "utils/bench_serving/benchmark_serving.py::sample_random_requests"
INFERENCEX_RECIPE = "benchmarks/benchmark_lib.sh::run_benchmark_serving"


def inferencex_fixed_seq_request_contract() -> dict[str, Any]:
    """Describe the pinned InferenceX request semantics used by onboarding."""
    return {
        "project": "InferenceX",
        "revision": INFERENCEX_REVISION,
        "generator": INFERENCEX_GENERATOR,
        "recipe": INFERENCEX_RECIPE,
        "dataset": "random",
        "request_rate": "inf",
        "ignore_eos": True,
        "use_chat_template": False,
        "generator_workers": 1,
        "transport": "in_process_sglang_engine",
        "performance_metrics": False,
    }


@dataclass(frozen=True)
class SyntheticRequest:
    """One reproducible synthetic request after tokenizer round-tripping."""

    prompt: str
    input_ids: tuple[int, ...]
    output_len: int


def request_manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the digest of a request manifest, excluding its digest field."""
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finalize_request_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Attach a stable digest to a JSON-compatible request manifest."""
    finalized = dict(manifest)
    finalized["manifest_sha256"] = request_manifest_digest(finalized)
    return finalized


def sample_random_token_requests(
    tokenizer: Any,
    *,
    num_prompts: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    seed: int | None = None,
    prefix_len: int = 0,
) -> list[SyntheticRequest]:
    """Generate requests with InferenceX's serial random-dataset algorithm."""
    import numpy as np

    if prefix_len < 0:
        raise ValueError("prefix_len must be non-negative")
    rng = np.random.RandomState(seed)
    vocab_size = int(tokenizer.vocab_size)
    prefix_token_ids = rng.randint(0, vocab_size, size=prefix_len).tolist()

    def sample_uniform(seq_len: int) -> list[int]:
        lower = int(seq_len * range_ratio)
        return rng.randint(lower, seq_len + 1, size=num_prompts).tolist()

    input_lens = sample_uniform(input_len)
    output_lens = sample_uniform(output_len)
    offsets = rng.randint(0, vocab_size, size=num_prompts)
    local_rng = np.random.RandomState(rng.get_state()[1][:4].tolist())

    requests: list[SyntheticRequest] = []
    for index, target_len in enumerate(input_lens):
        target_prompt_len = prefix_len + target_len
        token_ids = prefix_token_ids + [
            (int(offsets[index]) + index + position) % vocab_size
            for position in range(target_len)
        ]
        prompt = tokenizer.decode(token_ids)

        # Match InferenceX's text round-trip before storing canonical token IDs.
        for _ in range(10):
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) < target_prompt_len:
                missing = target_prompt_len - len(token_ids)
                token_ids.extend(local_rng.randint(0, vocab_size, size=missing).tolist())
            elif len(token_ids) > target_prompt_len:
                token_ids = token_ids[:target_prompt_len]
            else:
                break
            prompt = tokenizer.decode(token_ids)

        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        requests.append(
            SyntheticRequest(
                prompt=prompt,
                input_ids=tuple(int(token_id) for token_id in token_ids),
                output_len=int(output_lens[index]),
            )
        )
    return requests


def sample_random_requests(
    model_path: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    seed: Optional[int] = None,
    *,
    logger: Callable[[str], None] | None = None,
) -> list[tuple[str, int, int]]:
    """Return text request tuples consumed by SGLang's serving benchmark."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generated = sample_random_token_requests(
        tokenizer,
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        range_ratio=range_ratio,
        seed=seed,
        prefix_len=0,
    )
    requests = [(item.prompt, len(item.input_ids), item.output_len) for item in generated]
    if logger is not None and requests:
        actual_in = [request[1] for request in requests]
        actual_out = [request[2] for request in requests]
        logger(
            f"Generated {num_prompts} random prompts | "
            f"input_len min/mean/max = {min(actual_in)}/"
            f"{sum(actual_in) / len(actual_in):.1f}/{max(actual_in)} | "
            f"output_len min/mean/max = {min(actual_out)}/"
            f"{sum(actual_out) / len(actual_out):.1f}/{max(actual_out)}"
        )
    return requests


@dataclass
class _TestRequest:
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[list[str]] = None
    timestamp: Optional[float] = None
    extra_request_body: dict[str, Any] = field(default_factory=dict)
    routing_key: Optional[str] = None


class _DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return []


def _bench_args(disable_ignore_eos: bool) -> SimpleNamespace:
    return SimpleNamespace(
        backend="sglang",
        dataset_name="custom",
        disable_ignore_eos=disable_ignore_eos,
        disable_stream=True,
        return_logprob=False,
        return_routed_experts=False,
        output_file=None,
        output_details=False,
        warmup_requests=0,
        plot_throughput=False,
        header=None,
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        profile_activities=["CPU", "GPU"],
        profile_num_steps=None,
        profile_by_stage=False,
        profile_stages=None,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
    )


def run_benchmark(
    base_url: str,
    prompts: list[tuple[str, int, int]],
    batch_size: int,
    temperature: float = 0.0,
    top_k: int = -1,
    top_p: float = 1.0,
    disable_ignore_eos: bool = False,
    *,
    logger: Callable[[str], None] | None = None,
) -> list[Any]:
    """Dispatch all requests at infinite request rate with bounded concurrency."""
    from sglang.bench_serving import benchmark, set_global_args

    set_global_args(_bench_args(disable_ignore_eos))
    if not prompts:
        return []

    now = time.time()
    requests = [
        _TestRequest(
            prompt=prompt,
            prompt_len=prompt_len,
            output_len=request_output_len,
            timestamp=now,
            text_prompt_len=prompt_len,
            vision_prompt_len=0,
        )
        for prompt, prompt_len, request_output_len in prompts
    ]
    if logger is not None:
        logger(f"Dispatching {len(requests)} prompts, max_concurrency={batch_size}")
    return asyncio.run(
        benchmark(
            backend="sglang",
            api_url=f"{base_url}/generate",
            base_url=base_url,
            model_id="default",
            tokenizer=_DummyTokenizer(),
            input_requests=requests,
            request_rate=float("inf"),
            max_concurrency=batch_size,
            disable_tqdm=False,
            lora_names=None,
            lora_request_distribution=None,
            lora_zipf_alpha=None,
            extra_request_body={
                "sampling_params": {
                    "temperature": temperature,
                    **({"top_k": top_k} if top_k > 0 else {}),
                    **({"top_p": top_p} if top_p < 1.0 else {}),
                }
            },
            profile=False,
        )
    )


def run_engine_requests(
    engine: Any,
    requests: list[SyntheticRequest],
    sampling_params: list[dict[str, Any]],
    *,
    max_concurrency: int,
) -> list[Any]:
    """Dispatch requests in bounded batches through an in-process SGLang Engine."""
    if len(requests) != len(sampling_params):
        raise ValueError("requests and sampling_params must have equal length")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")

    results: list[Any] = []
    for start in range(0, len(requests), max_concurrency):
        request_batch = requests[start : start + max_concurrency]
        parameter_batch = sampling_params[start : start + max_concurrency]
        result = engine.generate(
            input_ids=[list(request.input_ids) for request in request_batch],
            sampling_params=parameter_batch,
        )
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)
    return results
