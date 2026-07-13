"""Modal platform boundary for onboarding stages."""

from __future__ import annotations

import os
from typing import Any

import modal


HF_CACHE_ROOT = "/mnt/hf-cache"


def _modal_image_name() -> str:
    return os.environ.get("FLASHINFER_BENCH_MODAL_IMAGE", "lmsysorg/sglang:v0.5.12.post1")


def _modal_gpu() -> str | None:
    value = os.environ.get("FLASHINFER_BENCH_MODAL_GPU", "").strip()
    return value or None


def _modal_timeout() -> int:
    value = os.environ.get("FLASHINFER_BENCH_MODAL_TIMEOUT", "3600")
    try:
        return max(1, int(value))
    except ValueError:
        return 3600


def _modal_app_name() -> str:
    return os.environ.get("FLASHINFER_BENCH_MODAL_APP", "flashinfer-bench-onboarding")


def _modal_hf_cache_volume_name() -> str:
    return os.environ.get(
        "FLASHINFER_BENCH_MODAL_HF_CACHE_VOLUME", "flashinfer-bench-hf-cache"
    )


def _modal_secrets() -> list[modal.Secret]:
    names = os.environ.get("FLASHINFER_BENCH_MODAL_SECRETS", "huggingface-secret")
    return [modal.Secret.from_name(name.strip()) for name in names.split(",") if name.strip()]


app = modal.App(_modal_app_name())
image = (
    modal.Image.from_registry(_modal_image_name())
    .env(
        {
            "HF_HOME": HF_CACHE_ROOT,
            "HF_HUB_CACHE": f"{HF_CACHE_ROOT}/hub",
        }
    )
    .add_local_python_source("flashinfer_bench", copy=True)
)
secrets = _modal_secrets()
hf_cache_volume = modal.Volume.from_name(
    _modal_hf_cache_volume_name(), create_if_missing=True
)


@app.function(
    image=image,
    gpu=_modal_gpu(),
    timeout=_modal_timeout(),
    name="run_onboarding_stage",
    serialized=True,
    secrets=secrets,
    volumes={HF_CACHE_ROOT: hf_cache_volume},
)
def run_onboarding_stage(plan: dict[str, Any]) -> dict[str, Any]:
    from flashinfer_bench.onboarding.runners.remote_pipeline import run_remote_stage

    return run_remote_stage(plan)


@app.local_entrypoint()
def stage(plan_path: str, output_dir: str, resume_call_id: str = "") -> None:
    """Submit one onboarding stage and materialize its result."""
    from flashinfer_bench.onboarding.runners.modal_client import run_modal_stage_entrypoint

    run_modal_stage_entrypoint(
        plan_path=plan_path,
        output_dir=output_dir,
        resume_call_id=resume_call_id,
        remote_function=run_onboarding_stage,
    )
