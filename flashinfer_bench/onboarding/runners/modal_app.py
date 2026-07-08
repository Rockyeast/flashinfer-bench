"""Modal CLI entrypoints for streaming probe runs."""

from __future__ import annotations

import os
from typing import Any

import modal


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
    return os.environ.get("FLASHINFER_BENCH_MODAL_APP", "flashinfer-bench-onboarding-probe")


def _modal_secrets() -> list[modal.Secret]:
    names = os.environ.get("FLASHINFER_BENCH_MODAL_SECRETS", "huggingface-secret")
    return [modal.Secret.from_name(name.strip()) for name in names.split(",") if name.strip()]


app = modal.App(_modal_app_name())
image = modal.Image.from_registry(_modal_image_name()).add_local_python_source("flashinfer_bench", copy=True)
secrets = _modal_secrets()


@app.function(
    image=image,
    gpu=_modal_gpu(),
    timeout=_modal_timeout(),
    name="run_sglang_probe",
    serialized=True,
    secrets=secrets,
)
def run_sglang_probe(plan: dict[str, Any]) -> dict[str, Any]:
    from flashinfer_bench.onboarding.runners.remote_runner import run_remote_sglang_probe

    return run_remote_sglang_probe(plan)


@app.local_entrypoint()
def probe(plan_path: str, output_dir: str, resume_call_id: str = "") -> None:
    """Run a Modal probe from a local modal_probe_plan.json file."""
    from flashinfer_bench.onboarding.runners.modal_client import run_modal_probe_entrypoint

    run_modal_probe_entrypoint(
        plan_path=plan_path,
        output_dir=output_dir,
        resume_call_id=resume_call_id,
        remote_function=run_sglang_probe,
    )
