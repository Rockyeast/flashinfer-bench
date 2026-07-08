from .conftest import *


def test_capture_scope_can_limit_events_to_sampling_targets() -> None:
    sampling = ProbeTarget(
        name="sampling",
        target="flashinfer.sampling.top_k_top_p_sampling_from_probs",
        module="flashinfer.sampling",
        attr="top_k_top_p_sampling_from_probs",
        op_type="sampling",
        capture=_capture_spec(full_args=[0, 2]),
    )
    attention = ProbeTarget(
        name="attention",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        capture=_capture_spec(),
    )

    scope = {"name": "sampling_supplemental", "allowed_op_types": ["sampling"]}

    assert CaptureSession._target_allowed_by_scope(sampling, scope)
    assert not CaptureSession._target_allowed_by_scope(attention, scope)
    assert CaptureSession._target_allowed_by_scope(attention, {"name": "base"})

def test_capture_session_limits_captures_per_target_and_scope(tmp_path: Path) -> None:
    target = ProbeTarget(
        name="sampling",
        target="flashinfer.sampling.top_k_top_p_sampling_from_probs",
        module="flashinfer.sampling",
        attr="top_k_top_p_sampling_from_probs",
        op_type="sampling",
        capture=_capture_spec(full_args=[0, 2]),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[target], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=2,
    )
    session.captures_dir.mkdir(parents=True)

    for _ in range(3):
        session._write_event(target, (), {}, {"name": "base"})
    session._write_event(target, (), {}, {"name": "sampling_supplemental"})

    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    captures = sorted((tmp_path / "captures").glob("*"))

    assert len(events) == 3
    assert len(captures) == 3

def test_page_size_dispatch_keeps_only_matching_paged_target(tmp_path: Path) -> None:
    ps1 = ProbeTarget(
        name="paged_ps1",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=1,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    ps64 = ProbeTarget(
        name="paged_ps64",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=64,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[ps1, ps64], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=4,
    )
    written: list[str] = []
    session._write_event = lambda target, args, kwargs, scope=None, traced_definition=None: written.append(target.name)  # type: ignore[method-assign]

    wrapped = session._make_multi_dispatch_wrapper([ps1, ps64], lambda *args, **kwargs: "ok")
    kv_cache = (_FakeTensor((16, 64, 8, 128)), _FakeTensor((16, 64, 8, 128)))

    assert wrapped("self", "q", kv_cache) == "ok"
    assert written == ["paged_ps64"]

def test_capture_event_uses_fitrace_definition_name(tmp_path: Path) -> None:
    target = ProbeTarget(
        name="agent_preview_target",
        target="flashinfer.demo.traced",
        module="flashinfer.demo",
        attr="traced",
        definition_name="agent_preview_name",
        op_type="demo",
        backend="flashinfer",
        collect=True,
        capture=_capture_spec(),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[target], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=4,
    )
    session.captures_dir.mkdir(parents=True)
    session._write_capture_file = lambda capture_path, target, args, kwargs, scope=None, definition_name=None, traced_definition=None: capture_path  # type: ignore[method-assign]

    def original(x: int) -> int:
        return x + 1

    original.fi_trace = lambda **kwargs: {"name": "fitrace_definition_name"}  # type: ignore[attr-defined]
    wrapped = session._make_wrapper(target, original)

    assert wrapped(1) == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["definition_name"] == "fitrace_definition_name"

def test_capture_session_tags_events_inside_warmup_window(tmp_path: Path) -> None:
    module_dir = tmp_path / "warmup_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "mod.py").write_text(
        "def kernel(x):\n"
        "    return x\n"
        "\n"
        "def warmup():\n"
        "    return kernel(1)\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        import importlib

        mod = importlib.import_module("warmup_pkg.mod")

        plan = ProbePlan(
            targets=[
                ProbeTarget(
                    name="kernel",
                    target="warmup_pkg.mod.kernel",
                    module="warmup_pkg.mod",
                    attr="kernel",
                    op_type="gqa_paged",
                    capture=_capture_spec(),
                )
            ],
            skipped=[],
            warmup_hooks=[WarmupHook(name="warmup", module="warmup_pkg.mod", attr="warmup")],
        )

        output_dir = tmp_path / "probe_out"
        session = CaptureSession(probe_plan=plan, output_dir=output_dir, max_captures_per_target=1)
        session.install()
        try:
            mod.warmup()  # kernel called inside the warmup window
            mod.kernel(2)  # kernel called as a real request
        finally:
            session.uninstall()

        events = [
            json.loads(line)
            for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        warmup_flags = [event["is_warmup"] for event in events]
        assert warmup_flags == [True, False]
        assert "capture_path" not in events[0]
        assert events[1]["capture_path"].endswith("000001_kernel.pt")
        assert sorted(path.name for path in (output_dir / "captures").glob("*.pt")) == ["000001_kernel.pt"]
        # Warmup window fully unwound after the run.
        assert session.is_warmup is False
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("warmup_pkg.mod", None)
        sys.modules.pop("warmup_pkg", None)

def test_companion_capture_merges_forward_args_into_run_payload(tmp_path: Path) -> None:
    import torch

    module_dir = tmp_path / "companion_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    # `forward` is the deprecated alias: it stashes sm_scale and calls run(),
    # mirroring how FlashInfer attention wrappers behave.
    (module_dir / "mod.py").write_text(
        "class Wrapper:\n"
        "    def forward(self, q, sm_scale=None):\n"
        "        self._sm_scale = sm_scale\n"
        "        return self.run(q)\n"
        "\n"
        "    def run(self, q):\n"
        "        return q\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        import importlib

        mod = importlib.import_module("companion_pkg.mod")

        plan = ProbePlan(
            targets=[
                ProbeTarget(
                    name="decode",
                    target="companion_pkg.mod.Wrapper.run",
                    module="companion_pkg.mod",
                    attr="Wrapper.run",
                    op_type="gqa_paged",
                    companion_attrs=["forward"],
                    capture=_capture_spec(full_kwargs=["sm_scale"]),
                )
            ],
            skipped=[],
        )

        output_dir = tmp_path / "probe_out"
        session = CaptureSession(probe_plan=plan, output_dir=output_dir, max_captures_per_target=4)
        session.install()
        try:
            wrapper = mod.Wrapper()
            # SGLang passes sm_scale to forward, not run.
            wrapper.forward("dummy_q", sm_scale=0.125)
        finally:
            session.uninstall()

        # The companion hook installed and the run capture recovered sm_scale.
        companion_status = [s for s in session.install_status if s.get("kind") == "companion"]
        assert companion_status and companion_status[0]["installed"] is True

        capture_files = sorted((output_dir / "captures").glob("*.pt"))
        assert len(capture_files) == 1
        payload = torch.load(capture_files[0], weights_only=False)
        kwargs = payload["payload"]["kwargs"]
        assert kwargs["sm_scale"]["value"] == 0.125
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("companion_pkg.mod", None)
        sys.modules.pop("companion_pkg", None)
