# Reference Test Task

Create independent pytest reference tests for the listed non-FlashInfer Definitions.

Rules:

- Write only `test_<definition_name>.py` files under the requested tests directory.
- Load and execute the Definition's `reference.run()` as one side of the comparison.
- The other side must call the exact source-backed SGLang module/callable, or an independent
  trusted PyTorch implementation derived from that source.
- Never copy the Definition reference into the test as the independent oracle.
- Use the reviewed capture tags and SGLang execution inventory to identify the callable and
  input mapping. Do not invent source evidence.
- Generate valid inputs that satisfy all constant axes and constraints.
- Construct each input with the dtype declared by the Definition and accepted by the
  source callable. Do not force all tensors to the model activation dtype; current SGLang
  rotary CUDA kernels require a `float32` cosine-sine cache.
- Compare every declared output with `torch.testing.assert_close`.
- Mark CUDA-dependent tests with a pytest CUDA skip condition.
- Do not edit definitions, output workloads, reports, repository source, or runtime config.
- Do not run Modal.

Human review is still required before submission.
