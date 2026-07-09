Non-FI definition JSON must match the formal Definition schema:
- axes: dynamic dimensions use {"type":"var"}; constants use {"type":"const","value":N}.
- input/output shape lists must reference axis names like "hidden_size", not raw numbers like 3072.
- scalar inputs use "shape": null, not an empty list.
- reference must define a top-level run(...).

Non-FI definition_name rules:
- Use stable dataset-style names: {op_type}_{variant}_{key_params}.
- Do not prefix definition_name with the model slug; use tags/review notes for model evidence.
- Common abbreviations: h=hidden size or heads, kv=num_kv_heads, d=head_dim, ps=page_size, i=intermediate size, v=vocab size.
- Examples: rmsnorm_h4096, fused_add_rmsnorm_h7168, silu_and_mul_i14336, top_k_sampling_from_probs_v129280.
- For rmsnorm use rmsnorm_h<N> or fused_add_rmsnorm_h<N>; for silu_and_mul use silu_and_mul_i<N>.

Non-FI source and hint rules:
- Follow the current source signature; do not add casts, defaults, or behavior unless source proves it.
- Hints are required when inputs are positional, attributes, renamed kwargs, squeezed axes, derived axes, tensor slices, or full tensor preservation.
- For module-level non-FI functions, hint arg_index starts at the first real function argument; do not add a self offset.

Minimal valid definition example:

```json
{
  "name": "rmsnorm_h3072",
  "op_type": "rmsnorm",
  "axes": {
    "tokens": {"type": "var"},
    "hidden_size": {"type": "const", "value": 3072}
  },
  "inputs": {
    "x": {"shape": ["tokens", "hidden_size"], "dtype": "bfloat16"},
    "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
    "eps": {"shape": null, "dtype": "float32"}
  },
  "outputs": {
    "out": {"shape": ["tokens", "hidden_size"], "dtype": "bfloat16"}
  },
  "reference": "import torch\n\ndef run(x, weight, eps):\n    ..."
}
```
