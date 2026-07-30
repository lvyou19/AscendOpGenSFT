"""Auto-generated benchmark file for MoeTokenUnpermuteWithRoutingMap.

Format mirrors /home/l00899543/Ascend_evaluation/AscendOpGenAgent/benchmarks/NPUKernelBench/level1
- get_input_groups(): list of [args...] per case
- get_init_inputs():  list of [init_args...] per case (same length, [] when none)
The Model class is the source's torch reference (NPU-only imports stripped).
prepare_inputs logic is embedded so each case is generated with the same
semantics as the original validation harness; the JSON file holds one CSV row
per line as the parameter dict.
"""

import sys as _sys
import types as _types

# Stub NPU / framework deps so import-time `import torch_npu` etc. succeeds
# even when this file is run on a host without those libraries installed.
for _n in (
    "torch_npu", "torch_npu.contrib", "torch_npu.contrib.transfer_to_npu",
    "kernel_gen_ops", "framework", "framework.utils", "framework.kernel_gen_config",
):
    if _n not in _sys.modules:
        _m = _types.ModuleType(_n); _m.__path__ = []
        _sys.modules[_n] = _m

import torch
import torch.nn as nn
import json
import os

# Patch torch.randn/rand to tolerate integer dtypes on CPU (some upstream
# prepare_inputs.py call torch.randn(shape, dtype=torch.int*) which only
# works on NPU).
_INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool}
_orig_randn, _orig_rand = torch.randn, torch.rand


def _safe_make(_orig, _lo, _hi, *args, **kwargs):
    dtype = kwargs.get("dtype")
    if dtype in _INT_DTYPES:
        size = args
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            size = tuple(args[0])
        new_kwargs = dict(kwargs)
        if dtype in (torch.uint8, torch.bool):
            _lo, _hi = 0, (2 if dtype == torch.bool else 100)
        return torch.randint(_lo, _hi, tuple(int(x) for x in size), **new_kwargs)
    return _orig(*args, **kwargs)


torch.randn = lambda *a, **kw: _safe_make(_orig_randn, -100, 100, *a, **kw)
torch.rand = lambda *a, **kw: _safe_make(_orig_rand, 0, 100, *a, **kw)


# ---- Model (cleaned from source module.py) ----
from typing import List, Optional
import torch
import torch.nn as nn

class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens, sorted_indices, routing_map, probs, drop_and_pad, restore_shape):
        """
        Args:
            permuted_tokens: [num_out_tokens, hidden_size]
            sorted_indices: [total_length] - indices into permuted_tokens
            routing_map: optional [num_tokens, num_experts] bool/int8 mask
            probs: optional [num_tokens, num_experts] - expert probabilities
            drop_and_pad: bool - padded mode flag
            restore_shape: List[int] - [num_tokens] used when probs is None
        Returns:
            unpermuted_tokens: [num_tokens, hidden_size]
        """
        orig_dtype = permuted_tokens.dtype
        num_out_tokens = permuted_tokens.shape[0]
        hidden_size = permuted_tokens.shape[1]
        total_length = sorted_indices.shape[0]
        if probs is not None:
            num_tokens = probs.shape[0]
            num_experts = probs.shape[1]
            topK = num_out_tokens // num_tokens
        else:
            num_tokens = restore_shape[0] if len(restore_shape) > 0 else total_length
            topK = num_out_tokens // num_tokens
        tokens_f = permuted_tokens.float()
        output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=permuted_tokens.device)
        indices_cpu = sorted_indices.cpu()
        tokens_cpu = tokens_f.cpu()
        if probs is not None:
            probs_cpu = probs.float().cpu()
            if routing_map is not None:
                routing_map_cpu = routing_map.cpu()
        for i in range(num_tokens):
            for j in range(topK):
                idx = indices_cpu[i * topK + j].item()
                if idx < num_out_tokens:
                    token = tokens_cpu[idx]
                    if probs is not None:
                        if routing_map is not None:
                            if not routing_map_cpu[i, j].item():
                                continue
                        prob_val = probs_cpu[i, j].item()
                        if prob_val == 0:
                            continue
                        token = token * prob_val
                    output[i] += token
        return output.to(orig_dtype)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    """Generate input tensors based on test case parameters."""
    num_tokens = param.get('num_tokens')
    num_experts = param.get('num_experts')
    topK = param.get('top_k')
    hidden_size = param.get('hidden_size')
    dtype_str = param.get('dtype', 'float32')
    has_probs = param.get('has_probs', True)
    has_routing_map = param.get('has_routing_map', True)
    drop_and_pad = param.get('drop_and_pad', False)
    if dtype_str == 'float32':
        dtype = torch.float32
    elif dtype_str == 'float16':
        dtype = torch.float16
    elif dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    num_out_tokens = num_tokens * topK
    total_length = num_tokens * topK
    permuted_tokens = torch.randn(num_out_tokens, hidden_size, dtype=dtype, device=device) * 0.1
    indices = torch.randint(0, num_out_tokens, (total_length,), dtype=torch.int32, device=device)
    if has_routing_map and has_probs:
        routing_map = torch.ones(num_tokens, num_experts, dtype=torch.bool, device=device)
    else:
        routing_map = None
    if has_probs:
        probs = torch.rand(num_tokens, num_experts, dtype=dtype, device=device) * 0.5 + 0.5
    else:
        probs = None
    if not has_probs:
        restore_shape = [num_tokens]
    else:
        restore_shape = [num_tokens]
    return (permuted_tokens, indices, routing_map, probs, drop_and_pad, restore_shape)

def get_init_inputs_per_case(param, device=None):
    return []


def _coerce(v):
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s == "":
        return s
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lstrip("-+").isdigit():
        try:
            return int(s)
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        pass
    return s


def _load_cases():
    json_path = os.path.join(os.path.dirname(__file__), "MoeTokenUnpermuteWithRoutingMap.json")
    with open(json_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _try_get_inputs(case):
    coerced = {k: _coerce(v) for k, v in case.items()}
    last_err = None
    for param in (coerced, case):
        try:
            return get_inputs(param, device=None), param
        except Exception as e:
            last_err = e
    raise last_err


def get_input_groups():
    groups = []
    for case in _load_cases():
        args, _ = _try_get_inputs(case)
        groups.append(list(args) if isinstance(args, (list, tuple)) else [args])
    return groups


def get_init_inputs():
    out = []
    for case in _load_cases():
        coerced = {k: _coerce(v) for k, v in case.items()}
        try:
            init = get_init_inputs_per_case(coerced, device=None)
        except Exception:
            try:
                init = get_init_inputs_per_case(case, device=None)
            except Exception:
                init = []
        out.append(list(init) if isinstance(init, (list, tuple)) else [init])
    return out
