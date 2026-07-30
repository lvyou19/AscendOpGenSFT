"""Auto-generated benchmark file for MaskedSoftmaxWithRelPosBias.

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
import torch.nn.functional as F

def _ref_masked_softmax_with_rel_pos_bias(x: torch.Tensor, atten_mask: Optional[torch.Tensor], relative_pos_bias: torch.Tensor, scale_value: float=1.0) -> torch.Tensor:
    """Reference: out = softmax(scale_value * x + atten_mask + relative_pos_bias)."""
    scaled = x * scale_value
    out = scaled + relative_pos_bias
    if atten_mask is not None:
        out = out + atten_mask
    return F.softmax(out, dim=-1)

class Model(torch.nn.Module):

    def forward(self, x: torch.Tensor, atten_mask: Optional[torch.Tensor], relative_pos_bias: torch.Tensor, scale_value: float=1.0) -> List[torch.Tensor]:
        y = _ref_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value)
        return [y]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for MaskedSoftmaxWithRelPosBias.
    x: 4D (B*W, N, S1, S2) or 5D (B, W, N, S1, S2)
    atten_mask (optional): 3D (W, S1, S2), 4D (W, 1, S1, S2) or 5D (1, W, 1, S1, S2)
    relative_pos_bias: 3D (N, S1, S2), 4D (1, N, S1, S2) or 5D (1, 1, N, S1, S2)
    """
    x_shape = eval(param.get('x_shape', '[1, 2, 4, 8, 16]'))
    bias_shape = eval(param.get('bias_shape', '[1, 1, 4, 8, 16]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    scale_value = float(param.get('scale_value', '1.0'))
    has_attn = int(param.get('has_attn', '0'))
    x = torch.rand(x_shape, device=device, dtype=dtype)
    bias = torch.rand(bias_shape, device=device, dtype=dtype)
    if has_attn:
        if len(x_shape) == 5:
            att_shape = [x_shape[1], x_shape[3], x_shape[4]]
        else:
            att_shape = [x_shape[0] // bias_shape[0] if bias_shape[0] > 0 else 1, x_shape[2], x_shape[3]]
        atten_mask = torch.rand(att_shape, device=device, dtype=dtype)
        return (x, atten_mask, bias, scale_value)
    return (x, None, bias, scale_value)

def get_init_inputs_per_case(param, device=None):
    """No init params for this op."""
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
    json_path = os.path.join(os.path.dirname(__file__), "MaskedSoftmaxWithRelPosBias.json")
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
