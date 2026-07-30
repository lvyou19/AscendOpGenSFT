"""Auto-generated benchmark file for TransformBiasRescaleQkv.

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
from typing import List
import torch
import torch.nn as nn

def _ref_transform_bias_rescale_qkv(qkv: torch.Tensor, qkv_bias: torch.Tensor, num_heads: int) -> List[torch.Tensor]:
    """Reference implementation: split qkv to q,k,v, add bias, rescale q by 1/sqrt(dim_per_head)."""
    batch, token, triple_dim = qkv.shape
    dim = triple_dim // 3
    dim_per_head = dim // num_heads
    scale = dim_per_head ** (-0.5)
    qkv_flat = (qkv + qkv_bias).float()
    qkv_flat = qkv_flat.view(batch, token, 3, num_heads, dim_per_head)
    q = qkv_flat[:, :, 0, :, :].permute(0, 2, 1, 3)
    k = qkv_flat[:, :, 1, :, :].permute(0, 2, 1, 3)
    v = qkv_flat[:, :, 2, :, :].permute(0, 2, 1, 3)
    q = (q * scale).to(qkv.dtype)
    return [q, k, v]

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, qkv: torch.Tensor, qkv_bias: torch.Tensor, num_heads: int) -> List[torch.Tensor]:
        return _ref_transform_bias_rescale_qkv(qkv, qkv_bias, num_heads)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    qkv: [batch, token, 3*dim], qkv_bias: [3*dim], num_heads: int.
    """
    batch = int(param.get('batch', 2))
    token = int(param.get('token', 128))
    dim = int(param.get('dim', 256))
    num_heads = int(param.get('num_heads', 8))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    triple_dim = 3 * dim
    qkv = torch.randn(batch, token, triple_dim, device=device, dtype=dtype)
    qkv_bias = torch.randn(triple_dim, device=device, dtype=dtype)
    return (qkv, qkv_bias, num_heads)

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
    json_path = os.path.join(os.path.dirname(__file__), "TransformBiasRescaleQkv.json")
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
