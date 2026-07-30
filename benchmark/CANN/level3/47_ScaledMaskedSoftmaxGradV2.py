"""Auto-generated benchmark file for ScaledMaskedSoftmaxGradV2.

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

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, yGrad: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, scale: float=1.0) -> torch.Tensor:
        orig_dtype = yGrad.dtype
        yGrad_f = yGrad.to(torch.float32)
        y_f = y.to(torch.float32)
        dy_mul_y = yGrad_f * y_f
        sum_dy_y = dy_mul_y.sum(dim=-1, keepdim=True)
        xGrad_f = y_f * (yGrad_f - sum_dy_y)
        if scale != 1.0:
            xGrad_f = xGrad_f * scale
        mask_f = mask.to(torch.float32)
        xGrad_f = xGrad_f * (1.0 - mask_f)
        return xGrad_f.to(orig_dtype)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    batch = int(param['batch'])
    num_heads = int(param['num_heads'])
    seq_len = int(param['seq_len'])
    head_dim = int(param['head_dim'])
    mask_batch = int(param['mask_batch'])
    mask_num_heads = int(param['mask_num_heads'])
    dtype_str = param['dtype']
    scale = float(param['scale'])
    dtype = getattr(torch, dtype_str)
    yGrad = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    y = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    y = torch.softmax(y.to(torch.float32), dim=-1).to(dtype)
    mask = torch.zeros(mask_batch, mask_num_heads, seq_len, head_dim, device=device, dtype=torch.bool)
    mask = torch.rand(mask_batch, mask_num_heads, seq_len, head_dim, device=device) > 0.7
    return (yGrad, y, mask, scale)

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
    json_path = os.path.join(os.path.dirname(__file__), "ScaledMaskedSoftmaxGradV2.json")
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
