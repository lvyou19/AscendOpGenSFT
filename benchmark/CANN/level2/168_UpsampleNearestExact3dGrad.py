"""Auto-generated benchmark file for UpsampleNearestExact3dGrad.

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
import torch.nn.functional as F

class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, input_size: List[int], output_size: List[int], scales_d: float=0.0, scales_h: float=0.0, scales_w: float=0.0) -> torch.Tensor:
        orig_dtype = grad_output.dtype
        grad_output = grad_output.to(torch.float32)
        N, C, oD, oH, oW = grad_output.shape
        _, _, iD, iH, iW = input_size
        scale_d = iD / oD
        scale_h = iH / oH
        scale_w = iW / oW
        d_idx = torch.arange(oD, device=grad_output.device, dtype=torch.float32)
        h_idx = torch.arange(oH, device=grad_output.device, dtype=torch.float32)
        w_idx = torch.arange(oW, device=grad_output.device, dtype=torch.float32)
        d_src = ((d_idx + 0.5) * scale_d).floor().long().clamp(0, iD - 1)
        h_src = ((h_idx + 0.5) * scale_h).floor().long().clamp(0, iH - 1)
        w_src = ((w_idx + 0.5) * scale_w).floor().long().clamp(0, iW - 1)
        src_3d = d_src.unsqueeze(1).unsqueeze(2) * (iH * iW) + h_src.unsqueeze(0).unsqueeze(2) * iW + w_src.unsqueeze(0).unsqueeze(0)
        src_flat = src_3d.reshape(-1)
        grad_out_flat = grad_output.reshape(N, C, oD * oH * oW)
        src_expanded = src_flat.unsqueeze(0).unsqueeze(0).expand(N, C, -1)
        grad_input_flat = torch.zeros(N, C, iD * iH * iW, device=grad_output.device, dtype=torch.float32)
        grad_input_flat.scatter_add_(2, src_expanded, grad_out_flat)
        result = grad_input_flat.reshape(N, C, iD, iH, iW)
        if orig_dtype != torch.float32:
            result = result.to(orig_dtype)
        return result

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """Generate input tensors matching Model.forward() signature."""
    dtype = getattr(torch, param['dtype'])
    N = int(param['N'])
    C = int(param['C'])
    iD = int(param['iD'])
    iH = int(param['iH'])
    iW = int(param['iW'])
    oD = int(param['oD'])
    oH = int(param['oH'])
    oW = int(param['oW'])
    grad_output = torch.randn(N, C, oD, oH, oW, device=device, dtype=dtype)
    output_size = [N, C, oD, oH, oW]
    input_size = [N, C, iD, iH, iW]
    return [grad_output, input_size, output_size]

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
    json_path = os.path.join(os.path.dirname(__file__), "UpsampleNearestExact3dGrad.json")
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
