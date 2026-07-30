"""Auto-generated benchmark file for UpsampleTrilinear3dBackward.

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

    def forward(self, grad_output: torch.Tensor, output_size: List[int], input_size: List[int], align_corners: bool=False, scale_d: float=0.0, scale_h: float=0.0, scale_w: float=0.0) -> torch.Tensor:
        N, C, oD, oH, oW = grad_output.shape
        _, _, iD, iH, iW = input_size
        if align_corners:
            scale_d_val = (iD - 1) / (oD - 1) if oD > 1 else 0.0
            scale_h_val = (iH - 1) / (oH - 1) if oH > 1 else 0.0
            scale_w_val = (iW - 1) / (oW - 1) if oW > 1 else 0.0
        else:
            if scale_d > 0.0:
                scale_d_val = scale_d
            else:
                scale_d_val = float(iD) / oD if oD != 0 else 0.0
            if scale_h > 0.0:
                scale_h_val = scale_h
            else:
                scale_h_val = float(iH) / oH if oH != 0 else 0.0
            if scale_w > 0.0:
                scale_w_val = scale_w
            else:
                scale_w_val = float(iW) / oW if oW != 0 else 0.0
        grad_input = torch.zeros(N, C, iD, iH, iW, device=grad_output.device, dtype=grad_output.dtype)
        grad_out_f = grad_output.float()
        grad_in_f = torch.zeros(N, C, iD, iH, iW, device=grad_output.device, dtype=torch.float32)
        for d_out in range(oD):
            for h_out in range(oH):
                for w_out in range(oW):
                    if align_corners:
                        d_src = scale_d_val * d_out
                        h_src = scale_h_val * h_out
                        w_src = scale_w_val * w_out
                    else:
                        d_src = max(scale_d_val * (d_out + 0.5) - 0.5, 0.0)
                        h_src = max(scale_h_val * (h_out + 0.5) - 0.5, 0.0)
                        w_src = max(scale_w_val * (w_out + 0.5) - 0.5, 0.0)
                    d0 = int(d_src)
                    h0 = int(h_src)
                    w0 = int(w_src)
                    d1 = min(d0 + 1, iD - 1)
                    h1 = min(h0 + 1, iH - 1)
                    w1 = min(w0 + 1, iW - 1)
                    d0 = min(d0, iD - 1)
                    h0 = min(h0, iH - 1)
                    w0 = min(w0, iW - 1)
                    ld = d_src - d0 if d0 != d1 else 0.0
                    lh = h_src - h0 if h0 != h1 else 0.0
                    lw = w_src - w0 if w0 != w1 else 0.0
                    w000 = (1 - ld) * (1 - lh) * (1 - lw)
                    w001 = (1 - ld) * (1 - lh) * lw
                    w010 = (1 - ld) * lh * (1 - lw)
                    w011 = (1 - ld) * lh * lw
                    w100 = ld * (1 - lh) * (1 - lw)
                    w101 = ld * (1 - lh) * lw
                    w110 = ld * lh * (1 - lw)
                    w111 = ld * lh * lw
                    g = grad_out_f[:, :, d_out, h_out, w_out]
                    grad_in_f[:, :, d0, h0, w0] += g * w000
                    grad_in_f[:, :, d0, h0, w1] += g * w001
                    grad_in_f[:, :, d0, h1, w0] += g * w010
                    grad_in_f[:, :, d0, h1, w1] += g * w011
                    grad_in_f[:, :, d1, h0, w0] += g * w100
                    grad_in_f[:, :, d1, h0, w1] += g * w101
                    grad_in_f[:, :, d1, h1, w0] += g * w110
                    grad_in_f[:, :, d1, h1, w1] += g * w111
        return grad_in_f

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
    ac = param.get('align_corners', 0)
    align_corners = bool(ac) if not isinstance(ac, str) else ac.lower() == 'true'
    return [grad_output, output_size, input_size, align_corners]

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
    json_path = os.path.join(os.path.dirname(__file__), "UpsampleTrilinear3dBackward.json")
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
